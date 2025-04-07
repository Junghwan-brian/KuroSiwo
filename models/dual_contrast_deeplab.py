import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from collections.abc import Iterable
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading
from segmentation_models_pytorch.base import SegmentationHead
import numpy as np
from typing import Any, Optional, Union, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md


class ProjectionHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # L2 정규화 (channel 차원, dim=1)
        x = F.normalize(x, p=2, dim=1)
        return x


class DualContrastDeepLabV3(SegmentationModel):
    """
    DualContrastDeepLabV3 SMP DualContrastDeepLabV3을 상속받아,
      - 각 시점(pre_event1, pre_event2, post_event)의 SAR 입력(채널=2)을 처리하고,
      - 공통 encoder를 통해 각 시점의 feature를 추출한 후,  
        최종 segmentation map은 세 시점의 feature를 channel-wise concat하여 decoder에 전달합니다.
      - 또한, contrastive loss 계산에 사용할 feature를 (예: encoder의 중간 레벨에 대해 projection head 적용 후)
        label 해상도에 맞게 upsample하여 반환합니다.
      - 모델 외부에서 반환된 feature와 label을 이용해 contrastive loss(예: dual contrastive loss)를 계산하며,
        positive anchor는 even_anchor_sampling() 함수를 통해 클래스별 균등하게 샘플링할 수 있습니다.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: Literal[3, 4, 5] = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: Literal[8, 16] = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: Iterable[int] = (12, 24, 36),
        decoder_aspp_separable: bool = True,
        decoder_aspp_dropout: float = 0.5,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError(
                "DeeplabV3Plus support output stride 8 or 16, got {}.".format(
                    encoder_output_stride
                )
            )

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
            **kwargs,
        )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=[c*3 for c in self.encoder.out_channels],
            encoder_depth=encoder_depth,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
            aspp_separable=decoder_aspp_separable,
            aspp_dropout=decoder_aspp_dropout,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            from segmentation_models_pytorch.base import ClassificationHead
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        # 추가: 각 시점의 contrastive feature를 위한 projection head
        # 여기서는 encoder의 마지막 레벨 이전 feature(예: layer3 출력, 채널 수 encoder.out_channels[-2])를 사용한다고 가정
        self.contrast_proj = ProjectionHead(self.encoder.out_channels[-2])

        self.name = f"dual-contrast-unet-{encoder_name}"
        self.initialize()

    def forward(self, pre_event1: torch.Tensor, pre_event2: torch.Tensor, post_event: torch.Tensor, eval_mode=False):
        """
        입력:
          pre_event1, pre_event2, post_event: 각각 [B, 2, H, W]
        출력:
          seg_logits: [B, classes, H, W] (원본 해상도)
          contrast_feats: dict with keys 'pre1', 'pre2', 'post'
                          each of shape [B, 64, H, W] (upsampled contrastive feature)
        """
        # 각 시점별로 encoder를 통과 (SMP encoder는 보통 skip connection list를 반환)
        # 여기서는 예시로 encoder(x)로부터 여러 레벨 feature list를 받고,
        # - segmentation에는 decoder가 skip connection을 활용하도록,
        # - contrastive feature에는 encoder의 중간 레벨(예: -2)을 사용합니다.
        feats_pre1 = self.encoder(pre_event1)  # list, 마지막은 가장 낮은 해상도
        feats_pre2 = self.encoder(pre_event2)
        feats_post = self.encoder(post_event)

        # decoder expects a list of features; 여기서는 간단히 [concat_feat]로 전달
        seg_feature = []  # 실제 구현에서는 decoder의 skip connections도 필요함.
        for i in range(len(feats_pre1)):
            feat_pre1_last = feats_pre1[i]  # [B, C1, h, w]
            feat_pre2_last = feats_pre2[i]  # [B, C2, h, w]
            feat_post_last = feats_post[i]  # [B, C3, h, w]
            concat_feat = torch.cat(
                [feat_pre1_last, feat_pre2_last, feat_post_last], dim=1)
            seg_feature.append(concat_feat)  # [B, C1+C2+C3, h, w]
        seg_logits = self.decoder(*(seg_feature))
        seg_logits = self.segmentation_head(seg_logits)
        # upsample segmentation to input resolution
        seg_logits = F.interpolate(
            seg_logits, size=pre_event1.shape[2:], mode='bilinear', align_corners=True)

        # Contrastive features: encoder의 중간 레벨 (예: -2 index)을 projection
        feat_pre1_con = self.contrast_proj(feats_pre1[-2])  # [B, 64, h2, w2]
        feat_pre2_con = self.contrast_proj(feats_pre2[-2])
        feat_post_con = self.contrast_proj(feats_post[-2])
        # upsample contrastive features to match label resolution (input H, W)
        feat_pre1_con = F.interpolate(
            feat_pre1_con, size=pre_event1.shape[2:], mode='bilinear', align_corners=True)
        feat_pre2_con = F.interpolate(
            feat_pre2_con, size=pre_event1.shape[2:], mode='bilinear', align_corners=True)
        feat_post_con = F.interpolate(
            feat_post_con, size=pre_event1.shape[2:], mode='bilinear', align_corners=True)

        if eval_mode:
            return seg_logits

        return seg_logits, feat_pre1_con, feat_pre2_con, feat_post_con
