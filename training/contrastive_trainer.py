from scipy.ndimage import distance_transform_edt
import numpy as np
import json
from pathlib import Path

import kornia
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from models.model_utilities import *
from utilities.utilities import *

CLASS_LABELS = {0: "No water", 1: "Permanent Waters",
                2: "Floods", 3: "Invalid pixels"}


def train_contrastive_semantic_segmentation(
    model, train_loader, val_loader, test_loader, configs, model_configs
):

    # Accuracy, loss, optimizer, lr scheduler
    accuracy, fscore, precision, recall, iou = initialize_metrics(configs)
    # TODO 단순 CrossEntropyLoss 말고 ce+dice 도 테스트.
    criterion = create_loss(configs, mode="train")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_configs["learning_rate"])
    lr_scheduler = init_lr_scheduler(  # TODO SAM optimizer 추가 테스트.
        optimizer, configs, model_configs, steps=len(train_loader)
    )

    num_classes = len(CLASS_LABELS) - 1
    sample_anchor_per_class = 10
    sample_anchor = 10

    model.to(configs["device"])
    best_val = 0.0
    best_stats = {}

    if configs["mixed_precision"]:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(configs["epochs"]):
        model.train()

        train_loss = 0.0

        for index, batch in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc="Epoch " + str(epoch),
        ):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=configs["mixed_precision"]):
                if configs["scale_input"] is not None:
                    (
                        image_scale_var_1,
                        image_scale_var_2,
                        post_event,
                        mask,
                        pre_scale_var_1,
                        pre_scale_var_2,
                        pre_event_1,
                        pre2_scale_var_1,
                        pre2_scale_var_2,
                        pre_event_2,
                        clz,
                        activation,
                    ) = batch

                else:
                    post_event, mask, pre_event_1, pre_event_2, clz, activation = batch

                post_event = post_event.to(configs["device"])
                mask = mask.to(configs["device"])

                pre_event_1 = pre_event_1.to(configs["device"])
                pre_event_2 = pre_event_2.to(configs["device"])
                logits, feat_pre1_up, feat_pre2_up, feat_post_up = model(
                    pre_event_1, pre_event_2, post_event)

                predictions = logits.argmax(1)
                ce_loss = criterion(logits, mask)

                B = post_event.size(0)
                changed_loss_sum = 0.0
                global_loss_sum = 0.0

                # 전체 배치에 대해 contrastive loss 계산
                for b in range(B):
                    # (A) Changed Contrastive Loss: pre_event1에서 Permanent Waters(1) -> post_event에서 Floods(2)
                    changed_loss_sum += changed_contrastive_loss(feat_pre1_up[b], feat_pre2_up[b], feat_post_up[b],
                                                                 mask[b], temperature=0.07, sample_anchor=sample_anchor)
                    # (B) Global Contrastive Loss: 세 시점의 feature concat
                    global_feat = torch.cat([feat_pre1_up[b].unsqueeze(0),
                                            feat_pre2_up[b].unsqueeze(0),
                                            feat_post_up[b].unsqueeze(0)], dim=1).squeeze(0)  # [192, H, W]
                    # BANE sampling: post-event 예측 결과를 사용 (각 이미지 별)
                    post_pred = torch.argmax(F.interpolate(logits[b].unsqueeze(0),
                                                           size=(
                                                               mask.shape[1], mask.shape[2]),
                                                           mode='bilinear', align_corners=True), dim=1)  # [1,H,W]
                    bane_neg = bane_sampling_single_image(
                        post_pred[0], mask[b], num_classes, ratio_k=0.5)
                    global_loss_sum += global_contrastive_loss(global_feat, mask[b], bane_neg,
                                                               num_classes=num_classes, temperature=0.07,
                                                               sample_anchor_per_class=sample_anchor_per_class)

                changed_loss_avg = changed_loss_sum / B
                global_loss_avg = global_loss_sum / B

                total_loss = ce_loss + 0.01*changed_loss_avg + 0.5*global_loss_avg

            if configs["mixed_precision"]:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            acc = accuracy(predictions, mask)
            score = fscore(predictions, mask)
            prec = precision(predictions, mask)
            rec = recall(predictions, mask)
            ious = iou(predictions, mask)
            mean_iou = (ious[0] + ious[1] + ious[2]) / 3

            if index % configs["print_frequency"] == 0 and configs["on_screen_prints"]:
                print(
                    f"Epoch: {epoch}, Iteration: {index}, Train Loss: {total_loss.item()}")
                for i, label in CLASS_LABELS.items():
                    if i == 3:
                        continue
                    print(
                        f"Train Accuracy ({label}): {100 * acc[i].item()}")
                    print(
                        f"Train F-Score ({label}): {100 * score[i].item()}")
                    print(
                        f"Train Precision ({label}): {100 * prec[i].item()}")
                    print(f"Train Recall ({label}): {100 * rec[i].item()}")
                    print(f"Train IoU ({label}): {100 * ious[i].item()}")
                print(f"Train MeanIoU: {mean_iou * 100}")
                print(f"lr: {lr_scheduler.get_last_lr()[0]}")

        # Update LR scheduler
        lr_scheduler.step()

        # Evaluate on validation set
        model.eval()
        val_acc, val_score, miou = eval_contrastive_semantic_segmentation(
            model,
            val_loader,
            settype="Val",
            configs=configs,
            model_configs=model_configs,
        )

        if miou > best_val:
            print("Epoch: ", epoch)
            print("New best validation mIOU: ", miou)
            print(
                "Saving model to: ",
                configs["checkpoint_path"] + "/" + "best_segmentation.pt",
            )
            best_val = miou
            best_stats["miou"] = best_val
            best_stats["epoch"] = epoch
            torch.save(model, configs["checkpoint_path"] +
                       "/" + "best_segmentation.pt")


def bane_sampling_single_image(
    pred: torch.Tensor,  # [H, W], 예측 클래스(0~C-1)
    gt: torch.Tensor,    # [H, W], 실제 클래스
    num_classes: int = 3,
    ratio_k: float = 0.5
):
    """
    pred, gt는 단일 이미지를 가정 (within-image)
    BANE: boundary 근처 오분류 픽셀을 negative로 선택

    Returns:
        negative_indices = dict: { class_id : Tensor of shape [K,2] (row,col) }
        (K개의 (row,col) 픽셀 좌표)


    Explanation:
    ------------
    1.	에러 마스크(error mask) 만들기
    •	“클래스 c여야 하는 픽셀”인데 “모델이 c로 예측하지 않은” 곳을 에러로 봅니다.
    •	예: \text{error_mask}(u,v) = 1 if (GT=c) and (Pred≠c), 그 외는 0
    2.	에러 마스크에서 “경계와의 거리” 계산
    •	distance_transform_edt(SciPy 함수)를 이용해, 에러 마스크가 1인 지점들(오분류된 영역) 각각이 “경계”로부터 얼마나 떨어져 있는지를 계산합니다.
    •	“경계”란, 에러 마스크가 1에서 0으로 바뀌는 지점(즉 잘못 예측된 영역의 테두리)이라고 생각하면 됩니다.
    3.	경계에 가까운 픽셀(작은 거리)을 골라냄
    •	distance가 작을수록 경계와 가까운 픽셀이 됩니다.
    •	BANE에서는 distance가 작은 픽셀들을 우선적으로 선택(예: 상위 K%만 선택)하여, “경계 부근의 하드 네거티브” 샘플로 삼습니다.

    """
    pred_np = pred.detach().cpu().long().numpy()
    gt_np = gt.detach().cpu().long().numpy()

    H, W = pred_np.shape
    neg_indices = {}

    for c in range(num_classes):
        # 1) error mask: GT는 c지만 pred가 c가 아님
        error_mask = (gt_np == c) & (pred_np != c)  # bool [H,W]

        if not np.any(error_mask):
            neg_indices[c] = torch.empty((0, 2), dtype=torch.long)
            continue

        # 2) distance transform (경계로부터의 거리)
        #   -> boundary에 가까울수록 distance가 0 ~ small
        dist_map = distance_transform_edt(~error_mask)  # [H,W] float

        # 3) error_mask가 1인 픽셀들 중 distance가 작은 순으로 정렬
        error_coords = np.stack(np.where(error_mask), axis=1)  # [N,2]
        dist_vals = dist_map[error_mask]                    # [N]

        N_err = len(error_coords)
        kth = int(N_err * ratio_k)
        kth = max(kth, 1)  # 최소 1픽셀 이상
        sorted_idx = np.argsort(dist_vals)  # ascending
        selected = sorted_idx[:kth]       # 상위 k% -> boundary 근처

        coords_torch = torch.from_numpy(error_coords[selected]).long()
        neg_indices[c] = coords_torch

    return neg_indices


def changed_contrastive_loss(feat_pre1, feat_pre2, feat_post, post_label, temperature=0.07, sample_anchor=100):
    """
    변화를 나타내는 픽셀 (no water -> Floods)에 대해 InfoNCE 스타일 contrastive loss 계산.
    가정: 클래스: 0:"No water", 1:"Permanent Waters", 2:"Floods", 3:"Invalid pixels"
    변화 영역: pre_event1에서 'no water'(0)이고, post_event에서 'Floods'(2)인 픽셀.

    각 changed 픽셀 위치에서 세 시점의 동일 위치 임베딩을 양성으로 간주하여 loss 계산.
    (여기서는 negatives는 고려하지 않는 단순 버전)

    feat_*: [C, H, W] (upsample된 contrastive feature)
    post_label: [H, W]
    """
    changed_mask = ((post_label == 2))
    if changed_mask.sum() == 0:
        return torch.tensor(0.0, device=feat_pre1.device)
    coords = torch.stack(torch.where(changed_mask), dim=1)  # [N,2]
    N = coords.shape[0]
    if N > sample_anchor:
        perm = torch.randperm(N)[:sample_anchor]
        coords = coords[perm]
        N = sample_anchor
    loss = 0.0
    for (i, j) in coords:
        anchor = feat_post[:, i, j]
        pos1 = feat_pre1[:, i, j]
        pos2 = feat_pre2[:, i, j]
        # 유사도가 높을수록 (즉, dot product가 크면) loss가 커지도록 함
        sim1 = torch.dot(anchor, pos1) / temperature
        sim2 = torch.dot(anchor, pos2) / temperature
        loss_anchor = (sim1 + sim2) / 2.0
        loss += loss_anchor
    return loss / N


def global_contrastive_loss(global_feat, gt, bane_neg_fn, num_classes, temperature=0.07, sample_anchor_per_class=50):
    """
    global_feat: [C_global, H, W] – 세 시점의 contrastive feature를 concat한 결과 ([192, H, W] if each is 64-dim)
    gt: [H, W] – ground truth (post-event 기준)
    bane_neg_fn: BANE sampling 결과 (dict {class: Tensor([K,2])})

    각 클래스별로 일정 수의 anchor 픽셀을 균등하게 샘플링하여, 
    해당 클래스의 양성 집합과 BANE로 선정된 네거티브 집합을 사용해 InfoNCE loss를 계산합니다.
    """
    loss = 0.0
    count = 0
    for cls_ in range(num_classes):
        # invalid pixels(3)는 계산에서 제외
        if cls_ == 3:
            continue
        class_mask = (gt == cls_)
        coords = torch.stack(torch.where(class_mask), dim=1)  # [N,2]
        if coords.shape[0] == 0:
            continue
        N = coords.shape[0]
        if N > sample_anchor_per_class:
            perm = torch.randperm(N)[:sample_anchor_per_class]
            class_coords = coords[perm]
        else:
            class_coords = coords

        for (i, j) in class_coords:
            anchor_feat = global_feat[:, i, j]
            pos_coords = torch.stack(torch.where(class_mask), dim=1)
            if pos_coords.shape[0] < 2:
                continue
            pos_feats = global_feat[:, pos_coords[:, 0], pos_coords[:, 1]]
            neg_coords_list = []
            for c, neg_coords in bane_neg_fn.items():
                if c != cls_ and c != 3 and neg_coords.shape[0] > 0:
                    neg_coords_list.append(neg_coords)
            if len(neg_coords_list) == 0:
                continue
            neg_coords = torch.cat(neg_coords_list, dim=0)
            neg_feats = global_feat[:, neg_coords[:, 0], neg_coords[:, 1]]
            anchor_feat_2d = anchor_feat.unsqueeze(1)
            pos_dot = torch.mm(anchor_feat_2d.t(), pos_feats) / \
                temperature  # [1, P]
            neg_dot = torch.mm(anchor_feat_2d.t(), neg_feats) / \
                temperature  # [1, N_neg]
            pos_exp = torch.exp(pos_dot)
            neg_exp = torch.exp(neg_dot)
            loss_anchor = -torch.log((pos_exp.sum() + 1e-10) /
                                     (pos_exp.sum() + neg_exp.sum() + 1e-10))
            loss += loss_anchor
            count += 1
    return loss / count if count > 0 else torch.tensor(0.0, device=global_feat.device)


def eval_contrastive_semantic_segmentation(
    model, loader, configs=None, settype="Test", model_configs=None
):
    accuracy, fscore, precision, recall, iou = initialize_metrics(
        configs, mode="val")
    if configs["evaluate_water"]:
        water_fscore = F1Score(
            task="multiclass",
            num_classes=2,
            average="none",
            multidim_average="global",
            ignore_index=3,
        ).to(configs["device"])
    if configs["log_zone_metrics"]:
        (
            accuracy_clzone1,
            fscore_clzone1,
            precision_clzone1,
            recall_clzone1,
            iou_clzone1,
        ) = initialize_metrics(configs, mode="val")
        (
            accuracy_clzone2,
            fscore_clzone2,
            precision_clzone2,
            recall_clzone2,
            iou_clzone2,
        ) = initialize_metrics(configs, mode="val")
        (
            accuracy_clzone3,
            fscore_clzone3,
            precision_clzone3,
            recall_clzone3,
            iou_clzone3,
        ) = initialize_metrics(configs, mode="val")

    if configs["log_AOI_metrics"]:
        activ_metrics = {
            activ: initialize_metrics(configs, mode="val")
            for activ in loader.dataset.activations
        }
        if configs["evaluate_water"]:
            water_only_metrics = {
                activ: F1Score(
                    task="multiclass",
                    num_classes=2,
                    average="none",
                    multidim_average="global",
                    ignore_index=3,
                ).to(configs["device"])
                for activ in loader.dataset.activations
            }

    model.to(configs["device"])
    criterion = create_loss(configs, mode="val")

    first_image = []
    first_mask = []
    first_prediction = []
    total_loss = 0.0

    samples_per_clzone = {1: 0, 2: 0, 3: 0}
    random_index = 0
    for index, batch in tqdm(enumerate(loader), total=len(loader)):
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                if configs["scale_input"] is not None:
                    (
                        image_scale_var_1,
                        image_scale_var_2,
                        post_event,
                        mask,
                        pre_scale_var_1,
                        pre_scale_var_2,
                        pre_event_1,
                        pre2_scale_var_1,
                        pre2_scale_var_2,
                        pre_event_2,
                        clz,
                        activ,
                    ) = batch

                else:
                    post_event, mask, pre_event_1, pre_event_2, clz, activ = batch

                post_event = post_event.to(configs["device"])
                mask = mask.to(configs["device"])

                pre_event_1 = pre_event_1.to(configs["device"])
                pre_event2 = pre_event2.to(configs["device"])
                output = model(pre_event_1, pre_event_2,
                               post_event, eval_mode=True)

                loss = criterion(output, mask)
                total_loss += loss.item() * post_event.size(0)
                predictions = output.argmax(1)

                water_only_predictions = predictions.clone()
                water_only_predictions[water_only_predictions == 2] = 1
                water_only_labels = mask.clone()
                water_only_labels[water_only_labels == 2] = 1
                water_fscore(water_only_predictions, water_only_labels)

                if index == random_index:
                    first_image = post_event.detach().cpu()[0]
                    pre_event_wand = pre_event_1.detach().cpu()[0]
                    pre_event_2_wand = pre_event_2.detach().cpu()[0]

                    first_mask = mask.detach().cpu()[0]
                    first_prediction = predictions.detach().cpu()[0]

                    if configs["scale_input"] is not None:
                        image_scale_vars = [
                            image_scale_var_1[0], image_scale_var_2[0]]
                        pre_scale_vars = [
                            pre_scale_var_1[0], pre_scale_var_2[0]]
                        pre2_scale_vars = [
                            pre2_scale_var_1[0], pre2_scale_var_2[0]]

                accuracy(predictions, mask)
                fscore(predictions, mask)
                precision(predictions, mask)
                recall(predictions, mask)
                iou(predictions, mask)

                if configs["log_zone_metrics"]:
                    clz_in_batch = torch.unique(clz)

                    if 1 in clz_in_batch:
                        accuracy_clzone1(
                            predictions[clz == 1, :, :], mask[clz == 1, :, :]
                        )
                        fscore_clzone1(
                            predictions[clz == 1, :, :], mask[clz == 1, :, :]
                        )
                        precision_clzone1(
                            predictions[clz == 1, :, :], mask[clz == 1, :, :]
                        )
                        recall_clzone1(
                            predictions[clz == 1, :, :], mask[clz == 1, :, :]
                        )
                        iou_clzone1(
                            predictions[clz == 1, :, :], mask[clz == 1, :, :])
                        samples_per_clzone[1] += predictions[clz ==
                                                             1, :, :].shape[0]

                    if 2 in clz_in_batch:
                        accuracy_clzone2(
                            predictions[clz == 2, :, :], mask[clz == 2, :, :]
                        )
                        fscore_clzone2(
                            predictions[clz == 2, :, :], mask[clz == 2, :, :]
                        )
                        precision_clzone2(
                            predictions[clz == 2, :, :], mask[clz == 2, :, :]
                        )
                        recall_clzone2(
                            predictions[clz == 2, :, :], mask[clz == 2, :, :]
                        )
                        iou_clzone2(
                            predictions[clz == 2, :, :], mask[clz == 2, :, :])
                        samples_per_clzone[2] += predictions[clz ==
                                                             2, :, :].shape[0]

                    if 3 in clz_in_batch:
                        accuracy_clzone3(
                            predictions[clz == 3, :, :], mask[clz == 3, :, :]
                        )
                        fscore_clzone3(
                            predictions[clz == 3, :, :], mask[clz == 3, :, :]
                        )
                        precision_clzone3(
                            predictions[clz == 3, :, :], mask[clz == 3, :, :]
                        )
                        recall_clzone3(
                            predictions[clz == 3, :, :], mask[clz == 3, :, :]
                        )
                        iou_clzone3(
                            predictions[clz == 3, :, :], mask[clz == 3, :, :])
                        samples_per_clzone[3] += predictions[clz ==
                                                             3, :, :].shape[0]

                if configs["log_AOI_metrics"]:
                    activs_in_batch = torch.unique(activ)

                    for activ_i in [i.item() for i in activs_in_batch]:
                        activ_metrics[activ_i][0](
                            predictions[activ == activ_i, :, :],
                            mask[activ == activ_i, :, :],
                        )  # accuracy
                        activ_metrics[activ_i][1](
                            predictions[activ == activ_i, :, :],
                            mask[activ == activ_i, :, :],
                        )  # fscore
                        activ_metrics[activ_i][2](
                            predictions[activ == activ_i, :, :],
                            mask[activ == activ_i, :, :],
                        )  # precision
                        activ_metrics[activ_i][3](
                            predictions[activ == activ_i, :, :],
                            mask[activ == activ_i, :, :],
                        )  # recall
                        activ_metrics[activ_i][4](
                            predictions[activ == activ_i, :, :],
                            mask[activ == activ_i, :, :],
                        )  # iou

                        if configs["evaluate_water"]:
                            water_only_metrics[activ_i](
                                water_only_predictions[activ == activ_i, :, :],
                                water_only_labels[activ == activ_i, :, :],
                            )

    # Calculate average loss over an epoch
    val_loss = total_loss / len(loader)
    if configs["wandb_activate"]:
        mask_example = first_mask
        prediction_example = first_prediction

        # Reverse image scaling for visualization purposes
        if (
            configs["scale_input"] not in [None, "custom"]
            and configs["reverse_scaling"]
        ):
            pre_event_wand = reverse_scale_img(
                pre_event_wand, pre_scale_vars[0], pre_scale_vars[1], configs
            )
            pre_event_2_wand = reverse_scale_img(
                pre_event_2_wand, pre2_scale_vars[0], pre2_scale_vars[1], configs
            )
            first_image = reverse_scale_img(
                first_image, image_scale_vars[0], image_scale_vars[1], configs
            )

        first_image = kornia.enhance.adjust_gamma(first_image, gamma=0.3)
        pre_event_wand = kornia.enhance.adjust_gamma(pre_event_wand, gamma=0.3)
        pre_event_2_wand = kornia.enhance.adjust_gamma(
            pre_event_2_wand, gamma=0.3)

        if (
            model_configs["architecture"] == "vivit"
            or model_configs["architecture"] == "convlstm"
        ):
            first_image = first_image[2]
            pre_event_wand = pre_event_wand[1]

        mask_img = wandb.Image(
            (first_image[0] * 255).int().cpu().detach().numpy(),
            masks={
                "predictions": {
                    "mask_data": prediction_example.float().numpy(),
                    "class_labels": CLASS_LABELS,
                },
                "ground_truth": {
                    "mask_data": mask_example.float().numpy(),
                    "class_labels": CLASS_LABELS,
                },
            },
        )
        mask_img_preevent_1 = wandb.Image(
            (pre_event_wand[0] * 255).int().cpu().detach().numpy(),
            masks={
                "predictions": {
                    "mask_data": prediction_example.float().numpy(),
                    "class_labels": CLASS_LABELS,
                },
                "ground_truth": {
                    "mask_data": mask_example.float().numpy(),
                    "class_labels": CLASS_LABELS,
                },
            },
        )
        mask_img_preevent_2 = wandb.Image(
            (pre_event_2_wand[0] * 255).int().cpu().detach().numpy(),
            masks={
                "predictions": {
                    "mask_data": prediction_example.float().numpy(),
                    "class_labels": CLASS_LABELS,
                },
                "ground_truth": {
                    "mask_data": mask_example.float().numpy(),
                    "class_labels": CLASS_LABELS,
                },
            },
        )
        wandb.log({settype + " Flood Masks ": mask_img})
        wandb.log({settype + " Pre-event_1 Masks ": mask_img_preevent_1})
        wandb.log({settype + " Pre-event_2 Masks ": mask_img_preevent_2})

    acc = accuracy.compute()
    score = fscore.compute()
    prec = precision.compute()
    rec = recall.compute()
    ious = iou.compute()
    mean_iou = ious[:3].mean()
    if configs["evaluate_water"]:
        water_total_fscore = water_fscore.compute()

    if configs["log_zone_metrics"]:
        acc_clz1 = accuracy_clzone1.compute()
        score_clz1 = fscore_clzone1.compute()
        prec_clz1 = precision_clzone1.compute()
        rec_clz1 = recall_clzone1.compute()
        ious_clz1 = iou_clzone1.compute()
        mean_iou_clz1 = ious_clz1[:3].mean()

        acc_clz2 = accuracy_clzone2.compute()
        score_clz2 = fscore_clzone2.compute()
        prec_clz2 = precision_clzone2.compute()
        rec_clz2 = recall_clzone2.compute()
        ious_clz2 = iou_clzone2.compute()
        mean_iou_clz2 = ious_clz2[:3].mean()

        acc_clz3 = accuracy_clzone3.compute()
        score_clz3 = fscore_clzone3.compute()
        prec_clz3 = precision_clzone3.compute()
        rec_clz3 = recall_clzone3.compute()
        ious_clz3 = iou_clzone3.compute()
        mean_iou_clz3 = ious_clz3[:3].mean()

    if configs["log_AOI_metrics"]:
        activ_i_metrics = {}
        for activ_i, activ_i_metrics_f in activ_metrics.items():
            activ_i_metrics[activ_i] = {}
            activ_i_metrics[activ_i]["accuracy"] = activ_i_metrics_f[
                0
            ].compute()  # accuracy
            activ_i_metrics[activ_i]["fscore"] = activ_i_metrics_f[
                1
            ].compute()  # fscore
            activ_i_metrics[activ_i]["precision"] = activ_i_metrics_f[
                2
            ].compute()  # precision
            activ_i_metrics[activ_i]["recall"] = activ_i_metrics_f[
                3
            ].compute()  # recall
            # iou
            activ_i_metrics[activ_i]["iou"] = activ_i_metrics_f[4].compute()

        water_act_metrics = {}
        for activ_i in water_only_metrics.keys():
            water_act_metrics[activ_i] = water_only_metrics[activ_i].compute()

    if configs["on_screen_prints"]:
        def print_metrics(prefix, metrics, labels):
            for i, label in labels.items():
                print(f"{prefix} {label}: {100 * metrics[i].item()}")

        print(f'\n{"="*20}')
        print(f"{settype} Loss: {val_loss}")
        print_metrics(f"{settype} Accuracy", acc, CLASS_LABELS)
        print_metrics(f"{settype} F-Score", score, CLASS_LABELS)
        print_metrics(f"{settype} Precision", prec, CLASS_LABELS)
        print_metrics(f"{settype} Recall", rec, CLASS_LABELS)
        print_metrics(f"{settype} IoU", ious, CLASS_LABELS)
        print(f"{settype} MeanIoU: {mean_iou * 100}")
        print(f'\n{"="*20}')

        if configs["log_zone_metrics"]:
            for zone in range(1, 4):
                print(f'\n{"="*20}\n')
                print(f"Metrics for climatic zone {zone}")
                print(
                    f"Number of samples for climatic zone {zone} = {samples_per_clzone[zone]}")
                print(f'\n{"="*20}')
                zone_metrics = eval(
                    f"acc_clz{zone}, score_clz{zone}, prec_clz{zone}, rec_clz{zone}, ious_clz{zone}, mean_iou_clz{zone}")
                print_metrics(f"{settype} Accuracy",
                              zone_metrics[0], CLASS_LABELS)
                print_metrics(f"{settype} F-Score",
                              zone_metrics[1], CLASS_LABELS)
                print_metrics(f"{settype} Precision",
                              zone_metrics[2], CLASS_LABELS)
                print_metrics(f"{settype} Recall",
                              zone_metrics[3], CLASS_LABELS)
                print_metrics(f"{settype} IoU", zone_metrics[4], CLASS_LABELS)
                print(f"{settype} MeanIoU: {zone_metrics[5] * 100}")
                print(f'\n{"="*20}')

        if configs["log_AOI_metrics"]:
            for activ_i, activ_i_metrics_list in activ_i_metrics.items():
                print(f'\n{"="*20}\n')
                print(f"Metrics for AOI {activ_i}")
                print(f'\n{"="*20}')
                print_metrics(f'{settype} AOI {activ_i} Accuracy',
                              activ_i_metrics_list["accuracy"], CLASS_LABELS)
                print_metrics(f'{settype} AOI {activ_i} F-Score',
                              activ_i_metrics_list["fscore"], CLASS_LABELS)
                print_metrics(f'{settype} AOI {activ_i} Precision',
                              activ_i_metrics_list["precision"], CLASS_LABELS)
                print_metrics(f'{settype} AOI {activ_i} Recall',
                              activ_i_metrics_list["recall"], CLASS_LABELS)
                print_metrics(f'{settype} AOI {activ_i} IoU',
                              activ_i_metrics_list["iou"], CLASS_LABELS)
                print(
                    f'{settype} AOI {activ_i} MeanIoU: {activ_i_metrics_list["iou"][:3].mean() * 100}')
                print(f'\n{"="*20}')

    return 100 * acc, 100 * score[:3].mean(), 100 * mean_iou
