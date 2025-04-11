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
from utilities.logger import FileLogger

CLASS_LABELS = {0: "No water", 1: "Permanent Waters",
                2: "Floods", 3: "Invalid pixels"}


def train_contrastive_semantic_segmentation(
    model, train_loader, val_loader, test_loader, configs, model_configs
):
    global logger
    logger = FileLogger(configs['checkpoint_path'] + '/train.log')
    logger.log("Training started")

    # Accuracy, loss, optimizer, lr scheduler
    accuracy, fscore, precision, recall, iou = initialize_metrics(configs)
    # TODO 단순 CrossEntropyLoss 말고 ce+dice 도 테스트.
    criterion = create_loss(configs, mode="train")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_configs["learning_rate"])
    lr_scheduler = init_lr_scheduler(  # TODO SAM optimizer 추가 테스트.
        optimizer, configs, model_configs, steps=len(train_loader)
    )
    start_epoch = 0
    if configs['resume_checkpoint']:
        checkpoint = torch.load(
            configs["checkpoint_path"]+"/best_segmentation.pt", map_location=configs['device'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val = checkpoint['best_val']
        best_stats = checkpoint['best_stats']
        logger.log(f"Resumed training from epoch {start_epoch}")

    num_classes = len(CLASS_LABELS) - 1

    model.to(configs["device"])
    best_val = 0.0
    best_stats = {}

    if configs["mixed_precision"]:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, configs["epochs"]):
        model.train()

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
                    # (A) Changed Contrastive Loss: pre_event1에서 no water (0) -> post_event에서 Floods(2)
                    changed_loss_sum += changed_contrastive_loss_region(feat_pre1_up[b], feat_pre2_up[b], feat_post_up[b],
                                                                        mask[b], temperature=0.07)
                    # (B) Global Contrastive Loss: 세 시점의 feature concat
                    global_feat = torch.cat([feat_pre1_up[b].unsqueeze(0),
                                            feat_pre2_up[b].unsqueeze(0),
                                            feat_post_up[b].unsqueeze(0)], dim=1).squeeze(0)  # [192, H, W]
                    global_loss_sum += global_contrastive_loss_region(global_feat, mask[b],
                                                                      num_classes=num_classes, temperature=0.07,
                                                                      )

                changed_loss_avg = changed_loss_sum / B
                global_loss_avg = global_loss_sum / B

                total_loss = ce_loss + 0.005*changed_loss_avg + 0.1*global_loss_avg

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
                logger.print(
                    f"Epoch: {epoch}, Iteration: {index}, Train Loss: {total_loss.item()}")
                for i, label in CLASS_LABELS.items():
                    if i == 3:
                        continue
                    logger.print(
                        f"Train Accuracy ({label}): {100 * acc[i].item()}")
                    logger.print(
                        f"Train F-Score ({label}): {100 * score[i].item()}")
                    logger.print(
                        f"Train Precision ({label}): {100 * prec[i].item()}")
                    logger.print(
                        f"Train Recall ({label}): {100 * rec[i].item()}")
                    logger.print(
                        f"Train IoU ({label}): {100 * ious[i].item()}")
                logger.print(f"Train MeanIoU: {mean_iou * 100}")
                logger.print(f"lr: {lr_scheduler.get_last_lr()[0]}")

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
            logger.print(f"Epoch: {epoch}")
            logger.print(f"New best validation mIOU: {miou}")
            # logger.print(
            #     "Saving model to: ",
            #     configs["checkpoint_path"] + "/" + "best_segmentation.pt",
            # )
            best_val = miou
            best_stats["miou"] = best_val
            best_stats["epoch"] = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_val': best_val,
                'best_stats': best_stats
            }, configs["checkpoint_path"] + "/" + "best_segmentation.pt")


def changed_contrastive_loss_region(
    feat_pre1: torch.Tensor,  # [C, H, W]
    feat_pre2: torch.Tensor,  # [C, H, W]
    feat_post: torch.Tensor,  # [C, H, W]
    post_label: torch.Tensor,  # [H, W], 값=0,1,2,3 등
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Region-aware Changed Contrastive:
      - post_label==2('Floods')인 영역 픽셀을 하나의 region으로 보고,
      - 그 영역의 pre1, pre2, post의 feature들을 각각 평균내어 3개의 centroid를 만든 후,
      - post centroid(anchor)와 pre1, pre2 centroid dot product를 계산.
      - 유사도가 높으면(즉 dot가 크면) penalty를 주도록(원래 코드 논리에 맞춰) => negative처럼 동작

    Returns:
      scalar: contrastive loss (float tensor)
    """

    # 1) changed_mask: post_label이 2('Floods')인 영역
    changed_mask = (post_label == 2)
    num_pixels = changed_mask.sum()
    if num_pixels.item() == 0:
        return torch.tensor(0.0, device=feat_pre1.device)

    # 2) region-level 임베딩 계산
    #    changed_mask 위치의 feature를 평균 => centroid
    #    feat_* = [C, H, W], boolean mask = [H, W]
    #    => masked_select 후 view -> mean
    #    shape: [C, ~]
    feat_pre1_region = feat_pre1[:, changed_mask]  # shape [C, #pixels_in_mask]
    feat_pre2_region = feat_pre2[:, changed_mask]
    feat_post_region = feat_post[:, changed_mask]

    # 평균 -> centroid
    centroid_pre1 = feat_pre1_region.mean(dim=1)  # shape [C]
    centroid_pre2 = feat_pre2_region.mean(dim=1)  # shape [C]
    centroid_post = feat_post_region.mean(dim=1)  # shape [C]

    # 3) dot product
    # 원래 코드상: dot이 클수록 loss가 커진다 => negative 관계
    sim1 = torch.dot(centroid_post, centroid_pre1) / temperature
    sim2 = torch.dot(centroid_post, centroid_pre2) / temperature
    # 최종 loss = sim1 + sim2
    # (값이 클수록 penalty => loss = sim1 + sim2)
    # 필요하다면 loss_anchor = sim1 + sim2, => total_loss = loss_anchor
    loss = sim1 + sim2

    return loss/2


def global_contrastive_loss_region(
    global_feat: torch.Tensor,  # [C_global, H, W]
    gt: torch.Tensor,           # [H, W]
    num_classes: int = 4,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Region-Aware Global Contrastive:
      1) 클래스별로 픽셀을 모아 평균 임베딩(centroid) 계산
      2) 같은 클래스끼리는 '양성'(pos), 다른 클래스는 '음성'(neg)으로 보고 InfoNCE 형태로 계산
      3) invalid pixel(3)은 제외

    Returns:
      scalar: contrastive loss (float tensor)
    """

    device = global_feat.device
    C, H, W = global_feat.shape

    # 1) 클래스별 centroid 계산
    class_centroids = {}
    for cls_id in range(num_classes):
        if cls_id == 3:
            continue  # invalid pixel은 제외
        mask = (gt == cls_id)
        count = mask.sum()
        if count.item() == 0:
            continue
        # region_aware feat mean => centroid
        feat_region = global_feat[:, mask]  # shape [C, #pixels_in_that_class]
        centroid = feat_region.mean(dim=1)  # shape [C]
        class_centroids[cls_id] = centroid

    # 클래스가 1개 이하라면 contrastive 계산 불가
    if len(class_centroids) < 2:
        return torch.tensor(0.0, device=device)

    # 2) InfoNCE-ish 계산
    #    for each cls => anchor=centroid[clsA], pos=centroid[clsA], neg=다른 cls centroid
    loss = 0.0
    count_pair = 0
    items = list(class_centroids.items())  # [(cls_id, centroid), ...]

    for i, (clsA, anchor) in enumerate(items):
        pos_dot_sum = 0.0
        neg_dot_sum = 0.0
        pos_count = 1  # 자기 자신이 pos
        neg_count = 0
        for j, (clsB, other) in enumerate(items):
            sim = anchor.dot(other) / temperature
            if clsA == clsB:
                # 같은 클래스는 양성
                pos_dot_sum += torch.exp(sim)
            else:
                # 다른 클래스는 음성
                neg_dot_sum += torch.exp(sim)
                neg_count += 1

        # InfoNCE: log( pos / (pos+neg) )
        if neg_count > 0:
            numerator = pos_dot_sum + 1e-10
            denominator = pos_dot_sum + neg_dot_sum + 1e-10
            loss_cls = -torch.log(numerator / denominator)
            loss += loss_cls
            count_pair += 1

    if count_pair == 0:
        return torch.tensor(0.0, device=device)

    return loss / count_pair


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
                pre_event_2 = pre_event_2.to(configs["device"])
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
                logger.print(f"{prefix} {label}: {100 * metrics[i].item()}")

        logger.print(f'\n{"="*20}')
        logger.print(f"{settype} Loss: {val_loss}")
        print_metrics(f"{settype} Accuracy", acc, CLASS_LABELS)
        print_metrics(f"{settype} F-Score", score, CLASS_LABELS)
        print_metrics(f"{settype} Precision", prec, CLASS_LABELS)
        print_metrics(f"{settype} Recall", rec, CLASS_LABELS)
        print_metrics(f"{settype} IoU", ious, CLASS_LABELS)
        logger.print(f"{settype} MeanIoU: {mean_iou * 100}")
        logger.print(f'\n{"="*20}')

        if configs["log_zone_metrics"]:
            for zone in range(1, 4):
                logger.print(f'\n{"="*20}\n')
                logger.print(f"Metrics for climatic zone {zone}")
                logger.print(
                    f"Number of samples for climatic zone {zone} = {samples_per_clzone[zone]}")
                logger.print(f'\n{"="*20}')
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
                logger.print(f"{settype} MeanIoU: {zone_metrics[5] * 100}")
                logger.print(f'\n{"="*20}')

        if configs["log_AOI_metrics"]:
            for activ_i, activ_i_metrics_list in activ_i_metrics.items():
                logger.print(f'\n{"="*20}\n')
                logger.print(f"Metrics for AOI {activ_i}")
                logger.print(f'\n{"="*20}')
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
                logger.print(
                    f'{settype} AOI {activ_i} MeanIoU: {activ_i_metrics_list["iou"][:3].mean() * 100}')
                logger.print(f'\n{"="*20}')

    return 100 * acc, 100 * score[:3].mean(), 100 * mean_iou
