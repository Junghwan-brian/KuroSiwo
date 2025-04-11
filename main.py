import argparse
import pyjson5 as json
import pprint
import random
from pathlib import Path

import numpy as np
import torch
from models.model_utilities import *
from torchmetrics import Accuracy, F1Score, Precision, Recall

import training.train_mae
from training.change_detection_trainer import (
    eval_change_detection,
    train_change_detection,
)
from training.segmentation_trainer import (
    eval_semantic_segmentation,
    train_semantic_segmentation,
)
from training.recurrent_trainer import (
    eval_recurrent_segmentation,
    train_recurrent_segmentation
)
from training.contrastive_trainer import (
    train_contrastive_semantic_segmentation,
    eval_contrastive_semantic_segmentation,
)
from utilities.utilities import *
from warnings import filterwarnings
filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--method", default=None)
parser.add_argument("--task", default=None)
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--backbone", default=None)
parser.add_argument("--loss_function", default=None)
parser.add_argument("--dem", type=int, default=None)
parser.add_argument("--slope", type=int, default=None)
parser.add_argument("--batch_size", default=None)
parser.add_argument("--inputs", nargs="+", default=None)
parser.add_argument("--seed", type=int, default=999)


args = parser.parse_args()


# Seed stuff
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

if __name__ == "__main__":
    configs = json.load(open("configs/config.json", "r"))
    if args.gpu is not None:
        configs["gpu"] = int(args.gpu)
    if args.task is not None:
        configs["task"] = args.task
    if args.method is not None:
        configs["method"] = args.method
    if configs["method"] == "convlstm":
        model_configs = json.load(
            open("configs/method/temporal/convlstm.json", "r"))
    elif configs["method"] == "vivit":
        model_configs = json.load(
            open("configs/method/temporal/vivit.json", "r"))
    else:
        model_configs = json.load(
            open(
                f'configs/method/{configs["method"].lower()}/{configs["method"].lower().replace("-", "_")}.json'
            )
        )
        if args.backbone is not None:
            model_configs["backbone"] = args.backbone

    configs.update(model_configs)

    if args.inputs is None and args.dem is None:
        configs = update_config(configs, None)
    else:
        configs = update_config(configs, args)

    if args.loss_function is not None:
        configs["loss_function"] = args.loss_function

    checkpoint_path = create_checkpoint_directory(configs, model_configs)

    if args.batch_size is not None:
        configs["batch_size"] = int(args.batch_size)

    configs["checkpoint_path"] = checkpoint_path
    pprint.pprint(configs)

    # Create Loaders
    train_loader, val_loader, test_loader = prepare_loaders(configs)

    # Begin Training
    if configs["task"] == "segmentation":
        if configs['method'] == 'convlstm':
            if not configs['test']:
                model = initialize_recurrent_model(configs, model_configs)

                train_recurrent_segmentation(
                    model,
                    train_loader,
                    val_loader,
                    test_loader,
                    configs=configs,
                    model_configs=model_configs,
                )

            # Evaluate on Test Set
            model = initialize_recurrent_model(configs, model_configs)
            rep_i = '0'  # NOTE 임시로 정해놓음.
            ckpt_path = Path(configs["checkpoint_path"]
                             ) / "best_segmentation.pt"

            print(f"Loading model from: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=configs['device'])
            model.load_state_dict(checkpoint["model_state_dict"])

            test_acc, test_score, miou = eval_recurrent_segmentation(
                model,
                test_loader,
                ckpt_path.parent,
                settype="Test",
                configs=configs,
                model_configs=model_configs,
            )

            # Print final results
            print("Test Mean IOU: ", miou)
        else:
            # Create model
            model = initialize_segmentation_model(configs, model_configs)
            if not configs["test"]:
                train_semantic_segmentation(
                    model,
                    train_loader,
                    val_loader,
                    test_loader,
                    configs=configs,
                    model_configs=model_configs,
                )

            # Evaluate on Test Set
            print(
                "Loading model from: ",
                configs["checkpoint_path"] + "/" + "best_segmentation.pt",
            )
            checkpoint = torch.load(
                configs["checkpoint_path"] + "/" + "best_segmentation.pt")
            model.load_state_dict(checkpoint['model_state_dict'])

            test_acc, test_score, miou = eval_semantic_segmentation(
                model,
                test_loader,
                settype="Test",
                configs=configs,
                model_configs=model_configs,
            )
            print("Test Mean IOU: ", miou)
    elif configs["task"] == "mae":
        print("Initializing Self-Supervised learning training with configs:")
        pprint.pprint(configs)
        training.train_mae.train(configs)
    elif configs["task"] == "cd":
        model = initialize_cd_model(configs, model_configs, "train")

        train_change_detection(
            model,
            train_loader,
            val_loader,
            test_loader,
            configs=configs,
            model_configs=model_configs,
        )

        # Evaluate on Test Set
        print(
            "Loading model from: ",
            configs["checkpoint_path"] + "/" + "best_segmentation.pt",
        )

        checkpoint = torch.load(
            configs["checkpoint_path"] + "/" + "best_segmentation.pt", map_location=configs['device']
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        test_acc, test_score, miou = eval_change_detection(
            model,
            test_loader,
            settype="Test",
            configs=configs,
            model_configs=model_configs,
        )

        print("Test Mean IOU: ", miou.item())
    elif configs["task"] == "contrastive":
        model = initialize_segmentation_model(configs, model_configs)
        if not configs["test"]:
            train_contrastive_semantic_segmentation(
                model,
                train_loader,
                val_loader,
                test_loader,
                configs=configs,
                model_configs=model_configs,
            )

        # Evaluate on Test Set
        print(
            "Loading model from: ",
            configs["checkpoint_path"] + "/" + "best_segmentation.pt",
        )
        checkpoint = torch.load(
            configs["checkpoint_path"] + "/" + "best_segmentation.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        test_acc, test_score, miou = eval_contrastive_semantic_segmentation(
            model,
            test_loader,
            settype="Test",
            configs=configs,
            model_configs=model_configs,
        )
        print("Test Mean IOU: ", miou)
