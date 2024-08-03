import json
import os
import random

import numpy as np

import torch

from data_loaders.dataloader import get_dataloader, load_data, TrainDataset
from model.networks import PureMLP
from train.training_loop import TrainLoop

from utils import dist_util

from utils.model_util import create_model_and_diffusion
from utils.parser_util import train_args

from utils.fixseed import fixseed

def main():
    args = train_args()
    fixseed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError("save_dir [{}] already exists.".format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    print("creating data loader...")

    motions, fulls, sparses, mean, std = load_data(
        args.dataset,
        args.dataset_path,
        "train",
        input_motion_length=args.input_motion_length,
    )

    dataset = TrainDataset(
        args.dataset,
        mean,
        std,
        motions,
        fulls,
        sparses,
        args.input_motion_length,
        args.train_dataset_repeat_times,
        args.no_normalization,
    )

    dataloader = get_dataloader(
        dataset, "train", batch_size=args.batch_size, num_workers=args.num_workers
    )

    print("creating model and diffusion...")
    args.arch = args.arch[len("diffusion_"):]

    num_gpus = torch.cuda.device_count()
    args.num_workers = args.num_workers * num_gpus

    model, diffusion = create_model_and_diffusion(args)

    if num_gpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        dist_util.setup_dist()
        model = torch.nn.DataParallel(model).cuda()
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.module.parameters()) / 1000000.0)
        )
    else:
        dist_util.setup_dist(args.device)
        model.to(dist_util.dev())
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )

    print("Training...")
    TrainLoop(args, model, diffusion, dataloader).run_loop()
    print("Done.")


if __name__ == "__main__":
    main()