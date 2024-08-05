import json
import os

import torch

from data_loaders.dataloader import get_dataloader, load_data, TrainDataset
from train.training_loop import TrainLoop
from human_body_prior.body_model.body_model import BodyModel as BM

from utils import dist_util

from utils.model_util import create_model_and_diffusion
from utils.parser_util import train_args

from utils.fixseed import fixseed


class BodyModel(torch.nn.Module):
    def __init__(self, support_dir):
        super().__init__()

        device = torch.device("cuda")
        subject_gender = "male"
        bm_fname = os.path.join(
            support_dir, "smplh/{}/model.npz".format(subject_gender)
        )
        dmpl_fname = os.path.join(
            support_dir, "dmpls/{}/model.npz".format(subject_gender)
        )
        num_betas = 16  # number of body parameters
        num_dmpls = 8  # number of DMPL parameters
        body_model = BM(
            bm_fname=bm_fname,
            num_betas=num_betas,
            num_dmpls=num_dmpls,
            dmpl_fname=dmpl_fname,
        ).to(device)
        self.body_model = body_model.eval()

    def forward(self, body_params):
        with torch.no_grad():
            body_pose = self.body_model(
                **{
                    k: v
                    for k, v in body_params.items()
                    if k in ["pose_body", "pose_hand", "trans", "root_orient"]
                }
            )
        return body_pose


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

    motions, fulls, sparses, heads, mean, std = load_data(
        args.dataset,
        args.dataset_path,
        "train",
        input_motion_length=args.input_motion_length,
    )

    body_model = BodyModel(args.support_dir)

    dataset = TrainDataset(
        args.dataset,
        mean,
        std,
        motions,
        fulls,
        sparses,
        heads,
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
    TrainLoop(args, model, diffusion, dataloader, body_model).run_loop()
    print("Done.")


if __name__ == "__main__":
    main()