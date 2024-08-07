import os
import numpy as np
import torch
import math
import random

from utils.fixseed import fixseed
from utils.parser_util import sample_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
import shutil
from tqdm import tqdm
from data_loaders.dataloader import load_data, TestDataset
from model.networks import PureMLP
from human_body_prior.body_model.body_model import BodyModel as BM
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose
from utils import utils_transform
from utils.util_plot import plot_skeleton_by_pos
from utils.metrics import get_metric_function

device = torch.device("cuda")

METERS_TO_CENTIMETERS = 100.0

pred_metrics = [
    "mpjre",
    "mpjpe",
    "mpjve",
    "handpe",
    "upperpe",
    "lowerpe",
    "rootpe",
    "pred_jitter",
]
gt_metrics = [
    "gt_jitter",
]
all_metrics = pred_metrics + gt_metrics

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)  # 57.2958 grads
metrics_coeffs = {
    "mpjre": RADIANS_TO_DEGREES,
    "mpjpe": METERS_TO_CENTIMETERS,
    "mpjve": METERS_TO_CENTIMETERS,
    "handpe": METERS_TO_CENTIMETERS,
    "upperpe": METERS_TO_CENTIMETERS,
    "lowerpe": METERS_TO_CENTIMETERS,
    "rootpe": METERS_TO_CENTIMETERS,
    "pred_jitter": 1.0,
    "gt_jitter": 1.0,
    "gt_mpjpe": METERS_TO_CENTIMETERS,
    "gt_mpjve": METERS_TO_CENTIMETERS,
    "gt_handpe": METERS_TO_CENTIMETERS,
    "gt_rootpe": METERS_TO_CENTIMETERS,
    "gt_upperpe": METERS_TO_CENTIMETERS,
    "gt_lowerpe": METERS_TO_CENTIMETERS,
}


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


def predict_sparse(args, sparse_original):
    predicted_sparse = sparse_original
    sparse_per_frame = sparse_original[:, -1, :]
    frame_count = 0
    while frame_count < args.predict_length:

        sparse_per_frame[:, :18] = predicted_sparse[:, -1, :18] + predicted_sparse[:, -1, 18:36]
        ang_acc = predicted_sparse[:, -1, 18:36] - predicted_sparse[:, -2, 18:36]
        sparse_per_frame[:, 18:36] = predicted_sparse[:, -1, 18:36] + ang_acc
        sparse_per_frame[:, 36:45] = predicted_sparse[:, -1, 36:45] + predicted_sparse[:, -1, 45:54]
        linear_acc = predicted_sparse[:, -1, 45:54] - predicted_sparse[:, -2, 45:54]
        sparse_per_frame[:, 45:54] = predicted_sparse[:, -1, 45:54] + linear_acc
        predicted_sparse = torch.cat((predicted_sparse, sparse_per_frame.unsqueeze(1)), dim=1)
        frame_count += 1
    return predicted_sparse


def load_diffusion_model(args, body_model):
    print("Creating model and diffusion...")
    args.arch = args.arch[len("diffusion_") :]
    model, diffusion = create_model_and_diffusion(args, body_model)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    model.to("cuda:0")  # dist_util.dev())
    model.eval()  # disable random masking
    return model, diffusion


def non_overlapping_test(
        args,
        data,
        sample_fn,
        dataset,
        model,
        num_per_batch=256
):
    gt_data, sparse_original, body_param, head_motion, filename = (
        data[0].unsqueeze(0),
        data[1],
        data[2],
        data[3],
        data[4],
    )
    gt_data = gt_data.cuda().float()
    sparse_original = sparse_original.cuda().float()
    head_motion = head_motion.cuda().float()
    num_frames = head_motion.shape[0]

    # print("GT Data Shape")
    # print(gt_data.shape)
    # print("Sparse Original Shape")
    # print(sparse_original.shape)

    output_samples = []
    count = 0
    sparse_splits = []
    gt_splits = []
    flag_index = None

    stride = args.predict_length

    if args.pre_motion_length <= num_frames:
        while count < num_frames:
            if count + args.pre_motion_length > num_frames:
                tmp_k = num_frames - args.pre_motion_length
                sub_sparse = sparse_original[
                    :, tmp_k: tmp_k + args.pre_motion_length
                ]
                sub_gt = gt_data[
                    :, tmp_k: tmp_k + args.pre_motion_length
                ]
                flag_index = count - tmp_k
            else:
                sub_sparse = sparse_original[
                    :, count: count + args.pre_motion_length
                ]
                sub_gt = gt_data[
                    :, count: count + args.pre_motion_length
                ]
            sparse_splits.append(sub_sparse)
            gt_splits.append(sub_gt)
            count += stride

    else:
        flag_index = args.pre_motion_length - num_frames
        tmp_init = sparse_original[:, :1].repeat(1, flag_index, 1).clone()
        sub_sparse = torch.cat([tmp_init, sparse_original], dim=1)
        sparse_splits = [sub_sparse]
        tmp_init = gt_data[:, :1].repeat(1, flag_index, 1).clone()
        sub_gt = torch.cat([tmp_init, gt_data], dim=1)
        gt_splits = [sub_gt]

    n_steps = len(sparse_splits) // num_per_batch
    if len(sparse_splits) % num_per_batch > 0:
        n_steps += 1

    if args.fix_noise:
        # fix noise seed for every frame
        noise = torch.randn(1, 1, 1).cuda()
        noise = noise.repeat(1, args.input_motion_length, args.motion_nfeat)
    else:
        noise = None

    sample = None
    pre_sample = None
    first = True
    for step_index in range(n_steps):
        sparse_per_batch = torch.cat(
            sparse_splits[
                step_index * num_per_batch: (step_index + 1) * num_per_batch
            ],
            dim=0,
        )

        gt_per_batch = torch.cat(
            gt_splits[
                step_index * num_per_batch: (step_index + 1) * num_per_batch
            ],
            dim=0,
        )

        # print("Sparse Per Batch Shape")
        # print(sparse_per_batch.shape)
        # print("GT Per Batch Shape")
        # print(gt_per_batch.shape)
        assert sparse_per_batch.shape[0] == gt_per_batch.shape[0]
        new_batch_size = sparse_per_batch.shape[0]

        model_kwargs = {}
        model_kwargs['y'] = {}
        model_kwargs['y']['inpainted_motion'] = torch.randn(
            new_batch_size,
            args.input_motion_length,
            args.motion_nfeat
        ).cuda()
        model_kwargs['y']['inpainting_mask'] = torch.zeros(
            (
                new_batch_size,
                args.input_motion_length,
                args.motion_nfeat
            ),
            dtype=torch.bool
        )
        if pre_sample is not None:
            model_kwargs['y']['inpainted_motion'][:, : args.pre_motion_length, :] = pre_sample[
                                                                                    :,
                                                                                    -args.pre_motion_length:,
                                                                                    :
                                                                                    ]

        model_kwargs['y']['inpainted_motion'][:, : args.pre_motion_length, 90:96] = gt_per_batch[
                                                                                    :,
                                                                                    : args.pre_motion_length,
                                                                                    90:96
                                                                                    ]
        model_kwargs['y']['inpainted_motion'][:, : args.pre_motion_length, 120:132] = gt_per_batch[
                                                                                    :,
                                                                                    : args.pre_motion_length,
                                                                                    120:132
                                                                                    ]

        model_kwargs['y']['inpainting_mask'][:, : args.pre_motion_length, :] = True
        model_kwargs['y']['inpainting_mask'][:, : args.pre_motion_length, 90:96] = True
        model_kwargs['y']['inpainting_mask'][:, : args.pre_motion_length, 120:132] = True

        sample = sample_fn(
            model,
            (new_batch_size, args.input_motion_length, args.motion_nfeat),
            sparse=predict_sparse(args, sparse_per_batch),
            clip_denoised=False,
            # model_kwargs=model_kwargs,
            model_kwargs=None,
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=noise,
            const_noise=False,
        )

        pre_sample = sample.clone().detach()

        if flag_index is not None and step_index == n_steps - 1:
            last_batch = sample[-1]
            last_batch = last_batch[flag_index:]
            sample = sample[:-1].reshape(-1, args.motion_nfeat)
            sample = torch.cat([sample, last_batch], dim=0)
        else:
            sample = sample.reshape(-1, args.motion_nfeat)

        if first:
            sample_split = sample.clone().detach().cpu().float()
            first = False
        else:
            sample_split = sample[:, -args.predict_length, :].clone().detach().cpu().float()
            
        if not args.no_normalization:
            output_samples.append(dataset.inv_transform(sample_split))
        else:
            output_samples.append(sample_split)

    return output_samples, body_param, head_motion, filename


def evaluate_prediction(
    args,
    metrics,
    sample,
    body_model,
    sample_index,
    head_motion,
    body_param,
    fps,
    filename,
):
    motion_pred = sample.squeeze().cuda()
    # Get the  prediction from the model
    model_rot_input = (
        utils_transform.sixd2aa(motion_pred.reshape(-1, 6).detach())
        .reshape(motion_pred.shape[0], -1)
        .float()
    )

    T_head2world = head_motion.clone().cuda()
    t_head2world = T_head2world[:, :3, 3].clone()

    # Get the offset between the head and other joints using forward kinematic model
    body_pose_local = body_model(
        {
            "pose_body": model_rot_input[..., 3:66],
            "root_orient": model_rot_input[..., :3],
            "pose_hand": model_rot_input[..., 66:156]
        }
    ).Jtr

    # Get the offset in global coordiante system between head and body_world.
    t_head2root = -body_pose_local[:, 15, :]
    t_root2world = t_head2root + t_head2world.cuda()

    predicted_body = body_model(
        {
            "pose_body": model_rot_input[..., 3:66],
            "root_orient": model_rot_input[..., :3],
            "pose_hand": model_rot_input[..., 66:156],
            "trans": t_root2world,
        }
    )
    predicted_position = predicted_body.Jtr[:, :52, :]

    # Get the predicted position and rotation
    predicted_angle = model_rot_input

    for k, v in body_param.items():
        body_param[k] = v.squeeze().cuda()
        body_param[k] = body_param[k][-predicted_angle.shape[0] :, ...]

    # Get the  ground truth position from the model
    gt_body = body_model(body_param)
    gt_position = gt_body.Jtr[:, :52, :]

    # 可视化
    plot_skeleton_by_pos(predicted_position, gt_position)

    gt_angle = body_param["pose_body"]
    gt_root_angle = body_param["root_orient"]

    predicted_root_angle = predicted_angle[:, :3]
    predicted_angle = predicted_angle[:, 3:]

    upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    lower_index = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    eval_log = {}
    for metric in metrics:
        eval_log[metric] = (
            get_metric_function(metric)(
                predicted_position,
                predicted_angle,
                predicted_root_angle,
                gt_position,
                gt_angle,
                gt_root_angle,
                upper_index,
                lower_index,
                fps,
            )
            .cpu()
            .numpy()
        )

    torch.cuda.empty_cache()
    return eval_log

def main():
    args = sample_args()
    fixseed(args.seed)

    fps = 60    # ?

    body_model = BodyModel(args.support_dir)

    filename_list, all_info, mean, std = load_data(
        args.dataset,
        args.dataset_path,
        "test",
    )
    dataset = TestDataset(
        args.dataset,
        mean,
        std,
        all_info,
        filename_list,
    )

    log = {}
    for metric in all_metrics:
        log[metric] = 0

    model, diffusion = load_diffusion_model(args, body_model)
    sample_fn = diffusion.p_sample_loop

    for sample_index in tqdm(range(len(dataset))):
        output, body_param, head_motion, filename = non_overlapping_test(
            args,
            dataset[sample_index],
            sample_fn,
            dataset,
            model,
            args.num_per_batch
        )

        sample = torch.cat(output, dim=0)

        instance_log = evaluate_prediction(
            args,
            all_metrics,
            sample,
            body_model,
            sample_index,
            head_motion,
            body_param,
            fps,
            filename,
        )
        for key in instance_log:
            log[key] += instance_log[key]

    # Print the value for all the metrics
    print("Metrics for the predictions")
    for metric in pred_metrics:
        print(log[metric] / len(dataset) * metrics_coeffs[metric])
    print("Metrics for the ground truth")
    for metric in gt_metrics:
        print(metric, log[metric] / len(dataset) * metrics_coeffs[metric])


if __name__ == "__main__":
    main()
