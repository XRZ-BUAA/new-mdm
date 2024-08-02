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
                    if k in ["pose_body", "trans", "root_orient"]
                }
            )
        return body_pose


def load_diffusion_model(args):
    print("Creating model and diffusion...")
    args.arch = args.arch[len("diffusion_") :]
    model, diffusion = create_model_and_diffusion(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    model.to("cuda:0")  # dist_util.dev())
    model.eval()  # disable random masking
    return model, diffusion


def cal_fulls(sample, body_model, gt_data):
    B, J, C = sample.shape
    motion = sample.squeeze().cuda()

    model_rot_input = (
        utils_transform.sixd2aa(motion.reshape(-1, 6).detach())
        .reshape(motion.shape[0], -1)
        .float()
    )
    gt_rot_input = (
        utils_transform.sixd2aa(gt_data.reshape(-1, 6).detach())
        .reshape(gt_data.shape[0], -1)
        .float()
    )
    model_rot_input[..., 45:48] = gt_rot_input[..., 45:48]
    model_rot_input[..., 60:72] = gt_rot_input[..., 60:72]

    rotation_local_matrot = aa2matrot(
        torch.tensor(model_rot_input).reshape(-1, 3)
    ).reshape(model_rot_input.shape[0], -1, 9)
    rotation_global_matrot = local2global_pose(
        rotation_local_matrot, body_model.kintree_table[0].long()
    )
    head_rotation_global_matrot = rotation_global_matrot[:, [15], :, :]
    T_head2world = head_rotation_global_matrot.clone().cuda()
    t_head2world = T_head2world[:, :3, 3].clone()
    body_pose_local = body_model(
        {
            "pose_body": model_rot_input[..., 3:72],
            "root_orient": model_rot_input[..., :3],
        }
    ).Jtr

    t_head2root = -body_pose_local[:, 15, :]
    t_root2world = t_head2root + t_head2world.cuda()

    predicted_body = body_model(
        {
            "pose_body": model_rot_input[..., 3:72],
            "root_orient": model_rot_input[..., :3],
            "trans": t_root2world,
        }
    )

    rotation_global_6d = utils_transform.matrot2sixd(
        rotation_global_matrot.reshape(-1, 3, 3)
    ).reshape(rotation_global_matrot.shape[0], -1, 6)
    rotation_velocity_global_matrot = torch.matmul(
        torch.inverse(rotation_global_matrot[:-1]),
        rotation_global_matrot[1:],
    )
    rotation_velocity_global_6d = utils_transform.matrot2sixd(
        rotation_velocity_global_matrot.reshape(-1, 3, 3)
    ).reshape(rotation_velocity_global_matrot.shape[0], -1, 6)

    position_global_full_gt_world = predicted_body.Jtr[
                                    :, :24, :
                                    ]
    num_frames = position_global_full_gt_world.shape[0] - 1

    output = torch.cat(
        [
            rotation_global_6d[1:, :24, :].reshape(num_frames, -1),
            rotation_velocity_global_6d[:, :24, :].reshape(num_frames, -1),
            position_global_full_gt_world[1:, :24, :].reshape(
                num_frames, -1
            ),
            position_global_full_gt_world[1:, :24, :].reshape(
                num_frames, -1
            )
            - position_global_full_gt_world[:-1, :24, :].reshape(
                num_frames, -1
            ),
        ],
        dim=-1,
    )
    return output.reshape(B, J, -1)


def non_overlapping_test(
        args,
        data,
        sample_fn,
        dataset,
        model,
        body_model,
        num_per_batch=256
):
    gt_data, sparse_original, body_param, head_motion, filename = (
        data[0],
        data[1],
        data[2],
        data[3],
        data[4],
    )
    gt_data = gt_data.cuda().float()
    sparse_original = sparse_original.cuda().float()
    head_motion = head_motion.cuda().float()
    num_frames = head_motion.shape[0]

    output_samples = []
    count = 0

    gt_splits = []
    sparse_splits = []

    flag_index = None

    # 没有前序输出时填入随机数
    pre_sample = torch.randn(num_per_batch, args.input_motion_length / 2, args.motion_nfeat)

    if args.input_motion_length / 2 <= num_frames:
        while count < num_frames:
            if count + args.input_motion_length / 2 > num_frames:
                tmp_k = num_frames - args.input_motion_length / 2
                sub_sparse = sparse_original[
                    :, tmp_k + args.input_motion_length / 2
                ]
                sub_gt = gt_data[
                    :, tmp_k : tmp_k + args.input_motion_length / 2 + 1
                ]

                flag_index = count - tmp_k
            else:
                sub_sparse = sparse_original[
                    :, count + args.input_motion_length / 2
                ]

                sub_gt = gt_data[
                    :, count: count + args.input_motion_length / 2 + 1
                ]
            sparse_splits.append(sub_sparse)
            gt_splits.append(sub_gt)
            count += args.input_motion_length / 2
    else:
        sub_sparse = sparse_original[:, num_frames]
        flag_index = args.input_motion_length / 2 - num_frames
        tmp_init = gt_data[:, :1].repeat(1, flag_index, 1).clone()
        sub_gt = torch.cat([tmp_init, gt_data], dim=1)
        sparse_splits = [sub_sparse]
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
        new_batch_size = sparse_per_batch.shape[0]

        full_per_batch = cal_fulls(pre_sample, body_model, gt_per_batch[:, :-1])

        model_kwargs = {}
        model_kwargs['y'] = {}
        model_kwargs['y']['inpainted_motion'] = torch.randn(num_per_batch, args.input_motion_length, args.motion_nfeat)

        model_kwargs['y']['inpainted_motion'][:, : args.input_motion_length / 2, :] = pre_sample
        model_kwargs['y']['inpainted_motion'][:, args.input_motion_length / 2, 90:96] = gt_data[:, -1, 45:48]
        model_kwargs['y']['inpainted_motion'][:, args.input_motion_length / 2, 120:144] = gt_data[:, -1, 60:72]

        model_kwargs['y']['inpainting_mask'] = torch.ones_like(model_kwargs['y']['inpainted_motion'], dtype=torch.bool,
                                                               device=model_kwargs['y']['inpainted_motion'].device)
        model_kwargs['y']['inpainting_mask'][:, : args.input_motion_length / 2, :] = False
        model_kwargs['y']['inpainting_mask'][:, args.input_motion_length / 2, 45:48] = False
        model_kwargs['y']['inpainting_mask'][:, args.input_motion_length / 2, 60:72] = False

        sample = sample_fn(
            model,
            (new_batch_size, args.input_motion_length, args.motion_nfeat),
            full=full_per_batch,
            sparse=sparse_per_batch,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=noise,
            const_noise=False,
        )

        pre_sample = sample[:, args.input_motion_length / 2: args.input_motion_length, :]

        if flag_index is not None and step_index == n_steps - 1:
            last_batch = sample[-1]
            last_batch = last_batch[flag_index:]
            sample = sample[:-1].reshape(-1, args.motion_nfeat)
            sample = torch.cat([sample, last_batch], dim=0)
        else:
            sample = sample.reshape(-1, args.motion_nfeat)

        if not args.no_normalization:
            output_samples.append(dataset.inv_transform(sample.cpu().float()))
        else:
            output_samples.append(sample.cpu().float())

    return output_samples, body_param, head_motion, filename

'''
def overlapping_test(
        args,
        data,
        sample_fn,
        dataset,
        model,
        body_model,
        sld_wind_size=70,
):
    gt_data, sparse_original, body_param, head_motion, filename = (
        data[0],
        data[1],
        data[2],
        data[3],
        data[4],
    )
    gt_data = gt_data.cuda().float()
    sparse_original = sparse_original.cuda().float()
    head_motion = head_motion.cuda().float()
    num_frames = head_motion.shape[0]

    output_samples = []
    count = 0
    sparse_splits = []
    flag_index = None
    
'''

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
        }
    ).Jtr

    # Get the offset in global coordiante system between head and body_world.
    t_head2root = -body_pose_local[:, 15, :]
    t_root2world = t_head2root + t_head2world.cuda()

    predicted_body = body_model(
        {
            "pose_body": model_rot_input[..., 3:66],
            "root_orient": model_rot_input[..., :3],
            "trans": t_root2world,
        }
    )
    predicted_position = predicted_body.Jtr[:, :22, :]

    # Get the predicted position and rotation
    predicted_angle = model_rot_input

    for k, v in body_param.items():
        body_param[k] = v.squeeze().cuda()
        body_param[k] = body_param[k][-predicted_angle.shape[0] :, ...]

    # Get the  ground truth position from the model
    gt_body = body_model(body_param)
    gt_position = gt_body.Jtr[:, :22, :]

    gt_angle = body_param["pose_body"]
    gt_root_angle = body_param["root_orient"]

    predicted_root_angle = predicted_angle[:, :3]
    predicted_angle = predicted_angle[:, 3:]

    upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
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

    model, diffusion = load_diffusion_model(args)
    sample_fn = diffusion.p_sample_loop

    for sample_index in tqdm(range(len(dataset))):
        output, body_param, head_motion, filename = non_overlapping_test(
            args,
            dataset[sample_index],
            sample_fn,
            dataset,
            model,
            body_model,
            args.num_per_batch
        )

        sample = torch.cat(output, axis=0)
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
