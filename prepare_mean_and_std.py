import argparse
import math
import os

import numpy as np
import torch

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose
from tqdm import tqdm
from utils import utils_transform

'''
AGRoL 的均值和标准差pt文件用不了
'''

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    mean = None
    std = None
    rot_sum = None
    num = 0
    print("Calculating mean")
    for dataroot_subset in ["BioMotionLab_NTroje", "CMU", "MPI_HDM05"]:
        print(dataroot_subset)
        subset_dir = os.path.join(args.root_dir, dataroot_subset)
        for root, dirs, files in os.walk(subset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                bdata = np.load(
                    file_path, allow_pickle=True
                )

                try:
                    bdata_poses = bdata["poses"]
                except KeyError:
                    print('Warning: Can not find poses')
                    continue

                output_aa = torch.Tensor(bdata_poses[:, :156]).reshape(-1, 3)
                output_6d = utils_transform.aa2sixd(output_aa).reshape(
                    bdata_poses.shape[0], -1
                )
                rotation_local_full_gt_list = output_6d[1:]
                if rot_sum == None:
                    rot_sum = torch.sum(rotation_local_full_gt_list, dim=0)
                    num = rotation_local_full_gt_list.shape[0]
                else:
                    rot_sum += torch.sum(rotation_local_full_gt_list, dim=0)
                    num += rotation_local_full_gt_list.shape[0]

    mean = rot_sum / num
    print("Mean Shape:")
    print(mean.shape)
    torch.save(mean, os.path.join(args.save_dir, "amass_mean.pt"))

    mse = None
    print("Calculating std")
    for dataroot_subset in ["BioMotionLab_NTroje", "CMU", "MPI_HDM05"]:
        print(dataroot_subset)
        subset_dir = os.path.join(args.root_dir, dataroot_subset)
        for root, dirs, files in os.walk(subset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                bdata = np.load(
                    file_path, allow_pickle=True
                )
                bdata_poses = bdata["poses"]
                output_aa = torch.Tensor(bdata_poses[:, :156]).reshape(-1, 3)
                output_6d = utils_transform.aa2sixd(output_aa).reshape(
                    bdata_poses.shape[0], -1
                )
                rotation_local_full_gt_list = output_6d[1:]
                for row in rotation_local_full_gt_list:
                    if mse == None:
                        mse = (row - mean) ** 2
                    else:
                        mse += (row - mean) ** 2

    std = math.sqrt(mse) / num
    print("STD:")
    print(std)
    torch.save(std, os.path.join(args.save_dir, "amass_std.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="=dir where you want to save your generated data",
    )
    parser.add_argument(
        "--root_dir", type=str, default=None, help="=dir where you put your AMASS data"
    )
    args = parser.parse_args()

    main(args)
