import matplotlib.pyplot as plt
from utils import utils_transform
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


line_between_point = [
    # 身体关节
    [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14],
    [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21],
    # 左手手部关节
    [20, 34], [34, 35], [35, 36],    # thumb
    [20, 22], [22, 23], [23, 24],   # index
    [20, 25], [25, 26], [26, 27],   # middle
    [20, 31], [31, 32], [32, 33],   # ring
    [20, 28], [28, 29], [29, 30],   # pink
    # 右手
    [21, 49], [49, 50], [50, 51],
    [21, 37], [37, 38], [38, 39],
    [21, 40], [40, 41], [41, 42],
    [21, 46], [46, 47], [47, 48],
    [21, 43], [43, 44], [44, 45]
                      ]


def plot_skeleton(pred_motion, gt_motion, body_model):
    if len(pred_motion.shape) > 2:
        pred_motion = pred_motion[0]  # (seq, 312)
        gt_motion = gt_motion[0]
    seq_len = pred_motion.shape[0]
    pred_motion_3 = utils_transform.sixd2aa(pred_motion.reshape(seq_len, 22, 6), batch=True).reshape(-1, 66)
    gt_motion_3 = utils_transform.sixd2aa(gt_motion.reshape(seq_len, 22, 6), batch=True).reshape(-1, 66)
    pred_jtr = body_model({
        "root_orient": pred_motion_3[..., :3],
        "pose_body": pred_motion_3[..., 3:],
    }).Jtr[:, :22].detach().cpu()
    gt_jtr = body_model({
        "root_orient": gt_motion_3[..., :3],
        "pose_body": gt_motion_3[..., 3:],
    }).Jtr[:, :22].detach().cpu()

    p1_all = pred_jtr
    p2_all = gt_jtr
    for i in range(p1_all.shape[0]):
        p1 = p1_all[i]
        p2 = p2_all[i]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(p1[:, 0], p1[:, 1], p1[:, 2])
        ax.scatter(p2[:, 0], p2[:, 1], p2[:, 2])
        for line in line_between_point:
            i = line[0]
            j = line[1]
            ax.plot([p1[i, 0], p1[j, 0]], [p1[i, 1], p1[j, 1]], [p1[i, 2], p1[j, 2]], 'r-')
            ax.plot([p2[i, 0], p2[j, 0]], [p2[i, 1], p2[j, 1]], [p2[i, 2], p2[j, 2]], 'b-')
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 显示图形
        plt.show()


def plot_skeleton_by_pos(pred_jtr, gt_jtr):
    # (seq, 22, 3)
    if len(pred_jtr.shape) > 3:  # (bs, seq, 22, 3)
        pred_jtr = pred_jtr[0].cpu()
        gt_jtr = gt_jtr[0].cpu()
    p1_all = pred_jtr.cpu()
    p2_all = gt_jtr.cpu()
    for frame in range(p1_all.shape[0]):
        p1 = p1_all[frame]
        p2 = p2_all[frame]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(p1[:22, 0], p1[:22, 1], p1[:22, 2], s=100)
        ax.scatter(p2[:22, 0], p2[:22, 1], p2[:22, 2], s=100)

        ax.scatter(p1[22:, 0], p1[22:, 1], p1[22:, 2], s=50)
        ax.scatter(p2[22:, 0], p2[22:, 1], p2[22:, 2], s=50)
        for line in line_between_point:
            i = line[0]
            j = line[1]
            ax.plot([p1[i, 0], p1[j, 0]], [p1[i, 1], p1[j, 1]], [p1[i, 2], p1[j, 2]], 'r-')
            ax.plot([p2[i, 0], p2[j, 0]], [p2[i, 1], p2[j, 1]], [p2[i, 2], p2[j, 2]], 'b-')
        # generate labels for axises
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # show
        plt.show()


def plot_animation_by_pos(pred_jtr, gt_jtr, save_path):
    if len(pred_jtr.shape) > 3:  # (bs, seq, 22, 3)
        pred_jtr = pred_jtr[0].cpu()
        gt_jtr = gt_jtr[0].cpu()
    p1_all = pred_jtr.cpu()     # 预测关节位置
    p2_all = gt_jtr.cpu()       # 真实关节位置

    num_frames = p1_all.shape[0]    # 帧数

    fig = plt.figure()
    gt_ax = fig.add_subplot(121, projection='3d')
    pred_ax = fig.add_subplot(122, projection='3d')
    # # 设置视角
    # gt_ax.view_init(elev=20, azim=30)
    # pred_ax.view_init(elev=20, azim=330)
    fig.tight_layout()

    for ax in [pred_ax, gt_ax]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    def update(frame):
        p1 = p1_all[frame]
        p2 = p2_all[frame]

        # 清除当前图形以便重新绘制
        pred_ax.cla()
        gt_ax.cla()

        pred_ax.scatter(p1[:22, 0], p1[:22, 1], p1[:22, 2], c='r', s=50, label='Predicted Motion')
        gt_ax.scatter(p2[:22, 0], p2[:22, 1], p2[:22, 2], s=50, c='b', label='Ground Truth')

        pred_ax.scatter(p1[22:, 0], p1[22:, 1], p1[22:, 2], s=10)
        gt_ax.scatter(p2[22:, 0], p2[22:, 1], p2[22:, 2], s=10)

        for line in line_between_point:
            i = line[0]
            j = line[1]
            pred_ax.plot([p1[i, 0], p1[j, 0]], [p1[i, 1], p1[j, 1]], [p1[i, 2], p1[j, 2]], 'r-')
            gt_ax.plot([p2[i, 0], p2[j, 0]], [p2[i, 1], p2[j, 1]], [p2[i, 2], p2[j, 2]], 'b-')

    ani = FuncAnimation(fig, update, frames=num_frames, repeat=False)
    ani.save(save_path, writer='ffmpeg', fps=60)
    plt.show()