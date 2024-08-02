from argparse import ArgumentParser
import argparse
import os
import json


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)
    '''
    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))
    '''
    # 结合 AGRoL 和 MDM
    for a in args_to_overwrite:
        if a in model_args.keys():
            if a == "dataset":
                if args.__dict__[a] is None:
                    args.__dict__[a] = model_args[a]
            elif a == "input_motion_length":
                continue
            else:
                setattr(args, a, model_args[a])
        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))
        '''
        elif 'cond_mode' in model_args:  # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)
        '''

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    # Copy from AGRoL
    group.add_argument("--timestep_respacing", default="", type=str, help="ddim timestep respacing.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')

    # 用的AGRoL版
    '''
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    '''
    group.add_argument(
        "--arch",
        default="DiffMLP",
        type=str,
        help="Architecture types as reported in the paper.",
    )

    # 这个参数 AGRoL 没有
    '''
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    '''

    # Copy from AGRoL
    group.add_argument(
        "--motion_nfeat", default=132, type=int, help="motion feature dimension"
    )
    # 稀疏信号新增手部两个关节点
    group.add_argument(
        "--sparse_dim", default=90, type=int, help="sparse signal feature dimension"
    )

    # 包含冗余信息的全身动作特征（24个关节点）
    group.add_argument(
        "--motion_dim", default=432, type=int, help="The full-body motion features containing redundant information"
    )

    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")

    # 把 default 改成了 0.0
    group.add_argument("--cond_mask_prob", default=0.0, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")

    # Copy from AGRoL
    group.add_argument(
        "--input_motion_length",
        default=14, # AGRoL默认为196，咱不能这样，暂定为14
        type=int,
        help="Limit for the maximal number of frames.",
    )
    group.add_argument(
        "--no_normalization",
        action="store_true",
        help="no data normalisation for the 6d motions",
    )

    # 后面这四个参数都是 AGRoL 丢掉了的，前三个是 mdm 用来计算损失的，最后一个和有没有条件控制有关，先保留
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    '''
    由于我们也有控制信号，所以不需要这个
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")
    '''


# 这个函数改成了 AGRoL 的版本
def add_data_options(parser):
    group = parser.add_argument_group("dataset")
    group.add_argument(
        "--dataset",
        default=None,
        choices=[
            "amass",
        ],
        type=str,
        help="Dataset name (choose from list).",
    )
    group.add_argument(
        "--dataset_path",
        default="./dataset/AMASS/",
        type=str,
        help="Dataset path",
    )


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")

    # 学习率的默认值两者设置不一样，AGRoL 设置的默认学习率为 2e-4，mdm 则是 1e-4，学习率设置大一点是不是可以快一点？
    group.add_argument("--lr", default=2e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")

    # Copy from AGRoL
    # mdm 貌似没有设置这个参数
    group.add_argument(
        "--train_dataset_repeat_times",
        default=1000,
        type=int,
        help="Repeat the training dataset to save training time",
    )

    # AGRoL 没有设置这两个参数，我觉得训练过程先抄 AGRoL?
    '''
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    '''

    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")

    '''
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    '''

    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=50_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")

    '''
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    '''

    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")

    # 这两个参数是 AGRoL 设置的
    group.add_argument(
        "--load_optimizer",
        action="store_true",
        help="If True, will also load the saved optimizer state for network initialization",
    )
    group.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of dataloader workers.",
    )

# 换成了 AGRoL 版本
def add_sampling_options(parser):
    group = parser.add_argument_group("sampling")
    group.add_argument(
        "--overlapping_test",
        action="store_true",
        help="enabling overlapping test",
    )
    group.add_argument(
        "--num_per_batch",
        default=256,
        type=int,
        help="the batch size of each split during non-overlapping testing",
    )
    group.add_argument(
        "--sld_wind_size",
        default=70,
        type=int,
        help="the sliding window size",
    )

    '''
    由于 AGRoL 可视化有问题，我们又要接入 Unity，就把这个先注释掉了
    group.add_argument(
        "--vis",
        action="store_true",
        help="visualize the output",
    )
    '''

    group.add_argument(
        "--fix_noise",
        action="store_true",
        help="fix init noise for the output",
    )
    group.add_argument(
        "--fps",
        default=30,
        type=int,
        help="FPS",
    )
    group.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to model####.pt file to be sampled.",
    )
    group.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Path to results dir (auto created by the script). "
        "If empty, will create dir in parallel to checkpoint.",
    )
    group.add_argument(
        "--support_dir",
        type=str,
        help="the dir that you store your smplh and dmpls dirs",
    )


'''
这些对我们没用
def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")


def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")
'''

# 换成了 AGRoL 版本
def add_evaluation_options(parser):
    group = parser.add_argument_group("eval")
    group.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to model####.pt file to be sampled.",
    )

'''
# 改了一下
def get_cond_mode(args):
    if args.unconstrained:
        cond_mode = 'no_cond'
    else:
        cond_mode = 'cond'
    return cond_mode
'''


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()

'''
def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    elif (args.action_file or args.action_name) and cond_mode != 'action':
        raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args


def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)
'''


# Copy from AGRoL
def sample_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)