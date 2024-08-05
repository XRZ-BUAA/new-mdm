# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import torch.nn as nn


###############################
############ Layers ###########
###############################


class MLPblock(nn.Module):
    def __init__(self, dim, seq0, seq1, first=False, w_embed=True):
        super().__init__()

        self.w_embed = w_embed
        # self.fc0 = nn.Conv1d(seq0, seq1, 1).cuda()

        if self.w_embed:
            if first:
                self.conct = nn.Linear(dim * 3, dim).cuda()
            else:
                self.conct = nn.Identity().cuda()
            self.emb_fc = nn.Linear(dim, dim).cuda()

        self.fc1 = nn.Linear(dim, dim).cuda()
        self.norm0 = nn.LayerNorm(dim).cuda()
        self.norm1 = nn.LayerNorm(dim).cuda()
        self.act = nn.SiLU().cuda()


    def forward(self, inputs):
        inputs = [input_tensor.to('cuda:0') for input_tensor in inputs]

        if self.w_embed:
            x = inputs[0]
            embed = inputs[1]
            x = self.conct(x) + self.emb_fc(self.act(embed))
        else:
            x = inputs

        print("MLPblock Inputs Shape")
        print(x.shape)
        x_ = self.norm0(x)

        # 我要开始魔改了
        '''
        n = x_.shape[-1]
        in_process = nn.Linear(n, 14*312).cuda()
        x_ = in_process(x_)
        x_ = x_.reshape(-1, 14, 312)
        x_ = self.fc0(x_)
        x_ = x_.reshape(-1, 14 * 312)
        out_process = nn.Linear(14 * 312, n).cuda()
        x_ = out_process(x_)
        '''
        x_ = self.act(x_)

        print(x.shape)
        print(x_.shape)
        x = x + x_

        x_ = self.norm1(x)
        x_ = self.fc1(x_)
        x_ = self.act(x_)

        x = x + x_

        if self.w_embed:
            return x, embed
        else:
            return x


class BaseMLP(nn.Module):
    def __init__(self, dim, seq, num_layers, w_embed=True):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(
                MLPblock(dim, seq, seq, first=i == 0 and w_embed, w_embed=w_embed).cuda()
            )

        self.mlps = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlps(x)
        return x


###############################
########### Networks ##########
###############################


class DiffMLP(nn.Module):
    def __init__(self, latent_dim=1024, seq=7, num_layers=12):
        super(DiffMLP, self).__init__()

        self.motion_mlp = BaseMLP(dim=latent_dim, seq=seq, num_layers=num_layers)

    def forward(self, motion_input, embed):

        motion_feats = self.motion_mlp([motion_input, embed])[0]

        return motion_feats


class PureMLP(nn.Module):
    def __init__(
        self, latent_dim=1024, seq=98, num_layers=12, input_dim=54, output_dim=312
    ):
        super(PureMLP, self).__init__()

        self.input_fc = nn.Linear(input_dim, latent_dim)
        self.motion_mlp = BaseMLP(
            dim=latent_dim, seq=seq, num_layers=num_layers, w_embed=False
        )
        self.output_fc = nn.Linear(latent_dim, output_dim)

    def forward(self, motion_input):

        motion_feats = self.input_fc(motion_input)
        motion_feats = self.motion_mlp(motion_feats)
        motion_feats = self.output_fc(motion_feats)

        return motion_feats
