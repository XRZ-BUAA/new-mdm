# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import numpy as np
import torch
import torch.nn as nn
from model.networks import DiffMLP


class MetaModel(nn.Module):
    def __init__(
        self,
        arch,
        nfeats,
        latent_dim=1024,    # 由于现在全身动作表示的维数太高，改了一下潜在空间维度，但不知道这个数值是否合适
        num_layers=8,
        dropout=0.1,
        dataset="amass",
        motion_dim=936, # 全身22个关节点 + 双手30个关节点
        sparse_dim=54,  # 3个关节点的稀疏信号
        **kargs,
    ):
        super().__init__()

        self.arch = DiffMLP
        self.dataset = dataset

        self.input_feats = nfeats
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.sparse_dim = sparse_dim
        #
        self.motion_dim = motion_dim

        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)
        self.input_process = nn.Linear(self.input_feats, self.latent_dim)

        self.mlp = self.arch(
            self.latent_dim, seq=kargs.get("input_motion_length"), num_layers=num_layers
        )
        self.embed_timestep = TimestepEmbeding(self.latent_dim)
        self.sparse_process = nn.Linear(self.sparse_dim, self.latent_dim)
        #
        self.motion_process = nn.Linear(self.motion_dim, self.latent_dim)
        self.output_process = nn.Linear(self.latent_dim, self.input_feats * 14)

    def mask_cond(self, cond, force_mask=True):
        bs, n = cond.shape

        # DEBUG
        print(cond)

        if force_mask:
            print('Hei')
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:

            # DEBUG
            print(self.cond_mask_prob)

            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(
                bs, 1
            )  # 1-> use null_cond, 0-> use real cond

            return cond * (1.0 - mask)
        else:
            print('I am here')
            return cond

    def forward(self, x, timesteps, motion_emb, sparse_emb, force_mask=False):
        """
        x: [batch_size, nfeats, nframes], denoted x_t in the paper
        motion: [batch_size, nframes/2, motion_dim], the full features
        sparse: [batch_size, 1, sparse_dim], the sparse features
        timesteps: [batch_size] (int)
        """
        emb = self.embed_timestep(timesteps)  # time step embedding : [1, bs, d]

        # 把输入和条件的后两维展平
        bs, nframes = x.shape[:2]
        x = x.reshape(bs, -1)
        motion_emb = motion_emb.reshape(bs, -1)
        sparse_emb = sparse_emb.reshape(bs, -1)

        # DEBUG
        print(motion_emb)

        motion_emb = self.motion_process(
            self.mask_cond(motion_emb, force_mask=force_mask)
        )
        # Pass the sparse signal to a FC
        sparse_emb = self.sparse_process(
            self.mask_cond(sparse_emb, force_mask=force_mask)
        )

        # Pass the input to a FC
        x = self.input_process(x)

        # Concat the sparse feature with input
        # x = torch.cat((sparse_emb, x), axis=-1)
        x = torch.cat((motion_emb, sparse_emb, x), dim=-1)
        output = self.mlp(x, emb)

        # Pass the output to a FC and reshape the output
        output = self.output_process(output)
        output = output.reshape(bs, nframes, -1)
        return output


class TimestepEmbeding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, timesteps):
        return self.pe[timesteps]
