# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from radatron.modeling.backbone import build_radatron_resnet_backbone
import code
import pdb


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, cfg=None, input_shape=None, single_gpu=None, in_batch_loss=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.single_gpu = single_gpu
        if cfg is not None:
            assert input_shape is not None
            assert self.single_gpu is not None

            # encoder_q: Radatron + MLP
            # encoder_k: Identity
            self.encoder_q = Radatron_Backbone_Encoder(cfg=cfg, input_shape=input_shape, moco_dim=dim)
            self.encoder_k = Identity(cfg, moco_dim=dim)

            self.use_radical = True
        else:

            self.use_radical = False
            raise NotImplementedError("Only Radical is supported")

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.in_batch_loss = in_batch_loss

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)


    @torch.no_grad()
    def _momentum_update_key_encoder_fc(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.fc.parameters(), self.encoder_k.fc.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if not self.single_gpu:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, feat):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        if self.use_radical:
            assert feat is not None
            # assert im_id is not None

            q1 = self.encoder_q({'image1': im_q['image1'], 'image2': im_q['image2']})
            q1_norm = nn.functional.normalize(q1, dim=1) # [bs, 128]

            q2 = self.encoder_q({'image1': im_k['image1'], 'image2': im_k['image2']})
            q2_norm = nn.functional.normalize(q2, dim=1) # [bs, 128]

            q = torch.stack([q1, q2], dim=0).mean(dim=0)
            q = nn.functional.normalize(q, dim=1) # [bs, 128]
            
            with torch.no_grad():  # no gradient to keys

                k = self.encoder_k(feat)
                k = nn.functional.normalize(k, dim=1) # [bs, 128]
     


        
        else:
            raise NotImplementedError('Not implemented')
        
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK

        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        if self.in_batch_loss:
            logits = torch.einsum("nc,ck->nk", [q, k.T.clone().detach()])
            logits /= self.T
            labels = torch.arange(logits.shape[0], dtype=torch.long).cuda()

            # logits_intra = torch.einsum("nc,ck->nk", [q1, q2.T])
        
        logits_intra = torch.einsum("nc,ck->nk", [q1_norm, q2_norm.T.clone().detach()])
        logits_intra /= self.T
        labels_intra = torch.arange(logits_intra.shape[0], dtype=torch.long).cuda()

        logits_intra_undetached = torch.einsum("nc,ck->nk", [q1_norm, q2_norm.T])
        logits_intra_undetached /= self.T

        logits_intra_reverse_detached = torch.einsum("nc,ck->nk", [q1_norm.clone().detach(), q2_norm.T])
        logits_intra_reverse_detached /= self.T

        # print('logits_intra: {}'.format(logits_intra.sum(-1)))
        # print('logits_intra_undetached: {}'.format(logits_intra_undetached.sum(-1)))
        # print('logits_intra_reverse_detached: {}'.format(logits_intra_reverse_detached.sum(-1)))


        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, logits_intra, labels_intra, None, logits_intra_undetached, logits_intra_reverse_detached


class Identity(nn.Module):
    def __init__(self, cfg, moco_dim=128):
        super(Identity, self).__init__()

    def forward(self, x):
        x = x.squeeze(1)
        return x


# class Identity_FC(nn.Module):
#     def __init__(self, cfg, moco_dim=128):
#         super(Identity_FC, self).__init__()
#         out_dim = cfg.SSL.DISTILLATION_SOURCE_DIM
#         self.fc = nn.Sequential(
#                     nn.Linear(out_dim, 2048), nn.ReLU(), nn.Linear(2048, moco_dim)
#                 )

#     def forward(self, x):
#         x = self.fc(x).squeeze(1)
#         return x


class Radatron_Backbone_Encoder(nn.Module):
    def __init__(self, cfg, input_shape, moco_dim=128):
        super(Radatron_Backbone_Encoder, self).__init__()
        self.bottom_up = build_radatron_resnet_backbone(cfg=cfg, input_shape=input_shape)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.use_conv = cfg.SSL.USE_CONV

        if self.use_conv:
            raise NotImplementedError
            # self.conv = nn.Conv2d(2048, 1024, kernel_size=4, stride=2, padding=0, bias=False)
            # self.norm = nn.BatchNorm2d(1024)
            # self.relu = nn.ReLU()
            # self.fc = nn.Sequential(
            #             nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, moco_dim)
            #         )
        else:
            self.fc = nn.Sequential(
                        nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, moco_dim)
                    )



    def forward(self, x):
        x = self.bottom_up(x)
        x = x['res5'] # [bs, 2048, 14, 6]
        # x = self.pooling(x) # [bs, 2048, 1, 1]
        # x = x.squeeze(-1).squeeze(-1) # [bs, 2048]


        if self.use_conv:
            raise NotImplementedError
            # x = self.conv(x)
            # x = self.norm(x)
            # x = self.relu(x)
        x = self.pooling(x) # [bs, 2048, 1, 1]
        x = x.squeeze(-1).squeeze(-1) # [bs, 2048]
        x = self.fc(x)

        return x


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
