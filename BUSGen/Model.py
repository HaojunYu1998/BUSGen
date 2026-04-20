import os
import math
import Utils
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(p=keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class ConditionalEmbedding(nn.Module):
    def __init__(self, d_model, dim, box_cond, cls_cond, dev_cond, dev_num):
        assert d_model % 2 == 0
        super().__init__()
        self.d_model = d_model
        self.dim = dim
        self.box_cond = False # box_cond
        self.cls_cond = cls_cond
        self.dev_cond = dev_cond
        in_dim = d_model * (box_cond + cls_cond + dev_cond)
        
        if self.cls_cond:
            self.cls_condEmbedding = nn.Embedding(num_embeddings=3, embedding_dim=d_model, padding_idx=0)
        if self.dev_cond:
            self.dev_condEmbedding = nn.Embedding(num_embeddings=dev_num, embedding_dim=d_model, padding_idx=0)
        self.condEmbedding = nn.Sequential(
            nn.Linear(in_dim, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def box_embadding(self, box):
        """
        Params:
            box: Tensor (N, 4)
        """
        dim_t = torch.arange(self.d_model // 8, dtype=torch.float32, device=box.device)
        dim_t = 2 ** dim_t * math.pi

        pos = box[:, :, None] * dim_t[None, None, :]

        pos_x1 = torch.stack((pos[:, [0]].sin(), pos[:, [0]].cos()), dim=1).flatten(1)
        pos_y1 = torch.stack((pos[:, [1]].sin(), pos[:, [1]].cos()), dim=1).flatten(1)
        pos_x2 = torch.stack((pos[:, [2]].sin(), pos[:, [2]].cos()), dim=1).flatten(1)
        pos_y2 = torch.stack((pos[:, [3]].sin(), pos[:, [3]].cos()), dim=1).flatten(1)
        pos = torch.cat((pos_x1, pos_y1, pos_x2, pos_y2), dim=-1)
        return pos

    def forward(self, labels):
        box, cls, dev = labels
        emb = []
        # if self.box_cond:
        #     box_emb = self.box_embadding(box)
        #     emb.append(box_emb)
        if self.cls_cond:
            cls_emb = self.cls_condEmbedding(cls)
            emb.append(cls_emb)
        if self.dev_cond:
            dev_emb = self.dev_condEmbedding(dev)
            emb.append(dev_emb)
        emb = torch.cat(emb, dim=-1)
        emb = self.condEmbedding(emb)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x, temb, cemb):
        x = self.c1(x) + self.c2(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.t = nn.ConvTranspose2d(in_ch, in_ch, 5, 2, 2, 1)

    def forward(self, x, temb, cemb):
        _, _, H, W = x.shape
        x = self.t(x)
        x = self.c(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch, num_groups, affine):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, in_ch, affine=affine)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, num_groups, affine, attn=True):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_ch, affine=affine),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_ch, affine=affine),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch, num_groups, affine)
        else:
            self.attn = nn.Identity()

    def forward(self, x, temb, cemb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h += self.cond_proj(cemb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, num_res_blocks, dropout, num_groups, affine,
                 box_cond, cls_cond, dev_cond, dev_num):
        super().__init__()
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.cond_embedding = ConditionalEmbedding(
            ch, tdim, False, cls_cond, dev_cond, dev_num
        )
        self.box_cond = box_cond
        if self.box_cond:
            self.head = nn.Conv2d(4, ch, kernel_size=3, stride=1, padding=1)
        else:
            self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim, 
                    dropout=dropout, num_groups=num_groups, affine=affine))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, num_groups, affine, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, num_groups, affine, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, 
                    dropout=dropout, num_groups=num_groups, affine=affine, attn=False))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(num_groups, now_ch, affine=affine),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        
    def cat_box(self, x, box):
        """
        Params:
            x: Tensor (B, 3, H, W)
            box: Tensor (B, N, 4)
        Returns:
            x: Tensor (B, 4, H, W)
        """
        c_box = torch.zeros_like(x)[ :, [0], :, :]
        H, W = x.shape[2:]
        box = box.cpu()
        for b in range(box.shape[0]):
            box_b = box[b]
            box_b = box_b[box_b[:, -1] == 1][:, :4]
            if len(box_b):
                box_b = box_b * torch.Tensor([W, H, W, H])[None]
                for i in range(box_b.shape[0]):
                    x1, y1, x2, y2 = box_b[i].tolist()
                    c_box[b, 0, int(y1):int(y2), int(x1):int(x2)] = 1
        cx = torch.cat([x, c_box], dim=1)
        return cx

    def forward(self, x, t, labels):
        # Timestep embedding
        temb = self.time_embedding(t)
        cemb = self.cond_embedding(labels)
        # Downsampling
        if self.box_cond:
            box = labels[0]
            x = self.cat_box(x, box)
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, cemb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb, cemb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, cemb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, 
        num_labels=10, 
        ch=128, 
        ch_mult=[1, 2, 2, 2],
        num_res_blocks=2, 
        dropout=0.1
    )
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, size=[batch_size])
    labels = torch.randint(10, size=[batch_size])
    y = model(x, t, labels)
    print(y.shape)

