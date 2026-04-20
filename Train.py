import os
import time
import json
import numpy as np
from typing import Dict
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

from BUSGen.Diffusion import GaussianDiffusionTrainer
from BUSGen.Model import UNet
from BUSGen.Dataset import DDPMDataset

import Utils


def train(Config: Dict):
    dist.init_process_group(backend="nccl", init_method = "env://")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    save_weight_dir = Config["save_weight_dir"]
    if rank == 0:
        os.makedirs(save_weight_dir, exist_ok=True)
        json.dump(Config, open(os.path.join(save_weight_dir, "config.json"), "w"), indent=4)

    # dataset
    if rank == 0:
        print("Preparing Dataset...")
    dataset = DDPMDataset(
        json_file=Config["json_file"],
        root=Config["data_root"],
        random_flip=Config["random_flip"],
        random_crop=Config["random_crop"],
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(Config["img_size"], Config["img_size"])),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    Utils.print_(f"Training Dataset Containing {len(dataset)} Images!")

    # model
    net_model = UNet(
        T=Config["T"],
        ch=Config["channel"],
        ch_mult=Config["channel_mult"],
        num_res_blocks=Config["num_res_blocks"],
        dropout=Config["dropout"],
        num_groups=Config["num_groups"],
        affine=Config["affine"],
        box_cond=Config["box_cond"],
        cls_cond=Config["cls_cond"],
        dev_cond=Config["dev_cond"],
        # NOTE: we only provide training code of a toy model with 3 device types
        dev_num=len(Utils.DEVICE_dict_toy),
    ).cuda()
    net_model = DDP(net_model, device_ids=[rank], output_device=rank)

    # optimizer
    optimizer = torch.optim.AdamW(
        net_model.parameters(), 
        lr=Config["lr"], 
        weight_decay=1e-4
    )
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, 
        T_max=Config["epoch"], 
        eta_min=0, 
        last_epoch=-1
    )
    warmUpScheduler = Utils.GradualWarmupScheduler(
        optimizer=optimizer, 
        multiplier=Config["multiplier"], 
        warm_epoch=Config["epoch"] // 10, 
        after_scheduler=cosineScheduler
    )
    trainer = GaussianDiffusionTrainer(
        model=net_model, 
        beta_1=Config["beta_1"], 
        beta_T=Config["beta_T"], 
        T=Config["T"]
    ).cuda()
    
    e_start = 0

    # start training
    os.makedirs(save_weight_dir, exist_ok=True)
    if rank == 0:
        with open(os.path.join(save_weight_dir, "train_log.txt"), "a") as log_file:
            now = time.strftime("%c")
            log_file.write("================ Training Loss (%s) ================\n" % now)
    for e in range(e_start, Config["epoch"]):
        dataset.resample(e, Config.get("sample_interval", 1), Config["cls_resample"])
        train_sampler = DistributedSampler(dataset=dataset, shuffle=True, drop_last=True)
        dataloader = DataLoader(
            dataset,
            batch_size=Config["batch_size"],
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        tqdmDataLoader = tqdm(dataloader, dynamic_ncols=True) if rank == 0 else dataloader
        for iter_idx, (images, labels) in enumerate(tqdmDataLoader):
            b = images.shape[0]
            optimizer.zero_grad()
            x_0 = images.cuda()
            box_cond, cls_cond, dev_cond = None, None, None
            
            if Config["box_cond"]:
                box_cond = labels["lesion_box"]
                box_cond = box_cond.cuda()
            if Config["cls_cond"]:
                cls_cond = labels["pathology"]
                cls_cond = cls_cond.cuda() + 1
            if Config["dev_cond"]:
                dev_cond = labels["device"]
                dev_cond = dev_cond.cuda()

            # conditional drop rate
            cond_drop_rate = Config.get("cond_drop_rate", 0.1)
            if np.random.rand() < cond_drop_rate:
                box_cond = None if box_cond is None else torch.zeros_like(box_cond).cuda()
                cls_cond = None if cls_cond is None else torch.zeros_like(cls_cond).cuda().long()
                dev_cond = None if dev_cond is None else torch.zeros_like(dev_cond).cuda().long()
            
            # calculate loss
            loss = trainer(
                x_0, (box_cond, cls_cond, dev_cond)
            ).sum() / b ** 2.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), 
                Config["grad_clip"]
            )
            optimizer.step()
            
            # print loss
            if rank == 0:
                Utils.print_content = {
                    "epoch": e,
                    "loss: ": loss.item(),
                    "LR": optimizer.state_dict()["param_groups"][0]["lr"]
                }
                tqdmDataLoader.set_postfix(ordered_dict=Utils.print_content)
                if (iter_idx % 100) == 0:
                    message = "[epoch %d, iters: %d] " % (e, iter_idx)
                    for k, v in Utils.print_content.items():
                        message += "%s: %.3f " % (k, v) if "loss" in k else ""
                    with open(os.path.join(save_weight_dir, "train_log.txt"), "a") as log_file:
                        log_file.write("%s\n" % message)
        warmUpScheduler.step()
        if (rank == 0) and (e % 5 == 0):
            torch.save({
                "model": net_model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": e,
                "scheduler": warmUpScheduler.state_dict()}, 
                os.path.join(save_weight_dir, "ckpt_" + str(e) + "_.pt")
            )
    if rank == 0:
        torch.save({
            "model": net_model.module.state_dict()}, 
            os.path.join(save_weight_dir, "ckpt_" + str(e) + "_.pt")
        )
    dist.destroy_process_group()
