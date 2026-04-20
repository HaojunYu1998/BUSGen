import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from typing import Dict
from PIL import Image
from glob import glob

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.utils import save_image

from BUSGen.DPMSolver import NoiseScheduleVP, DPM_Solver, model_wrapper
from BUSGen.Diffusion import GaussianDiffusionTrainer
from BUSGen.Model import UNet
from BUSGen.BoxSampler import BoxSampler

import Utils


@torch.no_grad()
def evaluation(Config: Dict):
    dist.init_process_group(backend="nccl", init_method = "env://")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    os.makedirs(Config["sampled_dir"], exist_ok=True)
    device_label = Utils.DEVICE_dict_toy
    if "BUSI" not in Config["json_file"]:
        device_label = Utils.DEVICE_dict_all

    # Build DDPM model
    pathology_dict = {"benign": 0, "malignant": 1}
    model = UNet(
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
        dev_num=len(device_label),
    ).cuda()
    
    # Load checkpoint
    ckpt = torch.load(os.path.join(Config["save_weight_dir"], Config["test_load_weight"]), map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    if rank == 0:
        print("Model load weight done.")
    model.eval()
    
    # Sampling tools
    noise_schedule = NoiseScheduleVP(
        schedule="discrete",
        betas=torch.linspace(Config["beta_1"], Config["beta_T"], Config["T"]).double().cuda(),
    )
    box_sampler = BoxSampler(Config["batch_size"])

    cls_names = ["benign", "malignant"]
    if "sample_classes" in Config:
        cls_names = Config["sample_classes"]
    dev_names = Config["sampled_devices"] if Config["dev_cond"] else ["None"]
    
    num_sampled_images = Config["num_sampled_images"]
    tqdm_range = tqdm(range(num_sampled_images)) if rank == 0 else range(num_sampled_images)
    
    for idx in tqdm_range:
        
        # Randomly sample device
        dev_name = np.random.choice(dev_names)
        dev_value = device_label[dev_name]
        dev_cond = torch.Tensor([dev_value]).long().cuda().reshape(1).repeat(Config["batch_size"])
        
        # Randomly sample pathology class
        cls_name = np.random.choice(cls_names)
        cls_value = pathology_dict[cls_name]
        cls_cond = torch.Tensor([cls_value]).long().cuda().reshape(1).repeat(Config["batch_size"])
        cls_cond = cls_cond + 1
        
        # Randomly sample bounding box
        lesion_box = box_sampler.sample_bounding_boxes(cls_value).reshape(Config["batch_size"], 1, 4).cuda()
        lesion_box = lesion_box.clamp(0, 1)
        lesion_box = torch.cat([lesion_box, torch.ones_like(lesion_box[:, :, [0]])], dim=-1)
        
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[Config["batch_size"], 3, Config["img_size"], Config["img_size"]], device=device
        )
        
        cond = (lesion_box, cls_cond, dev_cond)
        box, cls, dev = cond
        uncond_box = torch.zeros_like(box).cuda()
        uncond_cls = torch.zeros_like(cls).cuda().long()
        uncond_dev = torch.zeros_like(dev).cuda().long()
        uncond = (uncond_box, uncond_cls, uncond_dev)
        
        model_fn = model_wrapper(
            model,
            noise_schedule,
            guidance_type="classifier-free",
            condition=cond,
            unconditional_condition=uncond,
            guidance_scale=Config["w"]+1,
        )
        dpm_solver = DPM_Solver(model_fn, noise_schedule)
        sampledImgs = dpm_solver.sample(
            noisyImage,
            steps=Config["dpm_solver_T"],
            order=Config["dpm_solver_order"],
            method=Config["dpm_solver_method"]
        )
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        # Save Images
        for b in range(Config["batch_size"]):
            box = lesion_box[b, 0, :4]
            save_img = os.path.join(
                Config["sampled_dir"],
                f"{cls_name}_{dev_name}" + \
                f"_x1_{float(box[0]):.4f}_y1_{float(box[1]):.4f}_x2_{float(box[2]):.4f}_y2_{float(box[3]):.4f}.png")
            save_image(sampledImgs[[b]], save_img, nrow=1)
