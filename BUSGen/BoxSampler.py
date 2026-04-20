import os
import joblib
import numpy as np
from typing import Dict
from itertools import chain
from sklearn.neighbors import KernelDensity

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from Utils import gather

pathology_dict = {0: "benign", 1: "malignant"}


class BoxSampler:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.kde_center = {}
        self.kde_aspect_ratio = {}
        self.kde_area = {}
        for pathology in ["benign", "malignant"]:
            # Extract the individual components of the prior boxes
            kde_center = joblib.load(f"ModelFiles/{pathology}_center_kde.pkl")
            kde_aspect_ratio = joblib.load(f"ModelFiles/{pathology}_aspect_ratio_kde.pkl")
            kde_area = joblib.load(f"ModelFiles/{pathology}_area_kde.pkl")

            self.kde_center[pathology] = kde_center
            self.kde_aspect_ratio[pathology] = kde_aspect_ratio
            self.kde_area[pathology] = kde_area

    def sample_bounding_boxes(self, pathology: int = 0):
        # Sample new bounding boxes from the estimated distribution
        pathology = pathology_dict[pathology]
        sampled_centers = self.kde_center[pathology].sample(self.batch_size)
        sampled_aspect_ratios = self.kde_aspect_ratio[pathology].sample(self.batch_size)
        sampled_areas = self.kde_area[pathology].sample(self.batch_size)

        sampled_centers = torch.from_numpy(sampled_centers).to(torch.float32)
        sampled_aspect_ratios = torch.from_numpy(sampled_aspect_ratios).to(torch.float32)
        sampled_areas = torch.from_numpy(sampled_areas).to(torch.float32)

        # Convert aspect_ratio and area to width and height
        sampled_heights = torch.sqrt(sampled_aspect_ratios * sampled_areas).flatten()
        sampled_widths = torch.sqrt(sampled_areas / sampled_aspect_ratios).flatten()

        # Convert center_x, center_y to x1, y1, x2, y2
        sampled_x1 = sampled_centers[:, 0] - 0.5 * sampled_widths
        sampled_y1 = sampled_centers[:, 1] - 0.5 * sampled_heights
        sampled_x2 = sampled_centers[:, 0] + 0.5 * sampled_widths
        sampled_y2 = sampled_centers[:, 1] + 0.5 * sampled_heights

        # Create the sampled bounding boxes
        sampled_boxes = torch.stack([sampled_x1, sampled_y1, sampled_x2, sampled_y2], dim=1)
        return sampled_boxes