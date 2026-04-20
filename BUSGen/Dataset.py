import os
import torch
import json
import copy
import numpy as np
from PIL import Image
from typing import Any, Callable, Optional, Tuple, List
from torchvision.datasets.vision import VisionDataset

import Utils


class DDPMDataset(VisionDataset):

    def __init__(
        self,
        json_file: str,
        root: str,
        random_flip=True,
        random_crop=True,
        train: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=None)
        self.train = train
        self.json_file = json_file
        self.random_flip = random_flip
        self.random_crop = random_crop

        self.pathology_label = {"normal": -1, "benign": 0, "malignant": 1}
        # NOTE: we only provide training code of a toy model with 3 device types
        self.device_label = Utils.DEVICE_dict_toy
        self._prepare_data()

    def _prepare_data(self):
        assert os.path.exists(self.json_file)
        with open(self.json_file) as f:
            data = json.load(f)
        Utils.print_(f"Loaded {len(data)} images from {self.json_file}!")
        data = list(filter(lambda x: x["is_valid"], data))
        Utils.print_(f"Filter out {len(data)} images.")
        N_b = len([x for x in data if self.pathology_label[x["pathology"]] == 0])
        N_m = len([x for x in data if self.pathology_label[x["pathology"]] == 1])
        Utils.print_(f"Pathology label: {N_b} are benign; {N_m} are malignant.")
        Utils.print_(f"#"*10)
        
        self.data = data
        self.data_all = copy.deepcopy(data)
        
        Utils.print_(f"Statistics: {len(self.data_all)} valid images!")
        N_b = len([x for x in self.data_all if self.pathology_label[x["pathology"]] == 0])
        N_m = len([x for x in self.data_all if self.pathology_label[x["pathology"]] == 1])
        Utils.print_(f"Pathology label: {N_b} are benign; {N_m} are malignant.")
        self.max_num_lesion = max([len(x["lesion_box"]) for x in self.data_all])

    def resample(self, epoch, sample_interval, cls_resample):
        # NOTE: resample the video dataset for training efficiency
        data_vid = [x for x in self.data_all if ("video_file" in x) or ("video_folder" in x)]
        N1 = len(data_vid)
        data_img = [x for x in self.data_all if ("video_file" not in x) and ("video_folder" not in x)]
        data_vid = data_vid[epoch % sample_interval::sample_interval]
        N2 = len(data_vid)
        self.data = data_vid + data_img
        Utils.print_(f"Resample video frames: sampled {N2} frames from {N1} frames. Keep {len(self.data)} images.")
        
        if not cls_resample:
            return
        # NOTE: resample the pathology label for category balance
        data_b = [x for x in self.data if self.pathology_label[x["pathology"]] == 0]
        N1 = len(data_b)
        data_m = [x for x in self.data if self.pathology_label[x["pathology"]] == 1]
        N2 = len(data_m)
        N = min(N1, N2)
        data_b = list(np.random.permutation(data_b))
        data_m = list(np.random.permutation(data_m))
        self.data = data_b[:N] + data_m[:N]
        Utils.print_(f"Pathology label: {N1} are benign; {N2} are malignant.")
        Utils.print_(f"Resample pathology label: {N} are benign; {N} are malignant.")

    def _center_crop(self, img, info):
        H, W = img.shape[:2]
        min_size = min(H, W)
        crop_x1, crop_y1 = int((W - min_size) / 2), int((H - min_size) / 2)
        crop_x2, crop_y2 = crop_x1 + min_size, crop_y1 + min_size
        img = img[crop_y1: crop_y2, crop_x1: crop_x2]
        if self.valid_lesion_box:
            lesion_box = copy.deepcopy(list(info["lesion_box"].values()))
            boxes = torch.Tensor(lesion_box).float()
            boxes = boxes * torch.Tensor([W, H, W, H]).unsqueeze(0)
            boxes = boxes - torch.Tensor([crop_x1, crop_y1, crop_x1, crop_y1]).unsqueeze(0)
            boxes = boxes / min_size
            boxes = boxes.clamp(min=0.0, max=1.0).tolist()
            for i, k in enumerate(info["lesion_box"]):
                info["lesion_box"][k] = boxes[i]
        return img, info
        
    def _random_crop(self, img, info):
        """Random crop for data augmentation. 
        We keep all the lesion boxes in the cropped image.
        """
        H, W = img.shape[:2]
        CROP_FLAG = False
        if self.random_crop and self.valid_lesion_box:
            lesion_box = copy.deepcopy(list(info["lesion_box"].values()))
            boxes_max = torch.Tensor(lesion_box).float().max(dim=0).values.tolist()
            boxes_min = torch.Tensor(lesion_box).float().min(dim=0).values.tolist()
            x1, y1 = boxes_min[0], boxes_min[1]
            x2, y2 = boxes_max[2], boxes_max[3]
            assert x1 < x2 and y1 < y2, info["lesion_box"]
            assert x1 >= 0 and x2 <= 1 and y1 >= 0 and y2 <= 1, info["lesion_box"]
            x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
            for _ in range(100):
                crop_x1 = np.random.randint(0, max(1, x1))
                crop_y1 = np.random.randint(0, max(1, y1))
                crop_size_min = max(x2 - crop_x1, y2 - crop_y1)
                crop_size_max = min(W - crop_x1, H - crop_y1)
                if  crop_size_min < crop_size_max:
                    crop_size = np.random.randint(crop_size_min, crop_size_max)
                    CROP_FLAG = True
                    break
        if CROP_FLAG:
            crop_x2 = crop_x1 + crop_size
            crop_y2 = crop_y1 + crop_size
            img = img[crop_y1:crop_y2, crop_x1:crop_x2]
            H_new, W_new = img.shape[:2]
            assert H_new == W_new, f"{img.shape}, {info['file']}"
            for k, v in info["lesion_box"].items():
                x1, y1, x2, y2 = v
                x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
                x1 = max(0, (x1 - crop_x1) / W_new)
                y1 = max(0, (y1 - crop_y1) / H_new)
                x2 = min(1, (x2 - crop_x1) / W_new)
                y2 = min(1, (y2 - crop_y1) / H_new)
                info["lesion_box"][k] = [x1, y1, x2, y2]
        else:
            # NOTE: center crop is adopted when the lesion box is 
            # too close to the image boundary.
            img, info = self._center_crop(img, info)
        return img, info
    
    def _random_flip_horizon(self, img, info):
        if self.random_flip and np.random.random() > 0.5:
            img = np.flip(img, axis=1).copy()
            if self.valid_lesion_box:
                for k, v in info["lesion_box"].items():
                    x1, y1, x2, y2 = v
                    x1, x2 = 1 - x2, 1 - x1
                    info["lesion_box"][k] = [x1, y1, x2, y2]
        return img, info
    
    def _load_image(self, info):
        if "file" in info:
            img_file = info["file"]
            img_file = os.path.join(self.root, img_file)
            img = np.array(Image.open(img_file).convert("RGB")) # (H, W, 3)

        # NOTE: the following code is for pretraining, not used in the toy example
        elif "video_folder" in info: 
            rav_path = os.path.join(self.root, f"BUSV/{info['video_folder']}/video.ravideo")
            from Utils import RandomAccessVideo
            rav = RandomAccessVideo(rav_path)
            img = rav.frame(info["frame_idx"])
            if "crop_box_tight" in info and info["crop_box_tight"] != [-1,-1,-1,-1]:
                # NOTE: crop_box_tight is used for cropping the invalid margins 
                H, W = img.shape[:2]
                crop_x1, crop_y1, crop_x2, crop_y2 = info["crop_box_tight"]
                assert crop_x2 <= W, f"{crop_x2} <= {W}"
                assert crop_y2 <= H, f"{crop_y2} <= {H}"
                img = img[crop_y1:crop_y2, crop_x1:crop_x2]
                
                H_crop, W_crop = (crop_y2-crop_y1), (crop_x2-crop_x1)
                x1, y1, x2, y2 = info["lesion_box"]
                x1, y1, x2, y2 = int(x1*W), int(y1*H), int(x2*W), int(y2*H)
                x1, y1, x2, y2 = (x1-crop_x1)/W_crop, (y1-crop_y1)/H_crop, (x2-crop_x1)/W_crop, (y2-crop_y1)/H_crop
                x1, y1, x2, y2 = min(max(0, x1), 1), min(max(0, y1), 1), min(max(0, x2), 1), min(max(0, y2), 1)
                info["lesion_box"] = {"1": [x1, y1, x2, y2]}
            else:
                info["lesion_box"] = {"1": info["lesion_box"]}
        return img, info

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, label)
        """
        info = copy.deepcopy(self.data[index])
        # load image
        img, info = self._load_image(info)
        
        self.valid_lesion_box = len(info["lesion_box"]) > 0
        img, info = self._random_crop(img, info)
        img, info = self._random_flip_horizon(img, info)
        img = self.transform(img)

        label = {}
        # pathology label
        pathology = torch.tensor(self.pathology_label[info["pathology"]]).long()
        label["pathology"] = pathology
        # lesion box label
        lesion_box = torch.tensor(list(info["lesion_box"].values())).float().reshape(-1, 4)
        if len(lesion_box) < self.max_num_lesion:
            pad_box = torch.zeros(self.max_num_lesion - len(lesion_box), 4)
            lesion_box = torch.cat([lesion_box, pad_box], dim=0)
        valid_mask = torch.tensor([1] * len(info["lesion_box"]) + [0] * (self.max_num_lesion - len(info["lesion_box"])))
        lesion_box = torch.cat([lesion_box, valid_mask.unsqueeze(1)], dim=1)
        label["lesion_box"] = lesion_box
        # device label
        device = torch.tensor(self.device_label[info["device_type"]]).long()
        label["device"] = device
        return img, label

    def __len__(self) -> int:
        return len(self.data)
