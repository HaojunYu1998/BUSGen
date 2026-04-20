import io
import pickle
import functools
import numpy as np
from path import Path
from PIL import Image
from PIL import ImageDraw
from typing import Any, Callable, Optional, Tuple, Union, Dict, List, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import transforms


DEVICE_dict_toy = {
    "None": 0,
    "GE-LOGIQ-E9": 1,
    "Mindray-M9": 2,
    "Siemens-ACUSON-NX3-Elite": 3,
}
# NOTE: this is for in-house pretraining, not used in the toy example
DEVICE_dict_all = {
    "None": 0,
    "Siemens-ACUSON-Oxana": 1,
    "GE-LOGIQ-E9": 2,
    "Esaote-MyLab90": 3,
    "Esaote-MyLabClassC": 4,
    "Philips-EPIQ7": 5,
    "Sonoscape-S60": 6,
    "GE-EDVoluson-E8-Expert": 7,
    "Samsung-RS80A": 8,
    "Sonoscape-Clinic": 9,
    "TOSHIBA-Aplio-i700": 10,
    "Mindray-M9": 11,
    "Canon-Aplio-i800": 12,
    "Siemens-ACUSON-NX3-Elite": 13,
    "VINNO-G86": 14,
    "Mindray-Resona-R9": 15,
    # NOTE: 16 and 9 are repeated
    "SonoScape-Clinic": 16,
    "Mindray-NuewaI9": 17,
    "Mindray-Resona7": 18,
    "Samsung-Heraw10": 19
}


_LOCAL_PROCESS_GROUP = None
"""
A torch process group which only includes processes that on the same machine as the current process.
This variable is set when processes are spawned by `launch()` in "engine/launch.py".
"""

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert (
        _LOCAL_PROCESS_GROUP is not None
    ), "Local process group is not created! Please use launch() to spawn processes!"
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = dist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


def get_box_from_name(name):
    name = name.split("/")[-1].rstrip(".png")
    x1_idx = name.find("x1_")
    y1_idx = name.find("y1_")
    x2_idx = name.find("x2_")
    y2_idx = name.find("y2_")
    x1 = name[x1_idx + 3: x1_idx + 9]
    y1 = name[y1_idx + 3: y1_idx + 9]
    x2 = name[x2_idx + 3: x2_idx + 9]
    y2 = name[y2_idx + 3: y2_idx + 9]
    return [float(x1), float(y1), float(x2), float(y2)]


def print_(*args, **kwargs):
    rank = get_rank()
    if rank == 0:
        print(*args, **kwargs)


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


class RandomAccessVideo:
    """ Random access video frames reader and writer.
    NOTE: This class is not used in the toy example.
    """
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.head = dict()

    def _load_head(self) -> None:
        if not self.head:
            assert Path(self.file_name).exists(), f"{self.file_name} not exists!"
            with open(self.file_name, "rb") as fp:
                fp.seek(-4, 2)
                head_len = int.from_bytes(fp.read(4), "little")
                fp.seek(-4 - head_len, 2)
                self.head = pickle.loads(fp.read(head_len))

    def __len__(self) -> int:
        try:
            self._load_head()
        except OSError:
            return 0
        return len(self.head)

    def frame(self, idx: int) -> Union[None, np.ndarray]:
        try:
            self._load_head()
        except OSError:
            return None
        if idx not in self.head:
            raise IndexError("Index out of range")
        offset = self.head[idx]["offset"]
        length = self.head[idx]["length"]
        with open(self.file_name, "rb") as fp:
            fp.seek(offset, 0)
            bytes = fp.read(length)
            bytes = io.BytesIO(bytes)
        img = np.array(Image.open(bytes))
        bytes.close()
        return img

    def dump(self, frames: Iterable[np.ndarray]) -> None:
        with open(self.file_name, "wb") as fp:
            offset = 0
            for idx, frame in enumerate(frames):
                frame_pil = Image.fromarray(frame)
                img_byte_io = io.BytesIO()
                frame_pil.save(img_byte_io, format="jpeg")
                img_byte_arr = img_byte_io.getvalue()
                byte_len = len(img_byte_arr)
                self.head[idx] = {"offset": offset, "length": byte_len}
                fp.write(img_byte_arr)
                offset += byte_len
                img_byte_io.close()
            head_content_bytes = pickle.dumps(self.head)
            head_len = len(head_content_bytes)
            fp.write(head_content_bytes)
            fp.write(head_len.to_bytes(4, byteorder="little"))
