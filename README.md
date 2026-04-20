# BUSGen

Official code for the paper:

**A Foundation Generative Model for Breast Ultrasound Image Analysis**
*Nature Biomedical Engineering*, 2026.
[[Paper]](https://www.nature.com/articles/s41551-026-01639-1) [[Demo]](https://aibus.bio)

The code base includes BUSGen training and sampling code. We utilize the publicly available BUSI dataset (containing 780 images) as a toy training set to showcase BUSGen's capabilities. Since the BUSI dataset was acquired using only one type of ultrasound device ("GE-LOGIQ-E9"), we demonstrate the generative capacity across multiple devices by augmenting the BUSI dataset with styles transferred to two other devices: "Mindray-M9" and "Siemens-ACUSON-NX3-Elite".

## Getting Started

### Installation

We use [uv](https://docs.astral.sh/uv/) for environment management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv --python 3.9
source .venv/bin/activate
uv pip install -e .

# For CUDA 11.3 (PyTorch 1.10.1), use:
uv pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Environment Configuration

Copy `.env.example` to `.env` and modify the paths to match your setup:

```bash
cp .env.example .env
```

Edit `.env` to set:
- `DATA_ROOT`: Path to the dataset root directory (containing the `BUSI/` folder)
- `CHECKPOINT_DIR`: Path to the directory for saving/loading model checkpoints
- `SAMPLED_DIR`: Path to the directory for saving sampled images

These environment variables override the default paths in config files. If `.env` is not provided, the paths in config JSON files are used as-is.

### Model Files

The `ModelFiles/` directory contains pre-fitted KDE (Kernel Density Estimation) models for bounding box sampling during inference. These pkl files are built from the [BUSI dataset](https://www.sciencedirect.com/science/article/pii/S2352340919312181) and model the spatial distribution of lesion bounding boxes (center position, aspect ratio, and area) for benign and malignant classes respectively.

To rebuild these files from your own dataset, use the provided script:
```bash
python build_kde_priors.py --json_file DatasetFiles/BUSI_DevAug.json --output_dir ModelFiles
```

### Data Preparation

Download the [BUSI dataset](https://www.sciencedirect.com/science/article/pii/S2352340919312181) and unzip it. Set `DATA_ROOT` in your `.env` file to the data directory path.

Expected data structure:
```
data/
    BUSI/
        benign/
            benign (1).png
            ...
            benign (437).png
        malignant/
            malignant (1).png
            ...
            malignant (210).png
        device_augmentation/
            benign/
                benign (1)_Mindray-M9.png
                benign (1)_Siemens-ACUSON-NX3-Elite.png
                ...
            malignant/
                malignant (1)_Mindray-M9.png
                malignant (1)_Siemens-ACUSON-NX3-Elite.png
                ...
```

## Training the Toy BUSGen Model

```bash
torchrun --nproc_per_node=<GPU_NUM> Main.py --config_file Configs/Toy_BUSGen_BUSI_DevAug.json
```

Replace `<GPU_NUM>` with the number of available GPUs. To train on the original BUSI dataset without device-style augmentation:
```bash
torchrun --nproc_per_node=<GPU_NUM> Main.py --config_file Configs/Toy_BUSGen_BUSI.json
```

You can also refer to `Run.sh` for example commands.

## Sampling from the Toy BUSGen Model

```bash
torchrun --nproc_per_node=<GPU_NUM> Main.py --config_file <CONFIG_PATH> --eval
```

Replace `<CONFIG_PATH>` with the config path used for training. To sample with different checkpoints, modify `test_load_weight` in the config file.

## Sampling from the BUSGen Model

We provide a demo notebook `Demo.ipynb` for interactive sampling from the full BUSGen model. You can also try the online demo at [https://aibus.bio](https://aibus.bio).

## Citation

If you find this work useful, please cite:

```bibtex
@article{yu2026busgen,
    title={A foundation generative model for breast ultrasound image analysis},
    author={Yu, Haojun and Li, Youcheng and Zhang, Nan and Niu, Zihan and Gong, Xuantong and Luo, Yanwen and Ye, Haotian and He, Siyu and Wu, Quanlin and Qin, Wangyan and Zhou, Mengyuan and Han, Jie and Tao, Jia and Zhao, Ziwei and Dai, Di and He, Di and Wang, Dong and Tang, Binghui and Huo, Ling and Zou, James and Zhu, Qingli and Wang, Yong and Wang, Liwei},
    journal={Nature Biomedical Engineering},
    year={2026},
    doi={10.1038/s41551-026-01639-1},
    publisher={Nature Publishing Group}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
