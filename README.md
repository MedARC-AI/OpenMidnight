<h1> OpenMidnight </h1>

<p align="center"><img width="735" height="400" alt="2PJpVl2k51pfjFg2Vs-oS" src="https://github.com/user-attachments/assets/fd00a9f5-3248-46ed-883a-f423e54ac2b6" /></p>
<p align="center">Fully open-source and improved replication of Kaiko.AI's pathology foundation model <a href="https://arxiv.org/abs/2504.05186v1">Midnight</a>.</p>
<div id="badges" align="center">
  <a href="https://sophont.med/blog/openmidnight"><img src="https://img.shields.io/badge/Blog-Training%20SOTA%20Pathology%20Foundation%20Model%20with%20%241.6k-111827?style=for-the-badge&logo=read.cv&logoColor=white" /> </a>
  <a href="https://discord.gg/tVR4TWnRM9"><img src="https://img.shields.io/badge/Discord-Collaborate%20with%20us-5865F2?style=for-the-badge&logo=discord&logoColor=white" /></a>     
</div>
<p align="center"> <a href="https://sophont.med">Sophont</a> · <a href="https://medarc.ai">MedARC</a> </p>

This is a publicly developed, open-source project by [MedARC](https://www.medarc.ai/). If you are interested in helping out, [join our Discord server](https://discord.gg/tVR4TWnRM9) and introduce yourself in our `#path-fm` channel.


## Features
- Trains faster with improved average benchmarking performance compared to the original Midnight-12K model (~3 days to train using 1×8×H100)
- Supports single‑GPU up to multi‑node training with FSDP
- Robust resuming from last checkpoint functionality if training gets interrupted
- Weights & Biases (wandb) logging for monitoring/tracking model training
- Optionally stream data from Hugging Face so no need to download any data in advance (TCGA-12K is approximately 12 TB)

# Installation

Clone the repository:

```bash
git clone https://github.com/MedARC-AI/openmidnight.git
```

Change into the directory, then run the installation script:

```bash
./install.sh
```

This will create a virtual environment, "pathologydino", with all necessary packages pre-installed, located as a .venv folder in the same directory as openmidnight. It will automatically detect your machine's CUDA version and install packages appropriately.

Note: We have only personally verified training works as intended with this repository using H100 GPUs.

```bash
source .venv/bin/activate
wandb init
```

By default, we log model training to wandb. Run `wandb init` inside of `openmidnight/` before starting your training run so that wandb is properly configured.

You can now run one of our `run*.sh` scripts to train your model (see Training section below), using the YAML config specified in that script.

Once you have successfully completed model training (or have downloaded [our pretrained checkpoint](https://huggingface.co/SophontAI/OpenMidnight/blob/main/teacher_checkpoint.pth)), you can evaluate using [Kaiko.AI's eva framework](https://github.com/kaiko-ai/eva) and the [Mahmood Lab's HEST benchmark](https://github.com/mahmoodlab/HEST) (skip to the Evaluation section below).

# Training

We use `run*.sh` to initiate training runs via torchrun. Depending on your compute setting, modify and run the corresponding shell script described in the relevant subsection below.

We use YAML configs for specifying important hyperparameters during training. These files are located in `dinov2/configs/train/` (not in `eval_configs/`, which stores YAML configs for eva evaluation benchmarking). Our replication checkpoint specifically used `dinov2/configs/train/vitg14_reg4.yaml`.

There are some variables that are specified in `run*.sh` directly (as opposed to the YAML config), such as the output directory for saving checkpoints, whether to enable resume functionality, and the specific CUDA devices you want to train with.

If you are getting rate limited by huggingface, one easy method to increase your rate is to first `export HF_TOKEN=<your HF token here>` before running your code (https://huggingface.co/settings/tokens).

## Dataset prep

If you are wanting to exactly replicate our checkpoint, note that we did not train via streaming from huggingface. This feature was subsequently added and needs to be enabled in your YAML config (`train.streaming_from_hf=[True,False]`). Unless you enable this, you will need to first locally download the TCGA-12K dataset (~12TB) and then use the scripts provided in `prepatching_scripts/` to create a txt file containing the svs filepaths and locations/magnitude from which to create patches on-the-fly during model training. You can alternatively use [our original sample_dataset_30.txt file](https://huggingface.co/SophontAI/OpenMidnight/blob/main/sample_dataset_30.txt), but note you would need to modify that txt to correct its use of absolute filepaths.

## Training Single GPU (Short Config)

```bash
./run_ablation.sh
```

This does finetuning with a YAML config (`dinov2/configs/train/vits14_reg_ablations.yaml`) tweaked to support an informative, short training run on a single GPU that can be completed in under 24 hours. We hope this can be particularly useful for debugging and ablation experiments.

## Training Single Node, Multi‑GPU (Full Reproduction)

```bash
./run_1node.sh
```

Train across multiple GPUs on a single node. Our released checkpoint used this script for training.

## Training Multi‑Node (Full Reproduction)

Same as training single‑node multi‑GPU, except increase `NNODES` in both `run_master_node.sh` and `run_other_nodes.sh` to the number of total nodes you are training across. Then run the corresponding scripts.

```bash
./run_master_node.sh # on master node
```
```bash
./run_other_nodes.sh # on non-master nodes
```

If during training you get HTTP Error 429, try reducing the number of workers (set in the YAML config) and lowering the DataLoader's `prefetch_factor` (defined in `dinov2/train/train.py`). This error can happen when Hugging Face is being pinged too frequently during streaming. Another solution is to [download the data locally](https://huggingface.co/datasets/medarc/TCGA-12K-parquet) and replace `medarc/TCGA-12K-parquet` in `dinov2/train/train.py` with the full path to your locally downloaded dataset folder.

# Methods / Training Recipe

Below is a high‑level overview of our training recipe, with particular attention to deviations from the original DINOv2 paper. For additional context, refer to the [Midnight paper](https://arxiv.org/abs/2504.05186).

- Base model + init
  - Student/teacher are ViT‑G/14 with 4 register tokens.
  - We initialize the student backbone from Meta’s DINOv2 ViT‑G/14 register checkpoint via `torch.hub`.
  - Heads are re‑initialized, as Meta only shared pretrained weights for their model backbone.

- Objectives and heads
  - DINO self‑distillation on CLS tokens, with 131072 prototypes and a 384‑dim bottleneck head. iBOT masked patch prediction on global crops. 
  - DINOv2's KoLeo regularization is replaced by a KDE‑based entropy regularizer as done in the Midnight paper.

- Data and augmentations
  - Streaming directly from a Parquet dataset of pre-patched TCGA slides hosted on Hugging Face (medarc/TCGA-12K-parquet). See `prepatching_scripts/` and read our [hf dataset card](https://huggingface.co/datasets/medarc/TCGA-12K-parquet/blob/main/README.md) for more information on pre-patching. We read `image_bytes`, decode to RGB, and apply DINO augmentations.
  - H&E augmentation: before normalization, images are converted to HED space and perturbed, then converted back to RGB. See `DataAugmentationDINO` in `dinov2/data/augmentations.py`.
  - Multi‑crop: 2 global crops (224) and multiple local crops (98). iBOT masks are sampled per‑image with ratios drawn uniformly from a min/max range.

- Optimization and schedules
  - LR is scaled with sqrt(batch/1024) from a base LR of 2e‑4. Midnight paper originally used a base LR of 3.5e-4 but we observed that this led to training collapse.
  - We train for 8000 “epochs” by schedule (1 epoch = 1250 steps), but early‑stop at 200 epochs. We found early stopping was necessary to prevent worsening downstream performance with longer model training.

- Checkpointing, evaluation, logging
  - We save LOCAL_STATE_DICT FSDP checkpoints per rank and tag `last_checkpoint.rank_*` files (for resuming functionality).
  - The teacher weights are exported every cfg.evaluation.eval_period_iterations steps to `output_dir/eval/training_<iter>/teacher_checkpoint.pth`.
  - Weights & Biases logging is enabled with a persistent run id stored in the output directory to support resuming, in case model training gets interrupted.

# Downstream Evaluation

## eva Benchmarks

First ensure you have a checkpoint ready to be evaluated. Place your .pth file for your teacher checkpoint in the /checkpoints folder. You can download our pretrained checkpoint here: https://huggingface.co/SophontAI/OpenMidnight/blob/main/teacher_checkpoint.pth

Then, `cd` into the same `openmidnight` folder cloned from our Installation steps and clone our modified GitHub repo forked from the original [kaiko-eva](https://github.com/kaiko-ai/eva):

```bash
cd openmidnight
source .venv/bin/activate
git clone https://github.com/MedARC-AI/eva-probe
```

Then, install the eva framework to enable use of the `eva` command in your terminal (using `--no-deps` because the `openmidnight` virtual environment already contains the necessary packages):

```bash
uv pip install -e './eva-probe[vision]' --no-deps
```

We provide every YAML config file we used in our replication in `openmidnight/eval_configs/`. Not every dataset permits automatic downloading. For datasets like BACH that do, we automatically download the dataset when you specify `DOWNLOAD_DATA=true` when calling `eva predict_fit`. For other datasets, follow the manual download steps described in the [eva datasets documentation](https://kaiko-ai.github.io/eva/main/datasets/) and place a folder containing the dataset into `openmidnight/eva-probe/data/`. Consult each dataset's specific YAML config for details on whether automatic downloading is supported; if not, modify the config to provide the path to your dataset prior to eva benchmarking.

Below are the steps for running the [BACH](https://kaiko-ai.github.io/eva/main/datasets/bach/) evaluation. If your teacher checkpoint is not stored as `openmidnight/checkpoints/teacher_epoch250000.pth`, you will first need to modify your eva YAML file's `checkpoint_path` variable to specify the path to your model weights.

```bash
cd eva-probe # should be located in openmidnight/eva-probe
CUDA_VISIBLE_DEVICES=0 DOWNLOAD_DATA=true eva predict_fit --config ../eval_configs/bach.yaml 
```

All eva evaluations should be run on a single GPU by setting `CUDA_VISIBLE_DEVICES=0`. We observed inconsistent and worse results when trying to evaluate using multiple GPUs.

## HEST Benchmark

First ensure you have a checkpoint ready to be evaluated. Place the .pth file for your teacher model in the /checkpoints folder. You can download our pretrained checkpoint here: https://huggingface.co/SophontAI/OpenMidnight/blob/main/teacher_checkpoint.pth

Then, `cd` into the same `openmidnight` folder cloned from our Installation steps and clone the [Mahmood Lab's HEST GitHub repo](https://github.com/mahmoodlab/HEST):

```bash
cd openmidnight
source .venv/bin/activate
git clone https://github.com/mahmoodlab/HEST.git
cd HEST
git checkout afd42c3143092c51e6bcc0f1df65bbf58a467e5e
cd .. # cd back to openmidnight/ for subsequent install steps
```

Then install HEST framework so that we can import invoke their benchmark function  (using `--no-deps` because the `openmidnight` virtual environment already contains the necessary packages).

```bash
uv pip install -e ./HEST --no-deps
```

Now uncomment out specific lines in the HEST YAML config to enable PCA dimensionality reduction, benchmark across all HEST datasets used in the original paper, and solely benchmark our DINOv2 model as the encoder:

```bash
sed -i -E '/^datasets:/,/^\]/{s/^([[:space:]]*)#([[:space:]]*)"([^"]+)",/\1"\3",/;s/^([[:space:]]*)"HCC"(,?)/\1# "HCC"\2/}; s/^[[:space:]]*#([[:space:]]*dimreduce:[[:space:]]*".*")/\1/; /^encoders:/,/^\]/{s/^([[:space:]]*)"resnet50"(,?)/\1# "resnet50"\2/}' ./HEST/bench_config/bench_config.yaml
```

Then finally run our HEST_evaluation.py script to benchmark your checkpoint. Running HEST_evaluation.py will automatically download the necessary preprocessed patches and patch encoders and output results into a new `openmidnight/eval` folder (specifically, the benchmark results dump to the `ST_pred_results/` subfolder).

```bash
python HEST_evaluation.py
```

## Related Work / Citation

This repository adapts and extends Meta AI's DINOv2 codebase and follows modifications introduced by Kaiko's Midnight work. If you use this repository or models in academic work, please cite their and our work:

Kaplan, D., Grandhi, R. S., Lane, C., Warner, B., Abraham, T. M., & Scotti, P. S. (2025). How to train a state-of-the-art pathology foundation model with $1.6k. Sophont. https://sophont.med/blog/openmidnight

```
@article{kaplan2025openmidnight,
            author = {Kaplan, Daniel and Grandhi, Ratna Sagari and Lane, Connor and Warner, Benjamin and Abraham, Tanishq Mathew and Scotti, Paul S.},
            title = {How to Train a State-of-the-Art Pathology Foundation Model with \$1.6k},
            year = {2025},
            url = {https://sophont.med/blog/openmidnight},
          }
```

Karasikov, M., van Doorn, J., Känzig, N., Erdal Cesur, M., Horlings, H. M., Berke, R., ... & Otálora, S. (2025). Training state-of-the-art pathology foundation models with orders of magnitude less data. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 573-583). Cham: Springer Nature Switzerland.

```
@inproceedings{karasikov2025training,
  title={Training state-of-the-art pathology foundation models with orders of magnitude less data},
  author={Karasikov, Mikhail and van Doorn, Joost and K{\"a}nzig, Nicolas and Erdal Cesur, Melis and Horlings, Hugo Mark and Berke, Robert and Tang, Fei and Ot{\'a}lora, Sebastian},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={573—583},
  year={2025},
  organization={Springer}
}
```

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., ... & Bojanowski, P. (2023). DINOv2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193.

```
@article{oquab2023dinov2,
            title={DINOv2: Learning Robust Visual Features without Supervision}, 
            author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
            year={2024},
            eprint={2304.07193},
            archivePrefix={arXiv},
            primaryClass={cs.CV},
            url={https://arxiv.org/abs/2304.07193}, 
}
```

