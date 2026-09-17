#!/usr/bin/env bash
###############################################################################
# Phase 0 — Environment setup & PLM-interact baseline
# Project : multimodal-ppi  (Protein–Protein Interaction model)
# Author  : Khalid Zaman (RA)      Supervisor: Prof. Zhaoxi Sun, SUAT
# Date    : 2026-09-09
#
# GOAL OF PHASE 0
#   Build a clean, reproducible software environment on the GPU cluster and
#   reproduce the published PLM-interact baseline by RUNNING its pre-trained
#   model (no training yet). This proves the whole toolchain works end to end.
#
# HOW TO READ THIS FILE
#   Lines starting with "#" are explanations (for the reader / supervisor).
#   Lines without "#" are the actual commands.
#   Each command is tagged with WHERE to run it:
#     [LOGIN NODE] = the machine you land on after SSH. Has internet, NO GPU.
#     [GPU JOB]    = must run on a GPU node, submitted through SLURM.
#
# WHY THESE EXACT VERSIONS (the key reason, in plain terms)
#   Our GPUs are RTX 50-series (5090 / 5080 / 5070 Ti) plus the RTX 6000D.
#   These use NVIDIA's "Blackwell" architecture, which needs CUDA 12.8 or newer.
#   So PyTorch MUST be a CUDA-12.8 build. torch 2.7.1+cu128 is the version
#   confirmed with the supervisor (see PROJECT_STATE.md). An older CUDA 11.8
#   build would install fine but then fail at run time with the error
#   "no kernel image is available for execution on the device".
###############################################################################


# =============================================================================
# Step 1 — Create a dedicated conda environment            [LOGIN NODE]
# WHY: giving this project its own environment (named "mmppi") means our exact
#      package versions can't be disturbed by other projects, and vice-versa.
#      PLM-interact requires Python 3.10, so we pin that here.
# =============================================================================
conda create -n mmppi python=3.10 -y

# Step 1b — Activate the environment                       [LOGIN NODE]
# WHY: switches your shell into "mmppi" so everything we install lands there.
#      Your prompt should change to show (mmppi).
conda activate mmppi


# =============================================================================
# Step 2 — Install PyTorch 2.7.1 for CUDA 12.8             [LOGIN NODE]
# WHY: PyTorch is the deep-learning engine. This build is matched to our
#      Blackwell GPUs and to the version confirmed with the supervisor.
#      NOTE: this is a large download (~2 GB), so give it a few minutes.
# =============================================================================
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Step 2b — Confirm the exact version                      [LOGIN NODE]
# WHY: we want to SEE "2.7.1+cu128" printed, proving the correct build installed.
#      On the login node there is no GPU, so we only check the version string
#      here; we verify the GPU itself inside a SLURM job in Step 6.
python -c "import torch; print(torch.__version__)"


# =============================================================================
# Step 3 — Get the PLM-interact baseline code              [LOGIN NODE]
# WHY: PLM-interact is the published baseline we are reproducing. We clone it
#      next to our own repo and install it so its scripts become runnable.
# =============================================================================
git clone https://github.com/liudan111/PLM-interact.git ~/Projects/PLM-interact
cd ~/Projects/PLM-interact

# Step 3b — One-line fix to PLM-interact's packaging file    [LOGIN NODE]
# WHY: their pyproject.toml lists a "docs = [...]" block inside the [project]
#      section, which modern setuptools rejects ("`project` must not contain
#      {'docs'} properties"). This inserts the correct header just above it so
#      "docs" becomes a proper optional-dependencies group and the file is valid.
#      (Edits our local clone only.)
sed -i '/^docs = \[/i [project.optional-dependencies]' pyproject.toml

# Step 3c — Install PLM-interact and ALL its dependencies    [LOGIN NODE]
# WHY: this reads the (now-valid) pyproject.toml and installs the exact library
#      versions the baseline needs — notably transformers==4.40.1 — plus numpy,
#      pandas, scipy, biopython, scikit-learn, huggingface_hub, etc. No separate
#      dependency install is needed. (torch 2.7.1+cu128 already satisfies
#      "torch>=2.0.1", so it is left as-is.)
pip install -e .


# =============================================================================
# Step 4 — Download the models ONCE, while we have internet [LOGIN NODE]
# WHY: cluster GPU nodes often have NO internet. So we download everything now,
#      on the login node, into an "offline" folder the GPU job will read from.
#        - facebook/esm2_t33_650M_UR50D        : the ESM-2 (650M) backbone.
#        - danliu1226/PLM-interact-650M-humanV11: the pre-trained PLM-interact
#          weights (trained on the human PPI benchmark) that we will run.
#
# CHINA NETWORK NOTE: this cluster cannot reach huggingface.co directly (PyPI and
#   GitHub work, but the Hugging Face Hub host is blocked). We route downloads
#   through the community mirror hf-mirror.com by setting HF_ENDPOINT. Everything
#   downloaded is identical; only the download host changes.
# =============================================================================
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
from huggingface_hub import snapshot_download
import os
base = os.path.expanduser("~/Projects/PLM-interact/offline")
snapshot_download("facebook/esm2_t33_650M_UR50D",
                  local_dir=f"{base}/esm2_t33_650M_UR50D")
snapshot_download("danliu1226/PLM-interact-650M-humanV11",
                  local_dir=f"{base}/PLM-interact-650M-humanV11")
print("Downloaded both models into:", base)
PY


# =============================================================================
# Step 5 — Build the smoke-test script                     [LOGIN NODE]
# WHY: PLM-interact ships a tiny demo, PLMinteract/inference/toy_inference.py,
#      that runs ONE hardcoded pair of proteins and prints one interaction
#      probability. We copy it to toy_smoke.py and point two of its hardcoded
#      paths at OUR downloaded folders (the ESM-2 backbone and the pre-trained
#      checkpoint). embedding_size is already 1280, so it is left as-is.
# =============================================================================
cd ~/Projects/PLM-interact
cp PLMinteract/inference/toy_inference.py toy_smoke.py
sed -i "s|^model_name=.*|model_name='$HOME/Projects/PLM-interact/offline/esm2_t33_650M_UR50D'|" toy_smoke.py
sed -i "s|^folder_huggingface_download=.*|folder_huggingface_download='$HOME/Projects/PLM-interact/offline/PLM-interact-650M-humanV11/'|" toy_smoke.py
grep -nE "^model_name=|^folder_huggingface_download=|^embedding_size" toy_smoke.py   # shows your two local paths + embedding_size 1280


# =============================================================================
# Step 6 — Run the smoke test on a GPU, THROUGH SLURM      [GPU JOB]
# WHY: the login node has no GPU. The job (phase0_toy_smoke.sbatch) requests one
#      GPU on the general partitions gpu_5070ti,gpu_5080,gpu_5090 (nodes GPU03/04/
#      06-10). We deliberately AVOID gpu_6000D / GPU05, reserved for training.
#      It turns on offline mode so transformers reads the models from local disk.
# Submit and read the result:
#     sbatch phase0_toy_smoke.sbatch      # queue the job -> prints a job id
#     squeue -u $USER                     # watch it: PD=waiting, R=running, gone=done
#     cat phase0_toy_*.out                # read the log once finished
# SUCCESS = a single probability between 0 and 1 between the "running"/"done"
#      lines. First successful run: 0.9919 for the demo pair, on an RTX 5090.
# =============================================================================
# (job script is the companion file: phase0_toy_smoke.sbatch)


###############################################################################
# NEXT — Phase 0 second half (added once the smoke test passes)
#   Reproduce the benchmark NUMBERS on the D-SCRIPT human test set: build a real
#   query,text,label CSV from the benchmark, run the same inference command, and
#   compare AUPR / AUROC against the paper's reported values.
###############################################################################
