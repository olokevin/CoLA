# CoLA: Compute-Efficient Pre-Training of LLMs via Low-Rank Activation

## Getting Started

Recommend using huggingface transformer docker

To launch the pre-training scripts out-of-box, consider the following volume mount
- /workspace: the root directory where you clone this repo (i.e., /user/project/CoLA, then mount /user/project as /workspace)
- /datasets: optional, if you direct huggingface downloaded dataset to somewhere else other than default
- /results: optional, the logs will be saved to /results/cola

## Pre-Training Scripts
Run the following commands to pre-train CoLA(-M)-1B on 4 40GB A100 GPUs.

### CoLA
```
DEVICE=0,1,2,3 bash scripts/cola_scripts/cola1b.sh
```

### CoLA-M
```
DEVICE=0,1,2,3 bash scripts/cola_m_scripts/colam1b.sh
```
