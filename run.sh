docker pull huggingface/transformers-pytorch-gpu

docker run --gpus all --runtime=nvidia -it -d --shm-size=2g --name yequan\
  -v /home/yequan/Project/CoLA:/workspace \
  huggingface/transformers-pytorch-gpu \
  bash

pip install loguru bitsandbytes wandb easydict

DEVICE=0,1 bash scripts/cola_scripts/cola60m.sh
DEVICE=0,1 bash scripts/cola_m_scripts/colam60m.sh

ps aux | grep torchrun

