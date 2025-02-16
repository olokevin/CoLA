DEVICE=${DEVICE-"0"}
IFS=',' read -ra array <<< "$DEVICE"
NGPU="${#array[@]}"
PORT=$(($RANDOM + 10000))

RUN_NAME=${RUN_NAME:-"None"}
CONFIG_NAME=${CONFIG_NAME:-"cola_7b"}
LR=${LR:-"0.001"}
WD=${WD:-"0.01"}
GC=${GC:-"0.5"}
BZ=${BZ:-"16"}
CONTINUE=${CONTINUE:-"none"}
if [ "${CONTINUE}" != "none" ]; then
    readonly continue_from_flag="--continue_from=$CONTINUE"
else
    readonly continue_from_flag=""
fi

RUN_NAME=$CONFIG_NAME-LR-$LR
TAG=${TAG:-"none"}
if [ "${TAG}" != "none" ]; then
    RUN_NAME=$TAG-$RUN_NAME
fi
STEPS=${STEPS:-"150000"}
if [ "${STEPS}" != "150000" ]; then
    RUN_NAME=$RUN_NAME-STEPS-$STEPS
fi
WU=${WU:-"15000"}
if [ "${WU}" != "15000" ]; then
    RUN_NAME=$RUN_NAME-WU-$WU
fi

CUDA_VISIBLE_DEVICES=$DEVICE torchrun --standalone --nproc-per-node=$NGPU --master-port=$PORT main.py \
    --model_type cola \
    --model_config cola_configs/$CONFIG_NAME.json \
    --lr $LR \
    --optimizer adamw \
    --batch_size $BZ \
    --total_batch_size 512 \
    --num_training_steps $STEPS \
    --warmup_steps $WU \
    --weight_decay $WD \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 2000 \
    --grad_clipping $GC \
    --run_name $RUN_NAME \
    --wandb_project cola-7b \
    $continue_from_flag \
    > /results/cola/$RUN_NAME.log 2>&1 &