DEVICE=${DEVICE-"0"}
IFS=',' read -ra array <<< "$DEVICE"
NGPU="${#array[@]}"
PORT=$(($RANDOM + 10000))

RUN_NAME=${RUN_NAME:-"None"}
CONFIG_NAME=${CONFIG_NAME:-"colam_60m"}
LR=${LR:-"0.006"}
WD=${WD:-"0.01"}
GC=${GC:-"0.5"}
BZ=${BZ:-"256"}
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
STEPS=${STEPS:-"10000"}
if [ "${STEPS}" != "10000" ]; then
    RUN_NAME=$RUN_NAME-STEPS-$STEPS
fi
WU=${WU:-"2000"}
if [ "${WU}" != "2000" ]; then
    RUN_NAME=$RUN_NAME-WU-$WU
fi

CUDA_VISIBLE_DEVICES=$DEVICE torchrun --standalone --nproc-per-node=$NGPU --master-port=$PORT main.py \
    --model_type cola_m \
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
    --grad_clipping $GC \
    --run_name $RUN_NAME \
    > /results/cola/$RUN_NAME.log 2>&1 &