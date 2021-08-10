TOTAL_NUM_UPDATES=10000  # 10 epochs through IMDB for bsz 32
WARMUP_UPDATES=469      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
HEAD_NAME=classification_head     # Custom name for the classification head.
NUM_CLASSES=2           # Number of classes for the classification task.
MAX_SENTENCES=64         # Batch size.

DATA_PATH=$1
PREFIX=$2

INPUT_PATH="$DATA_PATH/roberta-classifier/$PREFIX"
OUTPUT_PATH="$INPUT_PATH"

# Add this parameter to the conmmand below
# to support the wandb backend, make sure you 
# install and setup your account first
# --wandb-project <wandb-project-name> 

fairseq-train "$INPUT_PATH/data-bin" \
    --restore-file "$DATA_PATH/roberta.base/model.pt" \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 8192 \
    --task sentence_prediction \
    --reset-optimizer \
    --reset-dataloader \
    --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 \
    --separator-token 2 \
    --arch roberta_base \
    --criterion sentence_prediction \
    --classification-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.1 \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay \
    --lr $LR \
    --total-num-update $TOTAL_NUM_UPDATES \
    --warmup-updates $WARMUP_UPDATES \
    --fp16 \
    --fp16-init-scale 4 \
    --threshold-loss-scale 1 \
    --fp16-scale-window 128 \
    --max-epoch 10 \
    --best-checkpoint-metric accuracy \
    --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --update-freq 1 \
    --save-dir "$OUTPUT_PATH/checkpoints"
    --tensorboard-logdir "$OUTPUT_PATH/tensorboard" | tee -a "$OUTPUT_PATH/train.log"


    
