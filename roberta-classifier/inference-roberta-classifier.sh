INPUTS=$1
LABELS=$2
PRETRAINED_MODEL_DIR=$3

rm -r test-bin
rm -r test-data
mkdir -p test-data

cp $INPUTS test-data
cp $LABELS test-data

mv test-data/$(echo $INPUTS | rev | cut -d "/" -f1 | rev) test-data/train.input0
mv test-data/$(echo $LABELS | rev | cut -d "/" -f1 | rev) test-data/train.label

# Dummy data
cp test-data/train.input0 test-data/dev.input0 
cp test-data/train.label test-data/dev.label 

echo "preprocessing data"
for SPLIT in train dev; do
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "test-data/$SPLIT.input0" \
        --outputs "test-data/$SPLIT.input0.bpe" \
        --workers 60 \
        --keep-empty 2> /dev/null

done

# Download fairseq dictionary.
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt' 2> /dev/null 

echo "binarizing data"
fairseq-preprocess \
    --only-source \
    --trainpref "test-data/train.input0.bpe" \
    --validpref "test-data/dev.input0.bpe" \
    --destdir "test-bin/input0" \
    --workers 60 \
    --srcdict dict.txt 1> /dev/null

fairseq-preprocess \
    --only-source \
    --trainpref "test-data/train.label" \
    --validpref "test-data/dev.label" \
    --destdir "test-bin/label" \
    --workers 60 \
    --srcdict $PRETRAINED_MODEL_DIR/data-bin/label/dict.txt 1> /dev/null


TOTAL_NUM_UPDATES=7812  # 10 epochs through IMDB for bsz 32
WARMUP_UPDATES=469      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
HEAD_NAME=classification_head     # Custom name for the classification head.
NUM_CLASSES=2           # Number of classes for the classification task.
MAX_SENTENCES=64         # Batch size.

CUDA_VISIBLE_DEVICES=3 fairseq-train test-bin/ \
    --restore-file $PRETRAINED_MODEL_DIR/checkpoints/checkpoint_best.pt \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 8192 \
    --task sentence_prediction \
    --reset-optimizer \
    --reset-dataloader \
    --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 
    --separator-token 2 \
    --arch roberta_base \
    --criterion sentence_prediction \
    --classification-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 
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
    -fp16-init-scale 4 \
    --threshold-loss-scale 1 \
    --fp16-scale-window 128 \
    --max-epoch 10 \
    --best-checkpoint-metric accuracy \
    --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --update-freq 1 \
    --validate-interval-updates 1 \
    --max-update 1 | tee test-data/run.log >/dev/null
cat test-data/run.log | grep "valid on" | sed 's/valid/test/g'
