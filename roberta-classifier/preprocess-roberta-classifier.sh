#!/bin/bash

DATA_PATH=$1
PREFIX=$2
DOMAIN_1=$3
DOMAIN_2=$4

INPUT_PATH="$DATA_PATH/$PREFIX"
OUTPUT_PATH="$DATA_PATH/roberta-classifier/$PREFIX"

mkdir -p "$OUTPUT_PATH"

for SPLIT in train valid; do
    cat $INPUT_PATH/$DOMAIN_1/$SPLIT.txt $INPUT_PATH/$DOMAIN_2/$SPLIT.txt > $INPUT_PATH/$SPLIT.txt
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json "$DATA_PATH/gpt2_bpe/encoder.json" \
        --vocab-bpe "$DATA_PATH/gpt2_bpe/vocab.bpe" \
        --inputs "$INPUT_PATH/$SPLIT.txt" \
        --outputs "$OUTPUT_PATH/$SPLIT.bpe" \
        --workers 60 \
        --keep-empty
done

fairseq-preprocess \
    --only-source \
    --trainpref "$OUTPUT_PATH/train.bpe" \
    --validpref "$OUTPUT_PATH/valid.bpe" \
    --destdir "$OUTPUT_PATH/data-bin/input0" \
    --workers 60 \
    --srcdict "$DATA_PATH/gpt2_bpe/dict.txt"

#rm "$OUTPUT_PATH/train.label"
#rm "$OUTPUT_PATH/valid.label"

# create actual labels for each style class
for i in $(seq $(wc -l $INPUT_PATH/$DOMAIN_1/train.txt | cut -d ' ' -f1)); do echo 0 >> $OUTPUT_PATH/train.label; done
for i in $(seq $(wc -l $INTPUT_PATH/$DOMAIN_1/valid.txt | cut -d ' ' -f1)); do echo 0 >> $OUTPUT_PATH/valid.label; done

for i in $(seq $(wc -l $INPUT_PATH/$DOMAIN_2/train.txt | cut -d ' ' -f1)); do echo 1 >> $OUTPUT_PATH/train.label; done
for i in $(seq $(wc -l $INPUT_PATH/$DOMAIN_2/valid.txt | cut -d ' ' -f1)); do echo 1 >> $OUTPUT_PATH/valid.label; done

rm -rf $FOLDER/data-bin/label
fairseq-preprocess \
    --only-source \
    --trainpref "$OUTPUT_PATH/train.label" \
    --validpref "$OUTPUT_PATH/valid.label" \
    --destdir "$OUTPUT_PATH/data-bin/label" \
    --workers 60

echo "Data in $INPUT_PATH has finished preprocessing, and the final output is in $OUTPUT_PATH/data-bin"
