#!/bin/bash

DATA_PATH=$1
PREFIX=$2
DOMAIN=$3

INPUT_PATH="$DATA_PATH/$PREFIX/$DOMAIN"
OUTPUT_PATH="$DATA_PATH/bart-denoising/$PREFIX/$DOMAIN"

mkdir -p "$OUTPUT_PATH"

for SPLIT in train valid
do
	python -m examples.roberta.multiprocessing_bpe_encoder \
	--encoder-json "$DATA_PATH/gpt2_bpe/encoder.json" \
	--vocab-bpe "$DATA_PATH/gpt2_bpe/vocab.bpe" \
	--inputs "$INPUT_PATH/$SPLIT.txt" \
	--outputs "$OUTPUT_PATH/$SPLIT.bpe" \
	--workers 60 \
	--keep-empty;
done

fairseq-preprocess \
	--only-source \
	--trainpref "$OUTPUT_PATH/train.bpe" \
	--validpref "$OUTPUT_PATH/valid.bpe" \
	--destdir "$OUTPUT_PATH/data-bin/" \
	--workers 60 \
	--srcdict "$DATA_PATH/bart.base/dict.txt" \
	--tgtdict "$DATA_PATH/bart.base/dict.txt"

