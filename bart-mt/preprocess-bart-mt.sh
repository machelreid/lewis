#!/bin/bash

DATA_PATH=$1
PREFIX=$2
DOMAIN=$3
VALID_SIZE=$4

cp "$DATA_PATH/bart.base/dict.txt" "$DATA_PATH/bart.base/mask_dict.txt"
echo "<mask> 0" >> "$DATA_PATH/bart.base/mask_dict.txt"

# INPUT_PATH="$DATA_PATH/$PREFIX/$DOMAIN"
INPUT_PATH="$DATA_PATH/$PREFIX"
OUTPUT_PATH="$DATA_PATH/bart-mt/$PREFIX/$DOMAIN"

mkdir -p $OUTPUT_PATH

# split data into train/valid.
echo "  "
echo "Done"
cat "$INPUT_PATH/mask.d$DOMAIN" | cut -d$'\t' -f1 | head -"$VALID_SIZE"  > "$INPUT_PATH/valid.src"
cat "$INPUT_PATH/mask.d$DOMAIN" | cut -d$'\t' -f1 | tail --lines=+"$VALID_SIZE"  > "$INPUT_PATH/train.src"

cat "$INPUT_PATH/mask.d$DOMAIN" | cut -d$'\t' -f2 | head -"$VALID_SIZE"  > "$INPUT_PATH/valid.tgt"
cat "$INPUT_PATH/mask.d$DOMAIN" | cut -d$'\t' -f2 | tail --lines=+"$VALID_SIZE"  > "$INPUT_PATH/train.tgt"
echo "  "
echo "Done"

for SPLIT in train valid
do
	for LANG in src tgt; do
		python huggingface_bpe_encode.py \
		"$INPUT_PATH/$SPLIT.$LANG" \
		"$OUTPUT_PATH/$SPLIT.bpe.$LANG" &
done
done
wait
echo "Hello"
fairseq-preprocess \
	--trainpref "$OUTPUT_PATH/train.bpe" \
	--validpref "$OUTPUT_PATH/valid.bpe" \
	--destdir "$OUTPUT_PATH/data-bin/" \
	--source-lang src \
	--target-lang tgt \
	--workers 60 \
	--srcdict "$DATA_PATH/bart.base/mask_dict.txt" \
	--tgtdict "$DATA_PATH/bart.base/mask_dict.txt"
