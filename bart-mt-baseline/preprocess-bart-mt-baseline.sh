wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

PREFIX=$1
for SPLIT in train valid
do
	for LANG in src tgt; do
		python -m examples.roberta.multiprocessing_bpe_encoder \
		--encoder-json encoder.json \
		--vocab-bpe vocab.bpe \
		--inputs "$PREFIX/$SPLIT.$LANG" \
		--outputs "$PREFIX/$SPLIT.bpe.$LANG" \
		--workers 60 \
		--keep-empty;
done
done


fairseq-preprocess \
	--trainpref "$PREFIX/train.bpe" \
	--validpref "$PREFIX/valid.bpe" \
	--destdir "$PREFIX/data-bin/" \
	--source-lang src \
	--target-lang tgt \
	--workers 60 \
	--srcdict bart.base/dict.txt \
	--tgtdict bart.base/dict.txt
