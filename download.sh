#!/bin/bash
DATA_PATH=$1

mkdir -p $DATA_PATH
cd $DATA_PATH

# download pre-trained BART
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz
tar -xvzf bart.base.tar.gz

# download pre-trained RoBERTa
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
tar -xvzf roberta.base.tar.gz

# download tokenizer for RoBERTA (which uses GPT-2 BPE)
mkdir gpt2_bpe
wget -P gpt2_bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json gpt2_bpe
wget -P gpt2_bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe gpt2_bpe/
wget -P gpt2_bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
