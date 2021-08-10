SRC=$1
TGT=$2
HYP=$3

cat $2 | sacremoses detokenize > /tmp/$(basename $2)
cat $1 | sacremoses detokenize > /tmp/$(basename $1) 
cat $3 | sacremoses detokenize > /tmp/$(basename $3)

echo "Accuracy: $(bash $HOME/storage/transferedit/roberta-finetuning/run_finetuning.sh $HYP $HOME/storage/transferedit/roberta-finetuning/jxhe.neg.labels $4 | tail -1 | cut -d '|' -f8 | cut -d " " -f3)" &

echo "Self-BERTScore: $(bert-score -r $SRC -c $HYP --lang en --rescale-with-baseline | rev | cut -d ':' -f1 | rev| xargs)"&

echo "BERTScore: $(bert-score -r $TGT -c $HYP --lang en --rescale-with-baseline | rev | cut -d ':' -f1 | rev | xargs)"&

echo "BLEU: $(sacrebleu -b /tmp/$(basename $TGT) < /tmp/$(basename $HYP))"

echo "Self-BLEU: $(sacrebleu -b /tmp/$(basename $SRC) < /tmp/$(basename $HYP))"
wait
