INPUT_FILE=$1
MODEL_FILE=$2
DATA_BIN=$3
HF_DUMP=$4
OUTPUT_FILE=$5

shardize $INPUT_FILE 8 ./shards

for i in {0..7}; do
	rm shards/shard$i.out
	CUDA_VISIBLE_DEVICES=$i python get_synthesized_data.py --model-path $MODEL_FILE --hf-dump $HF_DUMP --input-filename shards/$(basename $INPUT_FILE)_sharded.$i --data-bin $DATA_BIN --use-pos-tagger --out-file shards/shard$i.out &
done
wait
rm $OUTPUT_FILE
for i in {0..7}; do 
	cat shards/shard$i.out >> $OUTPUT_FILE
done
