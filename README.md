# LEWIS
Official code for LEWIS, from:  "[LEWIS: Levenshtein Editing for Unsupervised Text Style Transfer](https://machelreid.github.io/resources/reid21lewis.pdf)", ACL-IJCNLP 2021 Findings by Machel Reid and Victor Zhong

## Setup

### I. Install Dependencies

```bash
conda create -n lewis python=3.7
conda activate lewis
conda install pytorch cudatoolkit=10.2 -c pytorch
pip install transformers
pip install python-Levenshtein

# install fairseq
cd ~
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# for evaluation
pip install sacrebleu
pip install sacremoses

git clone https://github.com/machelreid/lewis.git
cd lewis
cp -r fairseq ~/fairseq/

```

### II. Download required pre-trained models

```bash
bash download.sh path/to/data
```

### III. Add a dataset

We need to add our data. Create a folder in `path/to/data` and name it as desired. Inside, create the following:

```
├── <dataset-prefix>
│   ├── <domain-1-suffix>
│   │   ├── train.txt
│   │   ├── valid.txt
│   │   ├── test.txt
│   ├── <domain-2-suffix>
│   │   ├── train.txt
│   │   ├── valid.txt
│   │   ├── test.txt
```

Where `prefix` is the name you are giving to the dataset, and `train.txt`, `valid.txt` and `test.txt` inside each domain folder are simple text files with one example (sentence) per line.

## Training

### I. Train a RoBERta-based Style Classifier

Fine-tune a RoBERTa for style classification to later use attentions to identify sections/tokens of the text that are strongly correlated with a given style class. The fine-tuning step is implemented mainly on the folder `roberta_classifier`. The while process is done using `fairseq` and works as follows:

1. Generate pre-processed data for fine-tuning using  `preprocess-roberta-classifier.sh` script, which will load the raw txt data for each style class, concatenate it and add labels, and create binary files that `fairseq` can use for faster training.

   ```bash
   bash preprocess-roberta-classifier.sh path/to/data <prefix> <domain-1-suffix> <domain-2-suffix>
   ```

   The outputs of this process will be stored in `path/to/data/roberta-classifier/<prefix>`

2. Fine-tune RoBERTa using `fairseq` using the `train-roberta-classifier.sh` script.

   ```bash
   bash train-roberta-classifier.sh path/to/data <prefix>
   ```

   This will save the model checkpoints at `/path/to/data/roberta-classifier/<prefix>/checkpoints`

3. Once the model has been fine-tuned we can convert it to plain pytorch/HuggingFace with the `convert_roberta_to_pytorch.py` script, as folllows.

   ```bash
   python convert_roberta_original_pytorch_checkpoint_to_pytorch.py --roberta_checkpoint_path /path/to/data/roberta-classifier/<prefix>/checkpoints --pytorch_dump_folder_path /path/to/data/roberta-classifier-py/<prefix> --classification_head classification_head
   ```

   When running this script as indicated, it will automatically use `checkpoint_best.pt` file inside the checkpoint directory provided. After the model has been exported, we will use it to find the tokens that are correlated with a given style in step 3.

4. We can also use our fine-tuned model to run inference on new data with `run_inference.sh`.

   - [ ] Finish Documenting `run-inference.sh`

### II. Fine-tune denoising BARTs for each style class

The second step is to fine-tune BART in the task of denoising to have generative models for each one of our styles.

1. Use `preprocess-bart-denoising.sh` to generate `fairseq` binary data for each style class.

   ```bash
   bash preprocess-bart-denoising.sh path/to/data <prefix> <domain-1-suffix>
   bash preprocess-bart-denoising.sh path/to/data <prefix> <domain-2-suffix>
   ```

   This will output the binary data at `path/to/data/bart-denoising/<prefix>/<domain-sufix>`.

2. Use `train-bart-denoising.sh` to fine-tune BART on each domain.

   ```bash
   bash train-bart-denoising.sh path/to/data <prefix> <domain-sufix>
   ```

   This will output checkpoints and training logs at `path/to/data/bart-denoising/<prefix>/<domain-sufix>`

   - [ ] Improve training stop criterion for this script (perplexity on vaild seems a good alternative)

### III. Using the style-classifier and MLMs to synthesize data

At this point we have a fine-tuned RoBERTa model for style classification that has been exported to pytorch, and two fine-tuned generative BART models, one for each style class. Now we can combine these elements to synthesize data. We can use the `get_synthesized_data.py`script to do so.

```bash
python get_synthesized_data.py --d1_model_path path/to/data/bart-denoising/<prefix>/<domain-1-suffix>/checkpoints/checkpoint.pt --d2_model_path path/to/data/bart-denoising/<prefix>/<domain-1-suffix>/checkpoints/checkpoint.pt --d1_file /path/to/data/<prefix>/<domain-1-suffix>/train.txt --d2_file /path/to/data/<prefix>/<domain-2-suffix>/train.txt --out_file output/json/file.json --hf_dump path/to/data/roberta-classifier-py/<prefix>
```

This will:

- Load the provided model RoBERTa model using HuggingFace and use the values on the 10-th attention layer to identity slots to generate style-agnostic sentences.

- Load the provided fine-tuned BART models for each style label, and use them to in-fill alternatives for each slot previously generated by looking at the RoBERTa classifier attentions.

- Pass the generated alternatives through the RoBERTa classifier to make sure there is class consistency between the each original sentence an synthetic alternative.

- Generate a JSON output file where each entry is structured as follows:

  ```json
  {"intermediate": "<synthetic-sentence>",
   "original": "<original-sentence>",
   "original_domain": "<0 or 1>"",
  }
  ```

This process can be be quite slow since it involves generation and prediction with two large models. To improve speed, the inputs sharded if you are using a lot of GPUs (using sharding the process took about 5 hours on politeness, and under 2 hours on yelp). For sharding, take a look at `sharded_get_synthesized_data.sh`.

- [ ] Document/update code for sharding

We can then extract parallel data from the file synthetic data we just generated, such that we can use this to fine-tune out final components that will actually fill in the blanks.  We do this with `extract_parallel_from_json.py`

```bash
python extract_parallel_from_json.py --input_json path/to/synth/data.json --output_path path/to/data/<synth-prefix>
```

This process will create the `path/to/data/<synth-prefix>` folder and inside place the files `para.d0/d1` and `masks.d0/d1` with the parallel unmasked/masked data. This process should take ~1,5 hours on 4 GPUs.

### IV. Fine-tune BART for machine translation (editing)

1. The first step is to pre-process the parallel generated by the previous step in order to use it to train our BART editor. Use `preprocess-bart-mt.sh` as follows.

```bash
bash preprocess-bart-mt.sh path/to/data <synth-prefix> <domain_prefix> <valid-split-size>
```

This will generate the training data. This includes creating a new dictionary for BART (updating its original `dict.txt`) so that it also contains the `<mask>` token which we will be feeding. Training/validations splits will be created at `path/to/data/<synth-prefix>` and the rest of the output will be placed at `path/to/data/bart-mt/<synth-prefix>/<domain-prefix>`.

2. Train BART on the parallel data.

```bash
bash train-bart-mt.sh path/to/data <synth-prefix> <domain-prefix>
```

This process will likely require multiple GPUs, so please adapt the code as required for this. This script will output checkpoints and training logs at  `path/to/data/bart-mt/<synth-prefix>/<domain-prefix>`.

### V. Train RoBERTa-based tagger

Our final training step is to fine-tune a RoBERTa token-level classifier (or tagger).

1. The first step here is to generate the Levenshtein editing operations from the parallel data, which  the tagger will learn to propose edits. For this, use `preprocess-roberta-tagger.py` as follows.

```bash
python preprocess-roberta-tagger.py --path-to-parallel-data-txt path/to/domain/parallel/txt --mask_threshold <mask-threshold> --output_path /path/to/output/parallel/json
```

Where `--path-to-parallel-data-txt ` should point to one of the `.para` files created in step III-1, `--mask-threshold` controls which sentences in the parallel data we will use for training based on the number of `<mask>` tokens per sentence (a good rule of thumb is to set it to approximately be 1/3 of the average total sequence length on the data) and `--output_path`should point to a json file.

2. Train the RoBERTa tagger with `train-roberta-tagger.py `, as follows.

```bash
python train-roberta-tagger.py --path-to-parallel-data-json path/to/parallel/json --hf_dump path/to/data/roberta-classifier-py/<prefix>/checkpoint.pt --save_dir path/to/data/<synth-prefix>/roberta-tagger --epochs <epochx> --bsz <batch_size> --update_freq <update_freq> 
```

This will initialize the token-level classifier with the sentence-level classifier that we trained on step I, using a plain `roberta-base` should work equally well. 

1. Perform inference with the RoBERTa-based tagger, using `inference-roberta-tagger.py` as follows.

```bash
python inference-roberta-tagger.py --hf-dump path/to/data/<synth-prefix>/roberta-tagger --path-to-parallel-data-json path/to/parallel/json --bsz 50 --update-freq 1 --output-path output/path

```

This will load the model trained on the above step and use it to generate edit labels. The complete output will be placed on `output/path/edits.json` and the masks only will be placed in `output/path/masks.txt`.

## Inference

Once the `bart-mt` and `roberta-tagger` models have been trained, we are ready to generate data for style changing, using `inference-lewis.py`

```bash
python inference-lewis.py --input_file_path path/to/masks.txt --output_file_path <> --bart-mt-checkpoint-path path/to/data/bart-mt/<synth-prefix> --bart-mt-data-bin-path path/to/data/bart-mt/<synth-prefix>/bin --hf-dump path/to/data/roberta-classifier-py/<prefix> --target_label_index <0 or 1>
```

This script requires the masks file created in the previous step, the fine-tuned BART trained in IV,  and again the RoBERTA-classifier trained in step I. Also, note that `--bart_mt_beam_size` and `--bart_mt_top_k` default to 5.

We thank Edison Marrese-Taylor for cleaning up this repository and refactoring the code for public use.
