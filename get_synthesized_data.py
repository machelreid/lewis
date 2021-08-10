import os
import json
import argparse
import time
from itertools import chain


from fairseq.models.bart import BARTModel
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from tqdm import tqdm
import torch



def get_file(path, sort=False):
    with open(path, "r") as f:
        file = [i.strip() for i in f.readlines()]
        if sort:
            file.sort(key=len)
    return file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--d1_model_path", type=str, metavar="STR")
    parser.add_argument("--d2_model_path", type=str, metavar="STR")
    parser.add_argument("--hf_dump", type=str, metavar="STR")
    parser.add_argument("--d1_file", type=str, metavar="STR")
    parser.add_argument("--d2_file", type=str, metavar="STR")
    parser.add_argument("--out_file", type=str, metavar="STR")
    args = parser.parse_args()

    d1_path = "/".join(args.d1_model_path.split("/")[:-1])
    d1_data_bin = "/".join(args.d1_model_path.split("/")[:-2]+["data-bin"])
    if d1_path == args.d1_model_path:
        d1_path = "."

    d2_path = "/".join(args.d2_model_path.split("/")[:-1])
    d2_data_bin = "/".join(args.d2_model_path.split("/")[:-2]+["data-bin"])
    if d2_path == args.d2_model_path:
        d2_path = "."

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    _time = time.time()
    print("Loading RoBERTa")
    classifier = (
        RobertaForSequenceClassification.from_pretrained(
            f"{args.hf_dump}/pytorch_model.bin",
            config=f"{args.hf_dump}/config.json",
            output_attentions=True,
        )
        .half()
        .cuda()
        .eval()
    )
    print(f"Loaded RoBERTa classifier in {time.time() - _time}s")

    d1_file = get_file(args.d1_file)
    d2_file = get_file(args.d2_file)

    print("Loading BARTs")
    _time = time.time()
    d1_bart = BARTModel.from_pretrained(
        d1_path,
        checkpoint_file=args.d1_model_path.split("/")[-1],
        data_name_or_path=d1_data_bin,
    )
    print(f"Loaded D1 BART in {time.time() - _time}s")
    _time = time.time()
    d2_bart = BARTModel.from_pretrained(
        d2_path,
        checkpoint_file=args.d2_model_path.split("/")[-1],
        data_name_or_path=d2_data_bin,
    )
    print(f"Loaded D2 BART in {time.time() - _time}s")

    mask_token = tokenizer.encode("<mask>", add_special_tokens=False)[0]
    output_sents = []
    print("generating masks")
    files = d1_file + d2_file
    print("Extracting SLOT tokens")
    for d, file in enumerate([d1_file, d2_file]):
        for i in tqdm(range(0, len(file), 32)):
            input_lines = file[i : i + 32]
            batch = tokenizer(
                input_lines, padding=True, return_tensors="pt", truncation=True
            )
            torch.cuda.empty_cache()
            classifier_output = classifier.forward(
                batch["input_ids"].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
            )
            torch.cuda.empty_cache()
            attentions = classifier_output["attentions"]
            lengths = [i.tolist().count(1) for i in batch["attention_mask"]]
            attentions = attentions[
                10
            ]  # 10 is chosen because it is the magical layer number of the grand elves
            cls_attns = attentions.max(1)[0][:, 0]
            for i, attn in enumerate(cls_attns):
                current_attn = attn[: lengths[i]][1:-1].softmax(-1)
                avg_value = current_attn.view(-1).mean().item()
                top_masks = ((current_attn > avg_value).nonzero().view(-1)) + 1
                torch.cuda.empty_cache()
                top_masks = top_masks.cpu().tolist()
                if len(top_masks) > min((lengths[i] - 2) // 3, 6):
                    top_masks = (
                        current_attn.topk(min((lengths[i] - 2) // 3, 6))[1] + 1
                    )
                    top_masks = top_masks.cpu().tolist()
                current_sent = batch["input_ids"][i][: lengths[i]]
                count = 0
                for index in top_masks:
                    if tokenizer.decode(current_sent[index]) not in [
                        " and",
                        " of",
                        " or",
                        " so",
                    ]:
                        current_sent[index] = mask_token
                        count += 1
                    else:
                        pass
                sent = (
                    tokenizer.decode(current_sent)[3:-4]
                    .replace("<mask>", " <mask>")
                    .strip()
                )
                if "<mask>" not in sent:
                    sent = sent + " <mask>."
                output_sents.append(sent)

    with open(args.out_file + ".intermediate_sentences.txt", "w") as f:
        f.write("\n".join(output_sents))

    output_json = []
    for target_idx, bart in enumerate((d1_bart, d2_bart)):
        torch.cuda.empty_cache()
        bart.eval()
        bart.half().cuda()
        for f in tqdm(range(0, len(output_sents), 7)):
            torch.cuda.empty_cache()
            y = bart.fill_mask(
                output_sents[f : f + 7], topk=5, match_source_len=False,
            )

            classifier_input = list(chain(*[[_[0] for _ in z] for z in y]))
            b = tokenizer(classifier_input, padding=True, return_tensors="pt",)
            total_classifier_output = classifier(
                b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda()
            )[0]
            torch.cuda.empty_cache()
            for i, x in enumerate(y):
                outputs = [_[0] for _ in x]
                classifier_output = total_classifier_output[i * 5 : (i * 5) + 5]
                if target_idx == 0:
                    current_json = {
                        "intermediate": output_sents[f + i],
                        "original": files[f + i],
                        "original_domain": 0 if f <= len(d1_file) else 1,
                    }
                else:
                    current_json = output_json[f + i]
                classifier_output = classifier_output.argmax(-1).cpu().tolist()
                sents = []
                for index, label in enumerate(classifier_output):
                    if label == target_idx:
                        sents.append(outputs[index])
                current_json[f"d{target_idx}"] = sents
                if target_idx == 0:
                    output_json.append(current_json)
                elif target_idx == 1:
                    with open(args.out_file, "a") as g:
                        g.write(json.dumps(current_json) + "\n")
