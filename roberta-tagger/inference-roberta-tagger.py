import os
from transformers import (
    RobertaTokenizerFast,
    RobertaForTokenClassification,
    RobertaConfig,
)
from torch.utils.data import Dataset, DataLoader
import torch
import json
import argparse


class NERDataset(Dataset):
    def __init__(
        self,
        text,
        labels=None,
        tokenizer=None,
        label_mapping=None,
        inference=False,
    ):
        self.text = text
        self.enc_text = tokenizer(text, truncation=True, padding=True)
        if label_mapping is None:
            self.label_mapping = list(set(" ".join(labels).split()))
        else:
            self.label_mapping = label_mapping
        if labels is not None:
            labels = [i.split() for i in labels]
            self.enc_labels = [
                [self.label_mapping.index("KEEP")]
                + [self.label_mapping.index(tok) for tok in ex]
                + [self.label_mapping.index("KEEP")]
                + [0] * self.enc_text["attention_mask"][i].count(0)
                for i, ex in enumerate(labels)
            ]
            self.inference = False
        else:
            self.inference = True

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]).cuda()
            for key, val in self.enc_text.items()
        }
        if not self.inference:
            item["labels"] = torch.tensor(self.enc_labels[idx]).cuda()
            assert len(self.enc_labels[idx]) == len(
                self.enc_text["input_ids"][idx]
            )
        return item

    def __len__(self):
        return len(self.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dump")
    parser.add_argument("--bsz", type=int)
    parser.add_argument("--path-to-parallel-data-json")
    parser.add_argument("--output-path")
    parser.add_argument("--update-freq", type=int, default=1)
    args = parser.parse_args()

    with open(args.path_to_parallel_data_json, "r") as f:
        text = [i.strip() for i in f.readlines()]
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    dataset = NERDataset(
        text, None, tokenizer, label_mapping=["KEEP", "MASK", "DELETE"]
    )
    config = RobertaConfig.from_pretrained("roberta-base")
    config.num_labels = 3
    model = RobertaForTokenClassification.from_pretrained(
        args.hf_dump, config=config
    ).cuda()
    print("Loaded roberta model")

    output_masks = open(os.path.join(args.output_path, "masks.txt"), "w")

    list_of_edits = []
    total_out_list = []
    for v, example in enumerate(DataLoader(dataset, batch_size=args.bsz)):
        output = model(**example)
        out_list = []
        for i, x in enumerate(output.logits.argmax(-1)):
            current_sent = []
            current_edits = []
            for j, tok in enumerate(tokenizer.tokenize(text[args.bsz * v + i])):
                val = x[j + 1]
                if val.item() == 0:
                    current_sent.append(tok)
                    current_edits.append("KEEP")
                elif val.item() == 1:
                    current_edits.append("REPLACE")
                    if current_sent == []:
                        current_sent.append("<mask>")
                    if current_sent[-1] != "<mask>":
                        current_sent.append("<mask>")
                elif val.item() == 2:
                    current_edits.append("DELETE")
                    pass
            current_sent = (
                tokenizer.convert_tokens_to_string(current_sent)
                .replace("<mask>", " <mask>")
                .replace("  ", " ")
                .strip()
            )
            out_list.append(current_sent)
            list_of_edits.append(" ".join(current_edits))
        total_out_list += out_list

        print("\n".join(out_list))
        output_masks.write("\n".join(out_list))

    output_masks.close()

    with open(os.path.join(args.output_path, "edits.json"), "w") as f:
        jsonlist = [
            json.dumps(
                {
                    "source": text[i],
                    "edit_ops": list_of_edits[i],
                    "output": total_out_list[i],
                }
            )
            for i in range(len(text))
        ]
        f.write("\n".join(jsonlist))
