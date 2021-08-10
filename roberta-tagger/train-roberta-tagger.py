from transformers import (
    RobertaTokenizerFast,
    RobertaForTokenClassification,
    RobertaConfig,
)
from torch.utils.data import Dataset, DataLoader
import torch
import json
import torch.optim as optim
import transformers
import argparse
import os
from tqdm import tqdm
import torch.nn as nn


def get_accuracy(logits, labels, attention_mask):
    out_val = 0
    for i, l in enumerate(logits):
        idx = attention_mask[i].tolist().count(1)
        out_val += (
            (l.argmax(-1)[:idx] == labels[i][:idx])
            .type(torch.float)
            .mean()
            .item()
        )
    return out_val / logits.shape[0]


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
    parser.add_argument("--save-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--bsz", type=int)
    parser.add_argument("--update-freq", type=int)
    parser.add_argument("--path-to-parallel-data-json")
    args = parser.parse_args()
    accumulation_steps = args.update_freq

    with open(args.path_to_parallel_data_json, "r") as f:
        data = [json.loads(i.strip()) for i in f.readlines()]
    text = [i["source"] for i in data]
    labels = [i["label"] for i in data]

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    dataset = NERDataset(
        text, labels, tokenizer, label_mapping=["KEEP", "MASK", "DELETE"]
    )
    model = RobertaForTokenClassification.from_pretrained(
        f"{args.hf_dump}/pytorch_model.bin",
        config=f"{args.hf_dump}/config.json",
        num_labels=len(dataset.label_mapping),
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
    ).cuda()
    optimizer = optim.Adam(model.parameters(), lr=3e-6)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=5000, num_training_steps=40000,
    )

    os.makedirs(args.save_dir, exist_ok=False)

    for _ in range(args.epochs):
        for i, example in tqdm(
            enumerate(DataLoader(dataset, batch_size=args.bsz, shuffle=True))
        ):
            optimizer.zero_grad()
            output = model(**example)
            output.loss = output.loss / accumulation_steps
            output.loss.backward()  # Backward pass
            if (
                i + 1
            ) % accumulation_steps == 0:  # Wait for several backward steps
                optimizer.step()  # Now we can do an optimizer step
                scheduler.step()
                optimizer.zero_grad()
            acc = get_accuracy(
                output.logits, example["labels"], example["attention_mask"]
            )
            tqdm.write(f"Accuracy: {acc}, LR: {scheduler.get_last_lr()}")
            if i % 500 == 0:
                print(output.logits.argmax(-1))
                torch.save(
                    model.state_dict(), f"{args.save_dir}/checkpoint{i}.pt"
                )
                torch.save(
                    model.state_dict(), f"{args.save_dir}/checkpoint_last.pt"
                )
