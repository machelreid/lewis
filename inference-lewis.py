import argparse
import torch

from transformers import RobertaForSequenceClassification, RobertaTokenizer
from fairseq.models.bart import BARTModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file_path", type=str, metavar="STR")
    parser.add_argument("--output_file_path", type=str, default="out_mask.txt")
    parser.add_argument(
        "--bart-mt-checkpoint-path",
        type=str,
        help="Path to fine-tuned bart-mt checkpoint",
    )
    parser.add_argument(
        "--bart-mt-data-bin-path",
        type=str,
        help="Path to fine-tuned bart-mt bin data",
    )
    parser.add_argument(
        "--hf-dump",
        type=str,
        metavar="STR",
        help="Path to RoBERTa-based classifier folder",
    )

    parser.add_argument(
        "--target_label_index",
        type=int,
        help="Index of the target label (for RoBERTA-classifer-based filtering)",
    )

    parser.add_argument("--bart_mt_beam_size", default=5, type=int)

    parser.add_argument("--bart_mt_top_k", default=5, type=int)

    args = parser.parse_args()

    if args.bart_mt_beam_size != args.bart_mt_top_k:
        raise NotImplementedError

    bart = BARTModel.from_pretrained(
        "/".join(args.bart_mt_checkpoint_path.split("/")[:-1]),
        checkpoint_file=args.bart_mt_checkpoint_path.split("/")[-1],
        data_name_or_path=args.bart_mt_data_bin_path,
    )

    target_idx = args.target_label_index

    bart.cuda()
    bart.eval()
    bart.half()
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    classifier = (
        RobertaForSequenceClassification.from_pretrained(
            f"{args.hf_dump}/pytorch_model.bin",
            config=f"{args.hf_dump}/config.json",
            output_attentions=True,
        )
        .half()
        .cuda()
        # .eval()
    )

    count = 1
    bsz = 32
    with open(args.input_file_path) as source, open(
        args.output_file_path, "w"
    ) as fout:
        sline = (
            source.readline()
            .strip()
            .replace("<mask>", " <mask> ")
            .replace("  ", " ")
        )
        slines = [sline]
        for sline in source:
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(
                        slines,
                        beam=args.bart_mt_beam_size,
                        topk=args.bart_mt_top_k,
                        match_source_len=False,
                        min_len=3,
                    )
                newb = []
                b = tokenizer(
                    hypotheses_batch, padding=True, return_tensors="pt",
                )
                total_classifier_output = classifier(
                    b["input_ids"].cuda(),
                    attention_mask=b["attention_mask"].cuda(),
                )[0]
                torch.cuda.empty_cache()
                for i in range(
                    0, len(hypotheses_batch), args.bart_mt_beam_size
                ):
                    _bool = False
                    outputs = hypotheses_batch[i : i + args.bart_mt_beam_size]
                    classifier_output = total_classifier_output[
                        i : i + args.bart_mt_beam_size
                    ]
                    classifier_output = (
                        classifier_output.argmax(-1).cpu().tolist()
                    )
                    for index, label in enumerate(classifier_output):
                        if label == target_idx:
                            _bool = True
                            newb.append(outputs[index])
                            break
                    if _bool == False:
                        newb.append(outputs[0])
                for hypothesis in newb:
                    fout.write(
                        hypothesis.replace("<mask>", "")
                        .replace("  ", " ")
                        .strip()
                        + "\n"
                    )
                    fout.flush()
                slines = []

            slines.append(sline.strip())
            count += 1
            print(count)
        if slines != []:
            hypotheses_batch = bart.sample(
                slines,
                beam=args.bart_mt_beam_size,
                match_source_len=False,
                topk=args.bart_mt_top_k,
                min_len=3,
            )
            newb = []
            b = tokenizer(hypotheses_batch, padding=True, return_tensors="pt",)
            total_classifier_output = classifier(
                b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda()
            )[0]
            torch.cuda.empty_cache()
            for i in range(0, len(hypotheses_batch), args.bart_mt_beam_size):
                _bool = False
                outputs = hypotheses_batch[i : i + args.bart_mt_beam_size]
                classifier_output = total_classifier_output[
                    i : i + args.bart_mt_beam_size
                ]
                classifier_output = classifier_output.argmax(-1).cpu().tolist()
                for index, label in enumerate(classifier_output):
                    if label == target_idx:
                        _bool = True
                        newb.append(outputs[index])
                        break
                if _bool == False:
                    newb.append(outputs[0])
            for hypothesis in newb:
                fout.write(
                    hypothesis.replace("<mask>", "").replace("  ", " ").strip()
                    + "\n"
                )
                fout.flush()
