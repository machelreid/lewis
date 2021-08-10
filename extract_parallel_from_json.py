import argparse
from genericpath import exists
import json
import tqdm
import os


def read_json(path):
    with open(path, "r") as f:
        file = [json.loads(i) for i in f.readlines()]
    return file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, metavar="STR")
    parser.add_argument("--output_path", type=str, metavar="STR")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    file = read_json(args.input_json)

    output_parallel = {"d1": [], "d0": []}
    output_mask_and_orig = {"d1": [], "d0": []}
    output_independent_examples = {"d1": [], "d0": []}

    for line in tqdm.tqdm(file):
        dmn = line["original_domain"]
        admn = abs(line["original_domain"] - 1)
        src_len = line[f"d{dmn}"].__len__() - 1
        tgt_len = line[f"d{admn}"].__len__() - 1
        if line[f"d{admn}"] == []:
            continue
        for i in range(max(len(line["d0"]), len(line["d1"])) + 1):
            if i == 0:
                try:
                    output_parallel[f"d{admn}"].append(
                        line["original"] + "\t" + line[f"d{admn}"][0]
                    )
                except IndexError:
                    continue
                output_parallel[f"d{dmn}"].append(
                    line[f"d{admn}"][0] + "\t" + line["original"]
                )

                output_mask_and_orig[f"d{admn}"].append(
                    line["intermediate"]
                    + " <|endoftext|> "
                    + line["original"]
                    + "\t"
                    + line[f"d{admn}"][0]
                )
                output_mask_and_orig[f"d{dmn}"].append(
                    line["intermediate"]
                    + " <|endoftext|> "
                    + line[f"d{admn}"][0]
                    + "\t"
                    + line[f"original"]
                )
            else:
                try:
                    output_parallel[f"d{admn}"].append(
                        line[f"d{dmn}"][min(src_len, i)]
                        + "\t"
                        + line[f"d{admn}"][min(tgt_len, i)]
                    )
                except IndexError:
                    continue
                output_parallel[f"d{dmn}"].append(
                    line[f"d{admn}"][min(tgt_len, i)]
                    + "\t"
                    + line[f"d{dmn}"][min(src_len, i)]
                )

                output_mask_and_orig[f"d{admn}"].append(
                    line["intermediate"]
                    + " <|endoftext|> "
                    + line[f"d{dmn}"][min(src_len, i)]
                    + "\t"
                    + line[f"d{admn}"][min(tgt_len, i)]
                )
                output_mask_and_orig[f"d{dmn}"].append(
                    line["intermediate"]
                    + " <|endoftext|> "
                    + line[f"d{admn}"][min(tgt_len, i)]
                    + "\t"
                    + line[f"d{dmn}"][min(src_len, i)]
            )

    with open(os.path.join(args.output_path, "mask.d0"), "w") as f:
        f.write("\n".join(output_mask_and_orig["d0"]))

    with open(os.path.join(args.output_path, "mask.d1"), "w") as f:
        f.write("\n".join(output_mask_and_orig["d1"]))

    with open(os.path.join(args.output_path, "para.d0"), "w") as f:
        f.write("\n".join(output_parallel["d0"]))

    with open(os.path.join(args.output_path, "para.d1"), "w") as f:
        f.write("\n".join(output_parallel["d1"]))
