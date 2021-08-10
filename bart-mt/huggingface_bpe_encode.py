from transformers import RobertaTokenizerFast
import sys
from multiprocessing import Pool
import tqdm

tknzr = RobertaTokenizerFast.from_pretrained("gpt2")
with open(sys.argv[1], "r") as f:
    file = [i.strip() for i in f.readlines()]


def _foo(line):
    return " ".join([str(i) for i in tknzr.encode(line)]).replace("50261", "<mask>")


if __name__ == "__main__":
    with Pool(15) as p:
        output_file = list(tqdm.tqdm(p.imap(_foo, file), total=len(file)))
with open(sys.argv[2], "w") as f:
    f.write("\n".join(output_file))
