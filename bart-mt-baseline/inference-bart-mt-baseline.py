import torch
from fairseq.models.bart import BARTModel
import sys

bart = BARTModel.from_pretrained(
    "/".join(sys.argv[2].split("/")[:-1]),
    checkpoint_file=sys.argv[2].split("/")[-1],
    data_name_or_path=sys.argv[3],
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 32
with open(sys.argv[1]) as source, open("out.txt", "w") as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(
                    slines,
                    beam=5,
                )

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + "\n")
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(
            slines,
            beam=5,
        )
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + "\n")
            fout.flush()
