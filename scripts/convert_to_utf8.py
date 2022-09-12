import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
args = parser.parse_args()


with open(args.input, "r", errors="ignore", encoding="utf8") as f_in:
    with open(args.output, "w", encoding="utf8") as f_out:
        for line in f_in:
            f_out.write(line)
