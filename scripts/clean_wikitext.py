from collections import Counter
import os
import re


def shrink_spaces(s):
    return re.sub(r" +", " ", s).strip()


if __name__ == "__main__":

    for corpus, extension in [("2", "tokens"), ("103-raw", "raw")]:
        for partition in ["train", "valid", "test"]:
            print(f"wikitext-{corpus} ({partition})\n  cleaning data...")

            root = os.path.join("data", f"wikitext-{corpus}")
            rf = open(os.path.join(root, f"wiki.{partition}.{extension}"))
            wf = open(os.path.join(root, f"wiki.{partition}.clean"), "w")

            counts = None
            if partition == "train":
                counts = Counter()

            for line in rf:
                # Ignore wikipedia headers.
                line = line.strip()
                if line == "" or line.startswith("="):
                    continue

                # Change numbers, possibly with commas or decimal points, to `<num>`.
                line = re.sub(r"[0-9]+", " <num> ", line)
                line = re.sub(r"<num> @.@", " ", line)
                line = re.sub(r"<num>(?: <num>)+", " <num> ", line)

                # Remove any other weird characters.
                line = re.sub(r"[^A-Za-z<>'\.,;:]", " ", line)
                line = shrink_spaces(line)

                # Segment into sentences / independent clauses by tokenized '.', ';'.
                line_buf = []
                for token in line.split():
                    line_buf.append(token)

                    # Check for end of "sentence".
                    if token in [".", ";"]:
                        wf.write(" ".join(line_buf) + "\n")
                        line_buf = []

                    if partition == "train":
                        counts[token] += 1

            rf.close()
            wf.close()

            # Write tokens sorted by frequency.
            if partition == "train":
                print("  sorting vocabulary...")
                with open(os.path.join(root, f"wiki.vocab"), "w") as wf_voc:

                    # Special tokens first!
                    unk_count = counts["<unk>"]
                    del counts["<unk>"]
                    wf_voc.write("<pad>\t0\n")
                    wf_voc.write(f"<unk>\t{unk_count}\n")
                    wf_voc.write("<sos>\t0\n")
                    wf_voc.write("<eos>\t0\n")

                    for token, count in counts.most_common():
                        wf_voc.write(f"{token}\t{count}\n")
