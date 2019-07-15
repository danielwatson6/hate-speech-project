from collections import Counter
import os
import re


def shrink_spaces(s):
    return re.sub(r" +", " ", s).strip()


if __name__ == "__main__":

    for corpus, extension in [("2", "tokens"), ("103-raw", "raw")]:
        for partition in ["train", "valid", "test"]:
            root = os.path.join("data", f"wikitext-{corpus}")
            print(f"wikitext-{corpus} ({partition})\n  cleaning data...")

            rf = open(os.path.join(root, f"wiki.{partition}.{extension}"))
            wf_inputs = open(os.path.join(root, f"wiki.{partition}.inputs"), "w")
            wf_labels = open(os.path.join(root, f"wiki.{partition}.labels"), "w")

            counts = None
            if partition == "train":
                counts = Counter()

            for line in rf:
                # Ignore wikipedia headers.
                line = line.lower().strip()
                if line == "" or line.startswith(" ="):
                    continue

                line = re.sub(r"[0-9]+", " <num> ", line)
                line = re.sub(r"<num> @.@", " ", line)
                line = re.sub(r"<num>(?: <num>)+", " <num> ", line)
                line = re.sub(r"[^a-z<>'\.,;:]", " ", line)
                line = shrink_spaces(line)

                # Segment into sentences / independent clauses by tokenised '.', ';'.
                lines = []
                line_buf = []
                for token in line.split():

                    # Check for end of "sentence".
                    if token in [".", ";"]:
                        line_buf.append(token)
                        lines.append(" ".join(line_buf))
                        line_buf = []
                    else:
                        line_buf.append(token)

                    if partition == "train":
                        if token not in counts:
                            counts[token] = 0
                        counts[token] += 1

                for line in lines:
                    no_punctuation = re.sub(r"[^A-Za-z<>]", " ", line)
                    no_punctuation = shrink_spaces(no_punctuation)
                    wf_inputs.write(no_punctuation + "\n")
                    wf_labels.write(line + "\n")

            # Write tokens sorted by frequency.
            if partition == "train":
                print("  sorting vocabulary...")
                with open(os.path.join(root, f"wiki.vocab.tsv"), "w") as wf_voc:

                    # Special tokens first!
                    unk_count = counts["<unk>"]
                    del counts["<unk>"]
                    wf_voc.write("<pad>\t0\n")
                    wf_voc.write(f"<unk>\t{unk_count}\n")
                    wf_voc.write("<sos>\t0\n")
                    wf_voc.write("<eos>\t0\n")

                    for token, count in counts.most_common():
                        wf_voc.write(f"{token}\t{count}\n")

            rf.close()
            wf_inputs.close()
            wf_labels.close()
