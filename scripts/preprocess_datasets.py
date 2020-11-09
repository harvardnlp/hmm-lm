
import torchtext


def process_file(infile, outfile, sep="<sep>"):
    lines = []
    # replace new lines (except for the ones in-between articles) with sep
    with open(infile, "r") as f:
        cur_line = []
        for line in f:
            text = line.strip().split()
            if len(text) < 2:
                # discard empty lines
                pass
            elif not cur_line:
                # first line
                cur_line.append(text)
            elif text[0] == "=" and text[1] != "=":
                # header
                lines.append(f" {sep} ".join(
                    " ".join(words) for words in cur_line
                ))
                cur_line = [text]
            else:
                cur_line.append(text)

    with open(outfile, "w") as g:
        g.write("\n".join(lines))

def remove_newline(infile, outfile, sep="<eos>"):
    lines = []
    # replace new lines (except for the ones in-between articles) with sep
    with open(infile, "r") as f:
        for line in f:
            text = line.strip()
            lines.append(text)
    with open(outfile, "w") as g:
        g.write(f" {sep} ".join(lines))


if __name__ == "__main__":
    infiles = [
        ".data/wikitext-2/wikitext-2/wiki.train.tokens",
        ".data/wikitext-2/wikitext-2/wiki.valid.tokens",
        ".data/wikitext-2/wikitext-2/wiki.test.tokens",
    ]
    outfiles = [f + ".flat" for f in infiles]

    infiles = [
        ".data/penn-treebank/ptb.train.txt",
        ".data/penn-treebank/ptb.valid.txt",
        ".data/penn-treebank/ptb.test.txt",
    ]
    outfiles = [f + ".flat" for f in infiles]

    for infile, outfile in zip(infiles, outfiles):
        remove_newline(infile, outfile, sep="<eos>")

