import argparse
import re
from nltk.tokenize.moses import MosesDetokenizer


def delinearize(sent):
    sent = re.compile(r'(<[0-9]+>)').split(sent.strip())

    parsed = [(sent[i].strip(), int(sent[i+1][1:-1])) for i in range(0, len(sent)-1, 2)]

    parsed.sort(key=lambda x: x[1])

    return [w for w, _ in parsed]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('output_file')
    args = parser.parse_args()

    detok = MosesDetokenizer()

    fo = open(args.output_file, 'w')
    with open(args.data_file) as f:
        for line in f:
            fo.write(detok.detokenize(delinearize(line), return_str=True) + '\n')

    fo.close()
