import argparse
import io

import tqdm

'''
EXPORT FORMAT
    Czech a-layer (surface-syntactic tree) in factored form:
        word-form|lemma|morphological-tag|index-in-sentence|index-of-governor|syntactic-function.
        Zachránil|zachránit_:W|VpYS---XR-AA---|1|0|Pred
    
CONLL FORMAT
    ID  FORM    LEMMA   UPOS    XPOS    FEATS   HEAD    DEPREL  DEPS    MISC
    1	štoček	štoček	NOUN	NNIS1-----A----	Animacy=Inan|Case=Nom|Gender=Masc|Number=Sing|Polarity=Pos	0	root	_	SpaceAfter=No
'''

# Constants for the column indices
COL_COUNT = 10
ID, FORM, LEMMA, UPOSTAG, XPOSTAG, FEATS, HEAD, DEPREL, DEPS, MISC = range(COL_COUNT)
COL_NAMES = u"ID,FORM,LEMMA,UPOSTAG,XPOSTAG,FEATS,HEAD,DEPREL,DEPS,MISC".split(u",")


def from_file(file_path):
    """ Parse data from CoNLL-U format file """
    with io.open(file_path, 'r', encoding='utf-8') as f:
        sent_id = ''
        tmp_cont = []

        for line in tqdm.tqdm(f):
            line = line.strip()

            if len(line) == 0:
                continue

            if line[0].isdigit():
                data_line = line.split('\t')

                if len(data_line) != COL_COUNT:
                    raise 'Missing data: %s' % line

                tmp_cont.append(data_line)

            else:
                data_line = line.split()
                if len(data_line) == 4 and data_line[1] == 'sent_id':
                    if len(tmp_cont) > 0:
                        yield (sent_id, tmp_cont)
                        tmp_cont = []
                    sent_id = data_line[-1]

    if len(tmp_cont) > 1:
        yield (sent_id, tmp_cont)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('input_file')
    arg_parser.add_argument('output_file')

    args = arg_parser.parse_args()

    conllu_data = from_file(args.input_file)

    output_file = open(args.output_file, 'w', encoding='utf8')

    cnt = 0

    for _id, sentence in conllu_data:
        output = []
        for word in sentence:
            if word[FORM] == '|':
                word[FORM] = '\\bar'
                word[LEMMA] = '\\bar'

            # word-form|lemma|morphological-tag|index-in-sentence|index-of-governor|syntactic-function
            if word[ID].isnumeric():
                output.append('|'.join([word[FORM], word[LEMMA], word[XPOSTAG], word[ID], word[HEAD], word[DEPREL]]))
        output_file.write(' '.join(output) + '\n')
        cnt += 1

    print('\nParsed %d sentences' % cnt)
    output_file.close()
