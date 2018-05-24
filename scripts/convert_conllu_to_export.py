import argparse
import io

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
    ret = []
    with io.open(file_path, 'r', encoding='utf-8') as f:
        sent_id = ''
        tmp_cont = []

        for line in f:
            line = line.strip()

            if len(line) == 0:
                if len(tmp_cont) > 1:
                    ret.append((sent_id, tmp_cont))
                    tmp_cont = []

            elif line[0].isdigit():
                data_line = line.split('\t')

                if len(data_line) != COL_COUNT:
                    raise 'Missing data: %s' % line

                tmp_cont.append(data_line)

            else:
                data_line = line.split()
                if len(data_line) == 4 and data_line[1] == 'sent_id':
                    sent_id = data_line[-1]

    if len(tmp_cont) > 1:
        ret.append((sent_id, tmp_cont))

    return ret


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('input_file')
    arg_parser.add_argument('output_file')

    args = arg_parser.parse_args()

    conllu_data = from_file(args.input_file)

    output_file = open(args.output_file, 'w', encoding='utf8')

    for _id, sentence in conllu_data:
        output = []
        for word in sentence:
            if word[FORM] == '|':
                word[FORM] = '\\bar'
                word[LEMMA] = '\\bar'

            # word-form|lemma|morphological-tag|index-in-sentence|index-of-governor|syntactic-function
            if word[ID].isnumeric():
                output.append('|'.join([word[FORM], word[LEMMA], word[XPOSTAG], word[ID], word[HEAD], '_']))
        output_file.write(' '.join(output) + '\n')

    output_file.close()
