import os
import pickle

import numpy as np
import tensorflow as tf
import tqdm
from tensor2tensor.utils import usr_dir
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS

from utils.visualization import visualization, attention

matplotlib.rcParams.update({'font.size': 12})

# HOME_DIR = '/net/me/merkur3/pham/depatt'
HOME_DIR = '/home/phamthuonghai/Workspace/wheelie'
usr_dir.import_usr_dir(os.path.join(HOME_DIR, 't2t-str'))
MODEL_NAMES = ['Transformer\nbase'] + ['on layer %d' % i for i in range(6)]
MODEL_IDS = ['transformer_base'] + ['transformer_dep_parse_l%d' % i for i in range(6)]
TEST_FILE = os.path.join(HOME_DIR, 'data/tmp/data.export-format/09decode-test-1k.cs')
LIMIT = 100


def fake_name(p_name):
    if 'dep_parse' in p_name:
        p_name = p_name.split('_')
        return '_'.join(p_name[:1] + p_name[3:])
    else:
        return p_name


def get_att_mat(problem_name, model_name, hparams_set, input_sentences):
    CHECKPOINT = os.path.join(HOME_DIR, 'train_data', fake_name(problem_name), model_name + '-' + hparams_set)
    data_dir = os.path.join(HOME_DIR, 'data', problem_name)
    with tf.Graph().as_default() as graph:
        visualizer = visualization.AttentionVisualizer(hparams_set, model_name, data_dir, problem_name, beam_size=1)

        tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

        ret = [None for _ in range(6)]

        with tf.train.MonitoredTrainingSession(checkpoint_dir=CHECKPOINT, save_summaries_secs=0) as sess:
            for input_sentence in tqdm.tqdm(input_sentences):
                output_string, inp_text, out_text, att_mats = visualizer.get_vis_data_from_string(sess, input_sentence)

                att_mats = attention.resize(att_mats[0])
                for l_id in range(6):
                    t = np.reshape(att_mats[l_id], [8, -1])
                    if ret[l_id] is None:
                        ret[l_id] = t
                    else:
                        ret[l_id] = np.concatenate((ret[l_id], t), axis=-1)
    return ret


if __name__ == '__main__':

    att_mat_file = './data/att_mat_%d.pkl' % LIMIT

    if os.path.exists(att_mat_file):
        with open(att_mat_file, 'rb') as f:
            att = pickle.load(f)
    else:
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            inp_sentences = [line.strip() for line in f][:LIMIT]

        att = {
            'transformer_base': get_att_mat('translate_csen_czeng', 'transformer', 'transformer_base', inp_sentences)
        }

        for i in range(6):
               att['transformer_dep_parse_l%d' % i] = get_att_mat('translate_dep_parse_csen_czeng',
                                                                  'transformer_dep_parse',
                                                                  'transformer_dep_parse_l%d' % i, inp_sentences)

        with open(att_mat_file, 'wb') as f:
            pickle.dump(att, f)

    cols = ['{}'.format(col) for col in MODEL_NAMES]
    rows = ['Layer {}'.format(row) for row in range(6)]

    fig, axes = plt.subplots(nrows=6, ncols=len(MODEL_NAMES), figsize=(12, 8))

    for ax, col in zip(axes[0, :], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=0, size='large', ha='right')

    for m_id, m_names in enumerate(MODEL_IDS):
        for l_id in range(6):
            ax = axes[l_id, m_id]
            ax.hist(np.reshape(att[m_names][l_id], [-1]), bins=9, range=(0.1, 1),density=True,
                    color=CSS4_COLORS['sandybrown'] if l_id+1 == m_id else CSS4_COLORS['deepskyblue'])
            ax.set_ylim([0, 10])

    fig.text(0.6, 0.997, 'Syntax demanded from head', va='top', ha='center', size=14)
    fig.tight_layout()
    plt.savefig("att_dist.pdf")
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=8, figsize=(12, 1.6))

    for _id, ax in enumerate(axes):
        ax.set_title('Head #%d' % _id)

    for h_id in range(8):
        ax = axes[h_id]
        ax.hist(np.reshape(att['transformer_dep_parse_l4'][4][h_id], [-1]), bins=9, range=(0.1, 1),
                density=True, color=CSS4_COLORS['sandybrown'])
        ax.set_ylim([0, 10])

    fig.tight_layout()
    plt.savefig("att_dist_4.pdf")
    plt.show()
