#!/usr/bin/env bash
#$ -q gpu.q@dll[1256]
#$ -l gpu=1,gpu_cc_min6.1=1,gpu_ram=8G
#$ -cwd
#$ -j y
#$ -S /bin/bash

PROBLEM=translate_csen_czeng
MODEL=$1
HPARAMS=$2

HOME=$(pwd)

TMP_DIR=$HOME/data/tmp
DATA_DIR=$HOME/data/${PROBLEM}
TRAIN_DIR=$HOME/train_data/${PROBLEM}/${MODEL}-${HPARAMS}
USR_DIR=$HOME/t2t-str

# Decode
BEAM_SIZE=4
ALPHA=0.6
DECODE_SRC_FILE=${TMP_DIR}/data.export-format/09decode.cs
DECODE_TGT_FILE=${TMP_DIR}/data.export-format/09decode.en
DECODE_TO_FILE=${DATA_DIR}/${MODEL}-${HPARAMS}.en

t2t-decoder \
  --data_dir=${DATA_DIR} \
  --problems=${PROBLEM} \
  --model=${MODEL} \
  --hparams_set=${HPARAMS} \
  --output_dir=${TRAIN_DIR} \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA,batch_size=4" \
  --decode_from_file=${DECODE_SRC_FILE} \
  --decode_to_file=${DECODE_TO_FILE} \
  --t2t_usr_dir=${USR_DIR}

# Evaluate the BLEU score
cat ${DECODE_TO_FILE} | sacrebleu --tok none ${DECODE_TGT_FILE}
