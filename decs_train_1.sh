#!/usr/bin/env bash
#$ -q gpu-ms.q
#$ -l gpu=1,gpu_cc_min6.1=1,gpu_ram=11G
#$ -cwd
#$ -j y
#$ -S /bin/bash

PROBLEM=translate_decs
MODEL=$1
HPARAMS=$2

HOME=$(pwd)

TMP_DIR=$HOME/data/tmp
DATA_DIR=$HOME/data/${PROBLEM}
TRAIN_DIR=$HOME/train_data/${PROBLEM}/${MODEL}-${HPARAMS}
USR_DIR=$HOME/t2t-str

echo $1-$2-$3

mkdir -p ${TMP_DIR} ${TRAIN_DIR}

# Generate data
if [ ! -d "${DATA_DIR}" ]; then
  mkdir -p ${DATA_DIR}

  t2t-datagen \
    --data_dir=${DATA_DIR} \
    --tmp_dir=${TMP_DIR} \
    --problem=${PROBLEM} \
    --t2t_usr_dir=${USR_DIR}
fi

# Train
t2t-trainer \
  --data_dir=${DATA_DIR} \
  --problems=${PROBLEM} \
  --model=${MODEL} \
  --hparams_set=${HPARAMS} \
  --hparams='batch_size=3072' \
  --keep_checkpoint_max=1 \
  --train_steps=${3} \
  --output_dir=${TRAIN_DIR} \
  --t2t_usr_dir=${USR_DIR}

./decs_decode_1.sh ${MODEL} ${HPARAMS} dev
./decs_decode_1.sh ${MODEL} ${HPARAMS} test
