#!/usr/bin/env bash
#$ -N create_data
#$ -cwd
#$ -j y
#$ -S /bin/bash

PROBLEM=translate_csen_czeng

HOME=$(pwd)

deactivate
source $HOME/.cpu-env/bin/activate

TMP_DIR=$HOME/data/tmp
DATA_DIR=$HOME/data/${PROBLEM}
USR_DIR=$HOME/t2t-str

mkdir -p ${TMP_DIR}

# Generate data
if [ ! -d "${DATA_DIR}" ]; then
  mkdir -p ${DATA_DIR}
  t2t-datagen \
    --data_dir=${DATA_DIR} \
    --tmp_dir=${TMP_DIR} \
    --problem=${PROBLEM} \
    --t2t_usr_dir=${USR_DIR}
fi