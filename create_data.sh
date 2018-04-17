#!/usr/bin/env bash
#$ -q gpu.q@dll[1256]
#$ -l gpu=1,gpu_cc_min6.1=1,gpu_ram=8G
#$ -N create_data
#$ -cwd
#$ -j y
#$ -S /bin/bash

PROBLEM=translate_csen_czeng

HOME=$(pwd)

TMP_DIR=$HOME/data/tmp
DATA_DIR=$HOME/data/${PROBLEM}
USR_DIR=$HOME/t2t-str

mkdir -p ${TMP_DIR}

# Generate data
mkdir -p ${DATA_DIR}
t2t-datagen \
--data_dir=${DATA_DIR} \
--tmp_dir=${TMP_DIR} \
--problem=${PROBLEM} \
--t2t_usr_dir=${USR_DIR}
