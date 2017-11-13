#!/bin/bash

PROBLEM=translate_encs_small
MODEL=transformer
HPARAMS=transformer_base_single_gpu
HOME=`pwd`
TMP_DIR=$HOME/data/tmp
DATA_DIR=$HOME/data/encs_small
TRAIN_DIR=$HOME/train_data/$PROBLEM/$MODEL-$HPARAMS
USR_DIR=$HOME/t2t-dep

export CUDA_VISIBLE_DEVICES=0

mkdir -p $TMP_DIR $TRAIN_DIR

# Generate data
if [ ! -d "$DATA_DIR" ]; then
  mkdir -p $DATA_DIR

  t2t-datagen \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM \
    --t2t_usr_dir=$USR_DIR
fi

# Train
# *  In case of OOM, add --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --hparams='batch_size=1024,shared_embedding_and_softmax_weights=0' \
  --t2t_usr_dir=$USR_DIR
