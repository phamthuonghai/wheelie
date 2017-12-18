#!/bin/bash

PROBLEM=translate_csen_dep_raw
MODEL=transformer_dep_lrn
HPARAMS=transformer_base_single_gpu

HOME=`pwd`
REF_FILE=$HOME/data/plaintest_small.ref
DECODE_FILE=$HOME/data/deptest_small.src
TMP_DIR=$HOME/data/tmp
DATA_DIR=$HOME/data/$PROBLEM
TRAIN_DIR=$HOME/train_data/$PROBLEM/$MODEL-$HPARAMS
USR_DIR=$HOME/t2t-dep

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --t2t_usr_dir=$USR_DIR \
  --decode_hparams='use_last_position_only=True,batch_size=8' \
  --decode_from_file=$DECODE_FILE

sed -i '/^$/d' $DECODE_FILE.$MODEL.$HPARAMS.$PROBLEM.beam4.alpha0.6.decodes

cat $DECODE_FILE.$MODEL.$HPARAMS.$PROBLEM.beam4.alpha0.6.decodes | sacrebleu --force $REF_FILE
