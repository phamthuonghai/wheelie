#!/usr/bin/env bash

TMP_DIR=../data/tmp/data.export-format

for ((i=0;i<=9;i++)); do
    cat ${TMP_DIR}/0${i}train | ./convert_czeng16_to_17.pl > ${TMP_DIR}/0${i}train.filtered
    mv ${TMP_DIR}/0${i}train ${TMP_DIR}/0${i}train.bk
    mv ${TMP_DIR}/0${i}train.filtered ${TMP_DIR}/0${i}train
done
