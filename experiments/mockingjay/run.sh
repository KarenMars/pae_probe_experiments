#!/bin/bash 

set -e  # Exit on error

DATA_DIR=../../data/
FEATS_DIR=../../feats/


##############################################################################
# Configuration
##############################################################################
nj=40   # Number of parallel jobs for CPU operations.
stage=0

. path.sh

mkdir -p logs/


##############################################################################
# Install S3PRL
##############################################################################
if [ $stage -le 0 ]; then
    if [ ! -d s3prl ]; then
        echo "Installing S3PRL..."
        git clone https://github.com/s3prl/s3prl.git
        cd s3prl
        git checkout v0.1.0
	pip install -r requirements.txt
        cd ..
    fi
fi


##############################################################################
# Extract features
##############################################################################
MJ_MODEL=checkpoints/states-500000.ckpt
if [ $stage -le 1 ]; then
    for corpus in ctimit ffmtimit ntimit stctimit timit wtimit; do
        echo "Extracting Mockingjay features for ${corpus}..."
        export CUDA_VISIBLE_DEVICES=`free-gpu`
        gen_mockingjay_feats.py \
            --use-gpu --disable-progress \
	    $MJ_MODEL $FEATS_DIR/$corpus/mockingjay \
            $DATA_DIR/${corpus}/wav/*.wav \
            > logs/extract_mockingjay_${corpus}.stdout \
            2> logs/extract_mockingjay_${corpus}.stderr
done
fi


##############################################################################
# Run classification tasks.
##############################################################################
if [ $stage -le 2 ]; then
    echo "$0: Preparing config files..."
    # NOTE: MockingJay uses a nonstandard step size of 12.5 ms.
    gen_config_files.py \
        --step 0.0125 \
	$FEATS_DIR mockingjay configs/tasks $DATA_DIR
fi


if [ $stage -le 3 ]; then
    echo "$0: Running classification experiments..."
    for config in `ls configs/tasks/*.yaml`; do
        bn=${config##*/}
        name=${bn%.yaml}
        echo $name
        run_probe_exp.py \
            --n-jobs $nj $config \
            > logs/${name}.stdout \
            2> logs/${name}.stderr &
    done
    wait
fi
