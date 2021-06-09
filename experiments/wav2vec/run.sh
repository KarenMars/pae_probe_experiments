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
# Extract features
##############################################################################
W2V_MODEL=checkpoints/wav2vec_large.pt
if [ $stage -le 0 ]; then
    if [ ! -f $W2V_MODEL ]; then
        echo "Downloading wav2vec checkpoint..."
        mkdir -p checkpoints
	curl -#o $W2V_MODEL https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt
    fi
fi


if [ $stage -le 1 ]; then
    for corpus in ctimit ffmtimit ntimit stctimit timit wtimit; do
        echo "Extracting wav2vec-large features for ${corpus}..."
        export CUDA_VISIBLE_DEVICES=`free-gpu`
        gen_wav2vec_feats.py \
            --use-gpu --disable-progress \
            $W2V_MODEL $FEATS_DIR/$corpus/wav2vec-large/ \
            $DATA_DIR/${corpus}/wav/*.wav \
            > logs/extract_wav2vec-large_${corpus}.stdout \
            2> logs/extract_wav2vec-large_${corpus}.stderr
done
fi


##############################################################################
# Run classification tasks.
##############################################################################
if [ $stage -le 2 ]; then
    echo "$0: Preparing config files..."
    gen_config_files.py \
	--step 0.010 \
        $FEATS_DIR wav2vec-large configs/tasks $DATA_DIR
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
