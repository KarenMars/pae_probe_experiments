#!/usr/bin/env bash

set -e  # Exit on error

DATA_DIR=../../data/
FEATS_DIR=../../feats/
MODEL_NAME=wav2vec2-large


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
W2V2_MODEL=checkpoints/wav2vec2_big_960h.pt
DICT=checkpoints/dict.ltr.txt
if [ $stage -le 0 ]; then
    if [ ! -f $W2V2_MODEL ]; then
        echo "Downloading wav2vec 2.0 checkpoint..."
        mkdir -p checkpoints
	curl -#o $W2V2_MODEL https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_960h.pt
    fi
    if [ ! -f $DICT ]; then
	echo "Downloading wav2vec 2.0 vocab..."
	curl -#o $DICT https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt
    fi
fi


if [ $stage -le 1 ]; then
    for corpus in ctimit ffmtimit ntimit stctimit timit wtimit; do
        echo "Extracting wav2vec 2.0 features for ${corpus}..."
        export CUDA_VISIBLE_DEVICES=`free-gpu`
        gen_wav2vec_feats.py \
            --use-gpu --disable-progress \
	    --wav2vec2 --vocab $DICT \
	    $W2V2_MODEL $FEATS_DIR/$corpus/${MODEL_NAME} \
            $DATA_DIR/${corpus}/wav/*.wav \
            > logs/extract_${MODEL_NAME}_${corpus}.stdout \
            2> logs/extract_${MODEL_NAME}_${corpus}.stderr
done
fi


##############################################################################
# Run classification tasks.
##############################################################################
if [ $stage -le 2 ]; then
    echo "$0: Preparing config files..."
    # NOTE: Wav2vec 2.0 uses step size of 20 ms.
    gen_config_files.py \
	--step 0.020 \
        $FEATS_DIR $MODEL_NAME configs/tasks $DATA_DIR
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
