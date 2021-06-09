#!/bin/bash 

set -e  # Exit on error

DATA_DIR=../../data/
FEATS_DIR=../../feats/
MODEL_NAME=vq-wav2vec_kmeans_roberta


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
W2V_MODEL=checkpoints/vq-wav2vec_kmeans.pt
ROBERTA_MODEL=checkpoints/bert_kmeans.pt
ROBERTA_VOCAB=checkpoints/dict.txt
if [ $stage -le 0 ]; then
    mkdir -p checkpoints
    if [ ! -f $W2V_MODEL ]; then
        echo "Downloading vq-wav2vec checkpoint..."
	curl -#o $W2V_MODEL https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt
    fi
    if [ ! -f $ROBERTA_MODEL ]; then
        echo "Downloading RoBERTa model and vocabulary..."
	curl -#o checkpoints/bert_kmeans.tar https://dl.fbaipublicfiles.com/fairseq/wav2vec/bert_kmeans.tar
	cd checkpoints/
	tar -xf bert_kmeans.tar
	rm bert_kmeans.tar
	cd ..
    fi    
fi


if [ $stage -le 1 ]; then
    for corpus in ctimit ffmtimit ntimit stctimit timit wtimit; do
        echo "Extracting features vq-wav2vec-kmeans + RoBERTa features for ${corpus}..."
        export CUDA_VISIBLE_DEVICES=`free-gpu`
        gen_wav2vec_feats.py \
            --use-gpu --disable-progress \
	    --roberta $ROBERTA_MODEL --vocab $ROBERTA_VOCAB \
            $W2V_MODEL $FEATS_DIR/$corpus/${MODEL_NAME} \
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
    gen_config_files.py \
	--step 0.010 \
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
