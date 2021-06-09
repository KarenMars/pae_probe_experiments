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
# Install speech-representations
##############################################################################
if [ $stage -le 0 ]; then
    if [ ! -d kaldi ]; then
	echo "Building Kaldi..."
	./install_kaldi.sh \
	    > logs/install_kaldi.stdout \
	    2> logs/install_kaldi.stderr
    fi

    if [ ! -d speech-representations ]; then
	echo "Installing AWS speech-representations..."
	git clone https://github.com/awslabs/speech-representations.git
	cd speech-representations
	git checkout ae24ece
	pip install -e .
	cd ..
    fi
fi


##############################################################################
# Extract features
##############################################################################
DECOAR_MODEL=checkpoints/decoar-encoder-29b8e2ac.params
if [ $stage -le 1 ]; then
    if [ ! -f $DECOAR_MODEL ]; then
	echo "Downloading DeCoAR model checkpoint..."
	mkdir -p checkpoints
	wget \
	    -qO- https://apache-mxnet.s3-us-west-2.amazonaws.com/gluon/models/decoar-encoder-29b8e2ac.zip | \
        zcat > checkpoints/decoar-encoder-29b8e2ac.params
    fi
fi


if [ $stage -le 2 ]; then
    for corpus in ctimit ffmtimit ntimit stctimit timit wtimit; do
	echo "Extracting DeCoAR features for ${corpus}..."
	export CUDA_VISIBLE_DEVICES=`free-gpu`
	gen_decoar_feats.py \
	    --use-gpu --disable-progress \
	    $DECOAR_MODEL $FEATS_DIR/$corpus/decoar \
	    $DATA_DIR/${corpus}/wav/*.wav \
	    > logs/extract_decoar_${corpus}.stdout \
	    2> logs/extract_decoar_${corpus}.stderr
done
fi


##############################################################################
# Run classification tasks.
##############################################################################
if [ $stage -le 3 ]; then
    echo "$0: Preparing config files..."
    gen_config_files.py \
        $FEATS_DIR decoar configs/tasks $DATA_DIR
fi


if [ $stage -le 4 ]; then
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
