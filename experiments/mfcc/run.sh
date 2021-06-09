#!/bin/bash 

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
wl=0.035   # Window length in seconds.
step=0.01  # Step size in seconds.
if [ $stage -le 0 ]; then
    for corpus in ctimit ffmtimit ntimit stctimit timit wtimit; do
        echo "Extracting MFCC features for ${corpus}..."
        gen_librosa_feats.py \
            --ftype mfcc --config configs/feats/mfcc.yaml \
            --step $step --wl $wl --n_jobs $nj --disable-progress \
            $FEATS_DIR/${corpus}/mfcc $DATA_DIR/${corpus}/wav/*.wav \
            > logs/extract_mfcc_${corpus}.stdout \
            2> logs/extract_mfcc_${corpus}.stderr
    done
fi


##############################################################################
# Run binary classification tasks.
##############################################################################
if [ $stage -le 1 ]; then
    echo "$0: Preparing config files..."
    gen_config_files.py \
        --context-size 5 $FEATS_DIR mfcc configs/tasks $DATA_DIR
fi


if [ $stage -le 2 ]; then
    echo "$0: Running binary classification experiments..."
    for config in `ls configs/tasks/*.yaml`; do
	bn=${config##*/}
	name=${bn%.yaml}
	run_probe_exp.py \
	    --n-jobs $nj $config \
	    > logs/${name}.stdout \
	    2> logs/${name}.stderr &
    done
    wait
fi
