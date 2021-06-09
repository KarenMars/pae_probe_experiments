#!/bin/bash
# Installation script for Kaldi.
set -e


#######################
# Config
#######################
NJOBS=40  # Number of parallel jobs for make.


#######################
# Clone repo.
#######################
SR_GIT=https://github.com/awslabs/speech-representations.git
SCRIPT_DIR=$(realpath $(dirname "$0"))
SR_DIR=$SCRIPT_DIR/speech-representations
SR_REVISION=e2f6d16

if [ ! -d $SR_DIR ]; then
    git clone $SR_GIT $SR_DIR
    cd $SR_DIR
    git checkout $SR_REVISION
    pip install -e .
    cd ..
fi

echo "Successfully installed speech-representations."
