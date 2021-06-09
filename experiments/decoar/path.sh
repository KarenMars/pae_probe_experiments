export PATH=$PWD/../../bin/:$PATH
export KALDI_ROOT=$PWD/kaldi
if [ -f $KALDI_ROOT/tools/env.sh ]; then
    . $KALDI_ROOT/tools/env.sh
fi
if [ -f $KALDI_ROOT/tools/config/common_path.sh ]; then
    . $KALDI_ROOT/tools/config/common_path.sh
fi
