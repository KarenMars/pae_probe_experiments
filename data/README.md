Data preparation
================
Overview
--------
This directory contains instructions and scripts for performing preprocessing
of of the 6 TIMIT variants from the releases distributed by [Linguistic Data Consortium](https://www.ldc.upenn.edu/):

- TIMIT ([LDC93S1](https://catalog.ldc.upenn.edu/LDC93S1))
- NTIMMIT ([LDC93S2](https://catalog.ldc.upenn.edu/LDC93S2))
- CTIMIT ([LDC96S30](https://catalog.ldc.upenn.edu/LDC96S30))
- FFMTIMIT ([LDC96S32](https://catalog.ldc.upenn.edu/LDC96S32))
- STCTIMIT ([LDC2008S03](https://catalog.ldc.upenn.edu/LDC2008S03))
- WTIMIT ([LDC2010S02](https://catalog.ldc.upenn.edu/LDC2010S02))


Instructions
------------
After obtaining the corpora from LDC, edit the contents of ``config.yaml`` so
that the paths point to the locations of the corpora on your filesystem. Then,
run:

    python3 prepare_data.py

This will iterate over the corpora, converting the audio files to 16 kHz,
monochannel WAV and the phones files to HTK labels files.


Directory structure
-------------------
After running ``prepare_data.py``, you should see the following directory structure:

- ``ctimit/wav/``  --  CTIMIT WAV files
- ``ctimit/phones/``  --  CTIMIT phone-level segmentation, stored as HTK label
  files
- ``ffmtimit/wav/``  --  FFTIMIT WAV files
- ``ffmtimit/phones/``  --  FFMTIMIT phone-level segmentation
- ``ntimit/wav/``  --  NTIMIT WAV files
- ``ntimit/phones/``  --  NTIMIT phone-level segmentation
- ``stctimit/wav/``  --  STCTIMIT WAV files
- ``stctimit/phones/``  --  STCTIMIT phone-level segmentation
- ``timit/wav/``  --  TIMIT WAV files
- ``timit/phones/``  --  TIMIT phone-level segmentation
- ``wtimit/wav/``  --  WTIMIT WAV files
- ``wtimit/phones/``  --  WTIMIT phone-level segmentation
- ``lists/phones.60-48-39.map``  --  mapping between canonical TIMIT phone sets
- ``lists/test_core.ids``  --  listing of recordings in TIMIT core test set
- ``lists/test_full.ids``  --  listing of recordings in TIMIT full test set
- ``lists/train.ids``  --  listing of recordings in TIMIT training set



File formats
------------
### WAV files

For each corpus, for each combination of speaker/sentence, there is a 16 kHz,
monochannel WAV file located at:

    <CORPUS>/wav/<SPEAKER>_<SENT>.wav

where:

- CORPUS  --  name of the corpus; e.g., ``timit``
- SPEAKER  --  speaker of sentence; e.g., ``MJSR0``
- SENT  --  sentence; e.g., ``SX204``

WAV files are converted from the original distribution format (WAV, FLAC, or
headerless raw files) using SoX.


### HTK label files

Phone-level segmentations are stored as HTK label files containing one phone
per line, each line having the form:

    <ONSET>\t<OFFSET>\t<PHONE>

where:

- ONSET  --  onset of phone in seconds from beginning of recording
- OFFSET  --  offset of phone in seconds from beginning of recording
- PHONE  --  phone

Two changes have been made to the segmentations from the original TIMIT phones
files:

- the three silence classes (``h#``, ``pau``, ``epi``) have been remapped to
  ``sil``
- the offset of the final phone **ALWAYS** corresponds to the duration of the
  recording

The naming convention is the same as for WAV files.


### .ids files

The TIMIT training, core test, and full test set membership are described in
``.ids`` files under the ``lists/`` directory. For each partition, there is an
``.ids`` file containing one recording per line; e.g.:

    FDHC0_SA1
    FDHC0_SA2
    FDHC0_SI1559
