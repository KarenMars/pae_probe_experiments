#!/usr/bin/env python
"""Prepare TIMIT variants for experiments.

To run, edit the paths in ``config.yaml`` to point to the locations of the LDC
releases on your filesystem, then run:

    python prepare_data.py
"""
from abc import abstractmethod, ABC
import argparse
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import sys

import librosa
import yaml


THIS_DIR = Path(__file__).parent
CONFIG_PATH = Path(THIS_DIR, 'config.yaml')

def load_corpus_paths():
    """Load paths to corpora."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f)
    paths = {}
    for corpus in sorted(config):
        corpus_dir = Path(config[corpus])
        if not corpus_dir.exists():
            warning(f'Unable to locate "{corpus.upper()}" directory. Please '
                    f'check that the paths in "config.yaml" are correct.')
            continue
        paths[corpus] = corpus_dir
    return paths


def warning(msg):
    """Print warning message to STDERR."""
    print(f'WARNING: {msg}', file=sys.stderr)


class Converter(ABC):
    """Base class for converters.

    Parameters
    ----------
    path : Path
        Path to root directory of corpus.
    """
    PHONE_MAP = {
        'h#' : 'sil',
        'pau' : 'sil',
        'epi' : 'sil'}
    def __init__(self, path):
        self.path = path

    def convert_audio_files(self, dest_dir, n_jobs=1):
        """Convert audio files to 16 kHz, monochannel FLAC under
        ``dest_dir``.

        Parameters
        ----------
        dest_dir : Path
            Path to destination directory.

        n_jobs : int, optional
            Number of parallel jobs to use.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        pool = multiprocessing.Pool(n_jobs)
        def args_gen():
            for src_path in sorted(self.audio_paths):
                speaker_uri = src_path.parts[-2].upper()
                sent_uri = src_path.stem.upper()
                dest_flac_path = dest_dir / f'{speaker_uri}_{sent_uri}.flac'
                yield src_path, dest_flac_path
        pool.starmap(self.convert_audio_file, args_gen())

    @classmethod
    @abstractmethod
    def convert_audio_file(cls, src_audio_path, dest_flac_path):
        """Convert audio file at ``src_audio_path`` to 16 kHz, monochannel
        FLAC file at ``dest_flac_path``.
        """
        pass

    def convert_phones_files(self, dest_dir, flac_dir=None, n_jobs=1):
        """Convert phones files to HTK label files under ``dest_dir``.

        Parameters
        ----------
        dest_dir : Path
            Path to destination directory.

        flac_dir : Path, optional
            Path to directory containing corresponding FLAC files. Used to
            set offsets of final phones.
            (Default: None)

        n_jobs : int, optional
            Number of parallel jobs to use.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        pool = multiprocessing.Pool(n_jobs)
        def args_gen():
            for	phones_path in sorted(self.phones_paths):
                speaker_uri = phones_path.parts[-2].upper()
                sent_uri = phones_path.stem.upper()
                lab_path = dest_dir / f'{speaker_uri}_{sent_uri}.lab'
                yield phones_path, lab_path, flac_dir
        pool.starmap(self.convert_phones_file, args_gen())

    def convert_phones_file(self, phones_path, lab_path, flac_dir=None):
        """Convert phones file at ``phones_path`` to HTK label file at
        ``lab_path``.
        """
        # Load phone segments.
        with open(phones_path, 'r') as f:
            segs = []
            for line in f:
                onset, offset, label = line.strip().split()
                onset = float(onset) / self.phone_sr
                offset = float(offset) / self.phone_sr
                label = self.PHONE_MAP.get(label, label)
                segs.append([onset, offset, label])

        # Correct offset of final segment to recording duration.
        if flac_dir is not None:
            flac_path = Path(flac_dir, lab_path.stem + '.flac')
            segs[-1][1] = librosa.get_duration(filename=flac_path)

        # Write to HTK label file.
        with open(lab_path, 'w') as f:
            for onset, offset, label in segs:
                f.write(f'{onset:.3f}\t{offset:.3f}\t{label}\n')

    # TODO: Refactor to eliminate shared code in audio_paths/phones_paths.
    @property
    def audio_paths(self):
        """Paths to audio files."""
        paths = []
        for path in self.path.rglob(f'*{self.audio_ext}'):
            if path.parts[-4].upper() not in {'TRAIN', 'TEST'}:
                continue
            paths.append(path)
        return sorted(paths)

    @property
    def phones_paths(self):
        """Paths to phones files."""
        paths = []
        for path in self.path.rglob(f'*{self.phones_ext}'):
            if path.parts[-4].upper() not in {'TRAIN', 'TEST'}:
                continue
            paths.append(path)
        return sorted(paths)

    @property
    @abstractmethod
    def name(self):
        """Name of corpus."""
        pass

    @property
    @abstractmethod
    def phones_ext(self):
        """Extension of phones files."""
        pass

    @property
    @abstractmethod
    def phone_sr(self):
        """Sample rate used for onsets/offsets in phones files."""
        pass

    @property
    @abstractmethod
    def audio_ext(self):
        """Extension of phones files."""
        pass


class TIMITConverter(Converter):
    """Converter for TIMIT (LDC93S1)."""
    @classmethod
    def convert_audio_file(cls, src_audio_path, dest_flac_path):
        """Convert audio file at ``src_audio_path`` to 16 kHz, monochannel
        FLAC file at ``dest_flac_path``.
        """
        cmd = ['sox', str(src_audio_path), str(dest_flac_path)]
        subprocess.run(cmd, check=True)

    @property
    def name(self):
        return 'timit'

    @property
    def phones_ext(self):
        return 'PHN'

    @property
    def phone_sr(self):
        return 16000

    @property
    def audio_ext(self):
        return 'WAV'


class NTIMITConverter(Converter):
    """Converter for NTIMIT (LDC93S2)."""
    @classmethod
    def convert_audio_file(cls, src_audio_path, dest_flac_path):
        """Convert audio file at ``src_audio_path`` to 16 kHz, monochannel
        FLAC file at ``dest_flac_path``.
        """
        cmd = ['sox', str(src_audio_path), str(dest_flac_path)]
        subprocess.run(cmd, check=True)

    @property
    def name(self):
        return 'ntimit'

    @property
    def phones_ext(self):
        return 'phn'

    @property
    def phone_sr(self):
        return 16000

    @property
    def audio_ext(self):
        return 'flac'


class CTIMITConverter(Converter):
    """Converter for CTIMIT (LDC96S30)."""
    @classmethod
    def convert_audio_file(cls, src_audio_path, dest_flac_path):
        """Convert audio file at ``src_audio_path`` to 16 kHz, monochannel
        FLAC file at ``dest_flac_path``.
        """
        cmd = ['sox', str(src_audio_path), str(dest_flac_path),
               'rate', '16000']
        subprocess.run(cmd, check=True)

    @property
    def name(self):
        return 'ctimit'

    @property
    def phones_ext(self):
        return 'phn'

    @property
    def phone_sr(self):
        return 8000

    @property
    def audio_ext(self):
        return 'wav'


class WTIMITConverter(Converter):
    """Converter for WTIMIT (LDC2010S02)."""
    @classmethod
    def convert_audio_file(cls, src_audio_path, dest_flac_path):
        """Convert audio file at ``src_audio_path`` to 16 kHz, monochannel
        FLAC file at ``dest_flac_path``.
        """
        cmd = ['sox',
               '-r', '16000',
               '-e', 'signed',
               '-b', '16',
               '-c', '1',
               str(src_audio_path), str(dest_flac_path)]
        subprocess.run(cmd, check=True)

    @property
    def name(self):
        return 'wtimit'

    @property
    def phones_ext(self):
        return 'PHN'

    @property
    def phone_sr(self):
        return 16000

    @property
    def audio_ext(self):
        return 'raw'


class FFMTIMITConverter(Converter):
    """Converter for FFMTIMIT (LDC96S32)."""
    @classmethod
    def convert_audio_file(cls, src_audio_path, dest_flac_path):
        """Convert audio file at ``src_audio_path`` to 16 kHz, monochannel
        FLAC file at ``dest_flac_path``.
        """
        cmd = ['sox', str(src_audio_path), str(dest_flac_path)]
        subprocess.run(cmd, check=True)

    @property
    def name(self):
        return 'ffmtimit'

    @property
    def phones_ext(self):
        return 'phn'

    @property
    def phone_sr(self):
        return 16000

    @property
    def audio_ext(self):
        return 'wav'





class STCTIMITConverter(Converter):
    """Converter for STCTIMIT (LDC2008S03)."""
    def convert_audio_files(self, dest_dir, n_jobs=1):
        """Convert audio files to 16 kHz, monochannel FLAC under ``dest_dir``.

        Parameters
        ----------
        dest_dir : Path
            Path to destination directory.

        n_jobs : int, optional
            Number of parallel jobs to use.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        pool = multiprocessing.Pool(n_jobs)
        def args_gen():
            for src_path in sorted(self.audio_paths):
                dest_flac_path = Path(dest_dir, src_path.stem + '.flac')
                yield src_path, dest_flac_path
        pool.starmap(self.convert_audio_file, args_gen())

    @classmethod
    def convert_audio_file(cls, src_audio_path, dest_flac_path):
        """Convert audio file at ``src_audio_path`` to 16 kHz, monochannel
        FLAC file at ``dest_flac_path``.
        """
        cmd = ['sox', str(src_audio_path), str(dest_flac_path)]
        subprocess.run(cmd, check=True)

    @property
    def audio_paths(self):
        """Paths to audio files."""
        paths = []
        for path in self.path.rglob(f'*{self.audio_ext}'):
            if path.parts[-2].upper() not in {'TRAIN', 'TEST'}:
                continue
            paths.append(path)
        return sorted(paths)

    @property
    def name(self):
        return 'ffmtimit'

    @property
    def phones_ext(self):
        return 'phn'

    @property
    def phone_sr(self):
        return 16000

    @property
    def audio_ext(self):
        return 'wav'

# STCTIMIT is special.

CONVERTERS = {
#    'timit' : TIMITConverter,
#    'ntimit' : NTIMITConverter,
#    'ctimit' : CTIMITConverter,
#    'wtimit' : WTIMITConverter,
    'ffmtimit' : FFMTIMITConverter,
    'stctimit' : STCTIMITConverter,
}

def main():
    parser = argparse.ArgumentParser(
        description='prepare TIMIT variants for experiments', add_help=True)
    parser.add_argument(
        '--n-jobs', metavar='JOBS', default=1, type=int,
        help='run using JOBS parallel processes (default: %(default)s)')
    args = parser.parse_args()

    # Process the TIMIT variants into a common format.
    corpus_paths = load_corpus_paths()
    for corpus in  sorted(corpus_paths):
        print(f'Processing {corpus}...')
        corpus_dir = corpus_paths[corpus]
        if corpus not in CONVERTERS:
            continue
        converter = CONVERTERS[corpus](corpus_dir)

        # Convert audio to 16 kHz FLAC.
        flac_dir = Path(THIS_DIR, corpus, 'flac')
        converter.convert_audio_files(flac_dir, args.n_jobs)

        # Convert phones files to HTK label files.
        if corpus != 'stctimit':
            phones_dir =  Path(THIS_DIR, corpus, 'phones')
            converter.convert_phones_files(phones_dir, flac_dir, args.n_jobs)

    # STCTIMIT doest not include phones/words files. However, it is aligned
    # to TIMIT to within one sample, so reuse TIMIT label files.
    shutil.copytree(
        Path(THIS_DIR, 'timit', 'phones'),
        Path(THIS_DIR, 'stctimit', 'phones'))


if __name__ == '__main__':
    main()
