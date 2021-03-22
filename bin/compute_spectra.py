#!/usr/bin/env python
"""Compute spectra of features.

For each combination of TIMIT variant and feature compute a truncated
randomized SVD using scikit-learn's ``TruncatedSVD`` module. For instance,

    python compute_spectra.py /path/to/feats spectra.tsv

where ``feats/`` is the directory containing the extracted features for the
TIMIT variants that are produced by the recipes under the ``models/`` directory.

By default, this script will randomly sample 250,000 frames from each dataset
to reduce the runtime and memory overhead, then compute the 200 more important
components for each subsampled dataset. As the space complexity of sklearn's
implementation is O(4.5mn + 3n^2), where m is the number of frames and n the
number of components being computed, you may need to adjust these defaults,
which can be done via the ``n-samples`` and ``n-components`` flags.

The output will be a tab delimited file containing one component per row, each
row having the following columns:

- corpus_name  --  name of TIMIT variant
- exclude_calibration  --  are calibration sentences (SA1/SA2) removed prior to
  SVD
- feat_name  --  name of feature
- component  --  index of component (1-indexed)
- singular_value  --  singular value
- explained_variance  --  variance explained by component
- explained_variance_ratio  --  proportion of total variance explained by
  component
- cum_explained_variance_ratio  --  proportion of total variance explained
  cumulatively by the first ``component`` components
"""
import argparse
from collections import namedtuple
from pathlib import Path
import random
import sys
import warnings

from joblib import Parallel, delayed
from matplotlib import MatplotlibDeprecationWarning
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from run_probe_exp import add_context


warnings.filterwarnings(action='ignore', category=MatplotlibDeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

def load_feats(feat_paths, context_width=0):
    """Load concatenated features files.

    Features are stored in NumPy ``.npy`` files are (``n_frames``, ``feat_dim``)
    arrays, where:

    - n_frames  --  the number of frams
    - feat_dim  --  the feature dimensionality

    Parameters
    ----------
    feat_paths : iterable of Path
        NumPy ``.npy`` files containing features.

    context_width : int, optional
        Number of frames of left/right context to use.

    Returns
    -------
    feats : (n_total_frames, feat_dim)
        Concatenated features.
    """
    feats = []
    for feat_path in feat_paths:
        X = np.load(feat_path)
        if context_width > 0:
            X = add_context(X, context_width)
        feats.append(X)
    feats = np.concatenate(feats).astype('float32')
    return feats


def compute_spectrum(feats, n_components=25):
    """Compute spectrum of features.

    The eigenvalues of the spectrum are computed using a randomized SVD that
    retained the ``n_components`` largest singular values.

    Parameters
    ----------
    feats : (n_total_frames, feat_dim)
        Features.

    n_components : int, optional
        Retain the ``n_components`` top singular values.

    Returns
    -------
    singular_values : (n_components,)
        Top ``n_component`` singular values of ``feats``, sorted in descending
        order by percent variance explained.

    explained_variance : (n_components,)
        Variance explained by each component.

    explained_variance_ratio : (n_components,)
        Percentage of total variance explaiend by each component.
    """
    n_components = min(n_components, feats.shape[1])
    model = TruncatedSVD(n_components).fit(feats)
    sort_inds = np.argsort(model.explained_variance_)[::-1]
    return (model.singular_values_[sort_inds], model.explained_variance_[sort_inds],
            model.explained_variance_ratio_[sort_inds])


ExtractionConfig = namedtuple(
    'ExtractionConfig', ['corpus_name', 'feat_name', 'feat_dir', 'context_width',
                         'n_samples', 'n_components', 'exclude_calibration'])

def generate_extraction_configs(feats_dir, n_samples, n_components):
    """Generate extraction configurations for each combination of corpus/feature."""
    feats_dir = Path(feats_dir)
    configs = []
    for path in feats_dir.glob('*/*/'):
        if not path.is_dir():
            continue
        corpus_name, feat_name = path.parts[-2:]
        corpus_name = corpus_name.replace('_final', '')
        context_width = 5 if feat_name in {'mfcc', 'mel_fbank'} else 0
        configs.append(ExtractionConfig(
            corpus_name, feat_name, path, context_width, n_samples,
            n_components, True))
        configs.append(ExtractionConfig(
            corpus_name, feat_name, path, context_width, n_samples,
            n_components, False))
    return configs


def process_one(config):
    """Extract spectrum for one combination of corpus/features."""
    # Load features.
    def _is_calibration_sent(path):
        return path.stem.split('_')[1].startswith('SA')
    feat_paths = sorted(config.feat_dir.glob('*.npy'))
    if config.exclude_calibration:
        feat_paths = [path for path in feat_paths if not _is_calibration_sent(path)]
    feats = load_feats(feat_paths, config.context_width)
    
    # Subsample.
    if 0 < config.n_samples < len(feats):
        inds = np.random.choice(len(feats), config.n_samples, replace=False)
        feats = feats[inds]

    # Compute SVD.
    singular_values, explained_variance, explained_variance_ratio = compute_spectrum(
        feats, config.n_components)

    # Populate dataframe.
    df = pd.DataFrame({
        'singular_value' : singular_values,
        'explained_variance' : explained_variance,
        'explained_variance_ratio' : explained_variance_ratio})
    df['cum_explained_variance_ratio'] = np.cumsum(
        df.explained_variance_ratio)
    df['corpus_name'] = config.corpus_name
    df['feat_name'] = config.feat_name
    df['component'] = np.arange(1, len(singular_values)+1)
    df['exclude_calibration'] = config.exclude_calibration
    df = df[['corpus_name', 'exclude_calibration', 'feat_name', 'component',
             'singular_value', 'explained_variance',
             'explained_variance_ratio', 'cum_explained_variance_ratio']]
    return df


def main():
    parser = argparse.ArgumentParser(
        description='plot spectrum of embeddings by class', add_help=True)
    parser.add_argument(
        'feat_dir', metavar='feat-dir', type=Path,
        help='path to root of features directories')
    parser.add_argument(
        'spectra', type=Path, help='path to output tab-delimited file with '
                                   'spectra')
    parser.add_argument(
        '--n-samples', metavar='SAMPLES', default=100000, type=int,
        help='subsample SAMPLES frames from each corpus (default: %(default)s)')
    parser.add_argument(
        '--n-components', metavar='N', default=50, type=int,
        help='retain N most important singular components (default: %(default)s)')
    parser.add_argument(
        '--seed', metavar='SEED', default=11112930, type=int,
        help='seed for RNG (default: %(default)s)')
    parser.add_argument(
        '--n-jobs', metavar='JOBS', default=1, type=int,
        help='run using JOBS parallel jobs (default: %(default)s)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Compute spectra.
    configs = generate_extraction_configs(
        args.feat_dir, args.n_samples, args.n_components)
    random.shuffle(configs)  # Improve performance by randomly distributing
                             # high memory cases.
    f =	delayed(process_one)
    res = Parallel(args.n_jobs)(f(c) for c in configs)
    df = pd.concat(res)
    
    # Save as tab-delimited file for visualization.
    df.to_csv(args.spectra, index=False, header=True, sep='\t')

if __name__ == '__main__':
    main()
