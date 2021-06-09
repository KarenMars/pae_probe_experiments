#!/usr/bin/env python
"""TODO"""
import argparse
from pathlib import Path
import sys

import pandas as pd
from plotnine import (aes, geom_line, ggplot, ggtitle, labs,
                      scale_color_brewer, xlab, ylab)


FEATS = {'decoar' :'DeCoAR',
         'mel_fbank' : 'Mel fbank',
         'mfcc' : 'MFCC',
         'mockingjay' : 'MJ',
         'wav2vec-large' : 'wav2vec LARGE',
         'vq-wav2vec_kmeans_roberta' : 'vq-wav2vec + RoBERTa',
         'wav2vec2.0_960h' : 'wav2vec 2.0 LARGE (LS-960)',
         'wav2vec2.0_FT' : 'wav2vec 2.0 LARGE (LV-60k + FT)',
         'wav2vec2.0_vox' : 'wav2vec 2.0 LARGE (LV-60k)'}

W2V_FEATS = {'wav2vec 2.0 LARGE (LS-960))',
             'wav2vec 2.0 LARGE (LV-60k + FT)',
             'wav2vec 2.0 LARGE (LV-60k)'}


def plot_explained_variance_by_timit_variant(fig_dir, df):
    """Plot cumulative explained variance of components.

    Cumulative explained variance is plotted separately for each TIMIT variant.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(exist_ok=True, parents=True)
    for corpus_name, grp in df.groupby('corpus_name'):
        plt = (ggplot(grp, aes('component', 'cum_explained_variance_ratio',
                               color='feat_name')) +
               geom_line() + xlab('Component') + labs() +
               ylab('Cumulative explained variance') +
               scale_color_brewer(type='qual', palette='Set1') +
               labs(color='Feature') +
               ggtitle(f'Cumulative explained variance on {corpus_name.upper()}'))
        plt.save(
            Path(fig_dir, f'{corpus_name}.png'), verbose=False, width=15, height=8)


def plot_explained_variance_by_feature(fig_dir, df):
    """Plot cumulative explained variance of components.

    Cumulative explained variance is plotted separately for each feature.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(exist_ok=True, parents=True)
    for feat_name, grp in df.groupby('feat_name'):
        plt = (ggplot(grp, aes('component', 'cum_explained_variance_ratio',
                               color='corpus_name')) +
               geom_line() + xlab('Component') + labs() +
               ylab('Cumulative explained variance') +
               scale_color_brewer(type='qual', palette='Set1') +
               ggtitle(f'Cumulative explained variance of {feat_name} by corpus)'))
        feat_name = feat_name.lower().replace(' ', '_')
        png_path = Path(fig_dir, f'{feat_name}.png')
        plt.save(png_path, verbose=False, width=15, height=8)


def main():
    parser = argparse.ArgumentParser(
        description='plot spectrum of embeddings by class', add_help=True)
    parser.add_argument(
        'fig_dir', metavar='fig-dir', type=Path,
        help='path to output directory for figures')
    parser.add_argument(
        'spectra', type=Path, help='path to tab-delimited file with spectra')
    parser.add_argument(
        '--exclude-calibration', default=False, action='store_true',
        help='exclude calibration sentences')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    # Load precomputed spectra.
    df = pd.read_csv(args.spectra, header=0, sep='\t')
    df = df[df.exclude_calibration == args.exclude_calibration]
    df.feat_name = df.feat_name.replace(FEATS)

    # Plot explained variance by feature and variant.
    plot_explained_variance_by_timit_variant(
        Path(args.fig_dir, 'by_corpus'), df)
    plot_explained_variance_by_feature(
        Path(args.fig_dir, 'by_feat'), df)
    w2v_df = df[df.feat_name.isin(W2V_FEATS)]
    plot_explained_variance_by_timit_variant(
        Path(args.fig_dir, 'wav2vec'), w2v_df)


if __name__ == '__main__':
    main()
