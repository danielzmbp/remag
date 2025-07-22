"""4CAC bacterial sequence classifier.

This code is adapted from the 4CAC classifier:
https://github.com/Shamir-Lab/4CAC
"""

import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from joblib import load
from loguru import logger

import multiprocessing as mp
from multiprocessing import Manager

from . import xgb_utils as utils


class xgbClass:
    def __init__(
        self, n_procs=1, scales=[500, 1000, 5000, 10000, 50000], ks=[3, 4, 5, 6, 7]
    ):
        self._scales = scales
        self._ks = ks
        self._compute_kmer_inds()
        self._load_classifiers()
        self._n_procs = n_procs

    def classify(self, seq):
        """Classify the sequence(s), return the probability of the sequence(s) being a plasmid.
        Assumes seq is either an individual string or a list of strings
        Returns either an individual plasmid probability for seq or a list of
        plasmid probabilities for each sequence in seq
        """
        if isinstance(seq, str):  # single sequence
            logger.debug("Counting k-mers for sequence of length {}".format(len(seq)))
            kmer_freqs = [0]
            scale = self._get_scale(len(seq))
            utils.count_kmers(
                [0, seq, self._ks, self._kmer_inds, self._kmer_count_lens, kmer_freqs]
            )
            kmer_freqs = np.array(kmer_freqs)
            standardized_freqs = self._standardize(kmer_freqs, scale)
            logger.debug("Classifying")
            return self.classifiers[scale]["clf"].predict_proba(standardized_freqs)

        elif isinstance(seq, list):  # list of sequences
            logger.info(
                "{} sequences to classify. Classifying in batches of 100k".format(
                    len(seq)
                )
            )
            results = []
            seq_ind = 0
            pool = mp.Pool(self._n_procs)

            while seq_ind < len(seq):
                logger.debug("Starting new batch")
                seq_batch = seq[seq_ind : seq_ind + 100000]
                scales = [self._get_scale(len(s)) for s in seq_batch]
                scale_partitions = {
                    s: [seq_batch[i] for i, v in enumerate(scales) if v == s]
                    for s in self._scales
                }

                partitioned_classifications = {}
                for scale in self._scales:
                    part_seqs = scale_partitions[scale]
                    if len(part_seqs) <= 0:
                        continue
                    logger.debug(
                        "Getting kmer frequencies for partition length {}".format(scale)
                    )
                    shared_list = Manager().list()
                    for cur in np.arange(len(part_seqs)):
                        shared_list.append(0)
                    pool.map(
                        utils.count_kmers,
                        [
                            [
                                ind,
                                s,
                                self._ks,
                                self._kmer_inds,
                                self._kmer_count_lens,
                                shared_list,
                            ]
                            for ind, s in enumerate(part_seqs)
                        ],
                    )
                    kmer_freqs_mat = np.array(shared_list)
                    standardized_freqs = self._standardize(kmer_freqs_mat, scale)
                    logger.debug("Classifying sequences of length scale {}".format(scale))

                    partitioned_classifications[scale] = self.classifiers[scale][
                        "clf"
                    ].predict_proba(standardized_freqs)

                # recollate the results:
                scale_inds = {s: 0 for s in self._scales}
                for s in scales:
                    results.append(partitioned_classifications[s][scale_inds[s]])
                    scale_inds[s] += 1

                seq_ind += 100000

            pool.close()
            return np.array(results)

        else:
            raise TypeError("Can only classify strings or lists of strings")

    def _load_classifiers(self):
        """Load the multi-scale classifiers and scalers"""
        curr_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(curr_path, "models")
        self.classifiers = {}
        for i in self._scales:
            logger.debug("Loading classifier " + str(i))
            self.classifiers[i] = {
                "clf": XGBClassifier(),
                "scaler": load(os.path.join(data_path, "s" + str(i))),
            }
            self.classifiers[i]["clf"].load_model(
                os.path.join(data_path, "xgb" + str(i) + ".json")
            )

    def _get_scale(self, length):
        """Choose which length scale to use for the sequence"""
        if length <= self._scales[0]:
            return self._scales[0]
        for i, l in enumerate(self._scales[:-1]):
            if length <= float(l + self._scales[i + 1]) / 2.0:
                return l
        return self._scales[-1]

    def _standardize(self, freqs, scale):
        """Use sklearn's standard scaler to standardize
        Choose the appropriate scaler based on sequence length
        """
        return self.classifiers[scale]["scaler"].transform(freqs)

    def _compute_kmer_inds(self):
        """Get the indeces of each canonical kmer in the kmer count vectors"""
        self._kmer_inds, self._kmer_count_lens = utils.compute_kmer_inds(self._ks)
