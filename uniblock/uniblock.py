#########################################################################
# ⨼(・ω・)⨽ ⨼(・ω・)⨽ ⨼(・ω・)⨽ 按住 按住 ⨼(・ω・)⨽ ⨼(・ω・)⨽ ⨼(・ω・)⨽ #
#########################################################################


import os
import sys
import argparse

import numpy as np
from tqdm import tqdm
from joblib import dump, load
from sklearn.mixture import BayesianGaussianMixture


#########################################################################
# ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ 波浪 波浪 ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ #
#########################################################################


def get_num_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as fi:
        for num_lines, _ in enumerate(fi, 1):
            pass
    return num_lines


#########################################################################
# ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ 波浪 波浪 ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ #
#########################################################################


class Block:

    def __init__(self, block_str, pseudo=False):
        self.ranges = []
        for range_str in block_str.split(';')[0].split():
            l, r = range_str.split('..')
            self.ranges.append(
                range(int(l, 16), int(r, 16) + 1)
            )
        self.pseudo = pseudo

    def has(self, char):
        for r in self.ranges:
            if ord(char) in r:
                return True
        return False


#########################################################################
# ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ 波浪 波浪 ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ #
#########################################################################


class Feature:

    def __init__(self, char_count, word_count, normalize):
        self.blocks = []  # pseudo blocks are before Unicode blocks
        self.char_count = char_count
        self.word_count = word_count
        self.normalize = normalize

    def cache_unicode2block(self):
        # get pseudo and unicode block indices
        self.pseudo_block_indices = []
        self.unicode_block_indices = []
        for j, block in enumerate(self.blocks):
            if block.pseudo:
                self.pseudo_block_indices.append(j)
            else:
                self.unicode_block_indices.append(j)
        # cache u2b, unicode first, then overwrite with pseudo
        self.u2b = {}
        for j in self.unicode_block_indices:
            block = self.blocks[j]
            for i in block.ranges[0]:
                self.u2b[i] = j
        for j in self.pseudo_block_indices:
            block = self.blocks[j]
            for sub_range in block.ranges:
                for i in sub_range:
                    self.u2b[i] = j

    def _character2block(self, character):
        return self.u2b[ord(character)]

    def add(self, block):
        self.blocks.append(block)

    def sentence_to_vector(self, sentence):
        vector = np.zeros(len(self.blocks))
        if len(vector) > 0:
            for char in sentence:
                try:
                    j = self._character2block(char)
                    vector[j] += 1
                except KeyError:
                    if len(self.unicode_block_indices) == 0:
                        continue
                    else:
                        print(self.unicode_block_indices)
                        msg = f'no block defined for character {char}'
                        raise RuntimeError(msg)
            if self.normalize:
                if not len(sentence) == 0:
                    vector /= len(sentence)
        if self.char_count:
            vector = np.append(vector, len(sentence))
        if self.word_count:
            vector = np.append(vector, len(sentence.split()))
        return vector

    def __len__(self):
        return len(self.blocks) + self.char_count + self.word_count

    def get_feature_matrix(self, corpus_path):
        num_lines = get_num_lines(corpus_path)
        X = np.zeros((num_lines, len(self)))
        with open(corpus_path, 'r', encoding='utf-8') as fi:
            for i, line in tqdm(enumerate(fi)):
                X[i] = self.sentence_to_vector(line.strip())
        return X


#########################################################################
# ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ 波浪 波浪 ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ #
#########################################################################


class Command:

    def __init__(self, argv):
        self.argv = argv
        self.args = self._parse_args(argv)

    def _parse_args(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def _log_scores(self, scores, corpus_path):
        uniblock_path = self._get_uniblock_path()
        corpus_name = os.path.basename(corpus_path)
        score_path = os.path.join(uniblock_path, corpus_name + '.score')
        with open(corpus_path, 'r', encoding='utf-8') as fi, \
                open(score_path, 'w') as fo:
            for i, line in enumerate(fi):
                fo.write(f'{scores[i]:8.4f} : {line.strip()}\n')

    def _log_stats(self, scores, corpus_path):
        uniblock_path = self._get_uniblock_path()
        corpus_name = os.path.basename(corpus_path)
        stat_path = os.path.join(uniblock_path, corpus_name + '.stat')
        with open(stat_path, 'w') as fo:
            fo.write(f'{np.min(scores):8.4f} : <min>\n')
            fo.write(f'{np.max(scores):8.4f} : <max>\n')
            fo.write(f'{np.mean(scores):8.4f} : <mean>\n')

    def _get_uniblock_path(self):
        return os.path.join(self.args.exp_path, 'uniblock')


#########################################################################
# ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ 波浪 波浪 ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ #
#########################################################################


class Setup(Command):

    def _parse_args(self, argv):
        parser = argparse.ArgumentParser(
            description='setup experiment path and feature definitions',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            '--exp-path',
            type=str,
            default='./',
            help=(
                'uniblock-related files will be stored under '
                'exp-path/uniblock'
            )
        )
        parser.add_argument(
            '--char-count',
            action='store_true',
            help='add character counts to the feature vectors',
        )
        parser.add_argument(
            '--word-count',
            action='store_true',
            help='add word counts to the feature vectors',
        )
        parser.add_argument(
            '--no-unicode-blocks',
            action='store_true',
            help='disable unicode block features',
        )
        parser.add_argument(
            '--no-normalize',
            action='store_true',
            help='do not normalize the block counts',
        )
        parser.add_argument(
            '--pseudo-blocks',
            nargs='*',
            help=(
                'special hexadecimal Unicode ranges to descriminate '
                'and treat as \'pseudo blocks\', '
                'e.g. \'0030..0039; ASCII digits\' and '
                '\'0020..002F 003A..0040 005B..0060 007B..007E; '
                'ASCII punctuation and symbols\' (semicolon between '
                'range and description)'
            ),
        )
        args = parser.parse_args(argv)
        return args

    def run(self):
        args = self.args
        uniblock_path = self._get_uniblock_path()
        os.mkdir(uniblock_path)
        feature = Feature(
            args.char_count,
            args.word_count,
            (not args.no_normalize),
        )
        # add pseudo blocks, must be added before Unicode blocks
        if args.pseudo_blocks is not None:
            for pseudo_block in args.pseudo_blocks:
                b = Block(pseudo_block)
                feature.add(b)
        # add Unicode blocks, must be added after pseudo blocks
        if not args.no_unicode_blocks:
            UnicodeBlocks_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'Blocks.txt',
            )
            with open(UnicodeBlocks_path, 'r') as fi:
                for line in fi:
                    if (not line.startswith('#')) and \
                            ('..' in line) and ('; ' in line):
                        b = Block(line)
                        feature.add(b)
        # cache mapping and dump feature
        feature.cache_unicode2block()
        dump(feature, os.path.join(uniblock_path, 'feature.dump'))


#########################################################################
# ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ 波浪 波浪 ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ #
#########################################################################


class Train(Command):

    def _parse_args(self, argv):
        parser = argparse.ArgumentParser(
            description='train an uniblock model',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            'corpus_path',
            type=str,
            help=(
                'path to the clean data, '
                'this is usually the development set'
            )
        )
        parser.add_argument(
            '--exp-path',
            type=str,
            default='./',
            help='where uniblock-related files are stored',
        )
        parser.add_argument(
            '--k',
            type=int,
            default=20,
            help='number of mixture components',
        )
        parser.add_argument(
            '--cov',
            type=str,
            default='full',
            help='covariance type, {full|tied|diag|spherical}',
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='print model training information',
        )
        parser.add_argument(
            '--n-init',
            type=int,
            default=1,
            help='number of initializations',
        )
        parser.add_argument(
            '--init-params',
            type=str,
            default='kmeans',
            help='type of initialization, {kmeans|random}',
        )
        parser.add_argument(
            '--tol',
            type=float,
            default=1e-2,
            help='tolerance, convergence threshold',
        )
        args = parser.parse_args(argv)
        return args

    def run(self):
        args = self.args
        uniblock_path = self._get_uniblock_path()
        feature = load(os.path.join(uniblock_path, 'feature.dump'))
        X = feature.get_feature_matrix(args.corpus_path)
        legal, mask = self._infer_nonzero(X)
        dump(legal, os.path.join(uniblock_path, 'legal.dump'))
        dump(mask, os.path.join(uniblock_path, 'mask.dump'))
        X = X[:, legal]
        bgm = BayesianGaussianMixture(
            n_components=args.k,
            covariance_type=args.cov,
            max_iter=200,
            random_state=0,
            verbose=0 if not args.verbose else 2,
            verbose_interval=1,
            tol=args.tol,
            n_init=args.n_init,
            init_params=args.init_params,
        )
        bgm.fit(X)
        dump(bgm, os.path.join(uniblock_path, 'bgm.dump'))
        scores = bgm.score_samples(X)
        self._log_scores(scores, args.corpus_path)
        self._log_stats(scores, args.corpus_path)

    @staticmethod
    def _infer_nonzero(X):
        _, observed_blocks = np.nonzero(X)
        legal = np.unique(observed_blocks)
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[legal] = 1
        return legal, mask


#########################################################################
# ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ 波浪 波浪 ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ #
#########################################################################


class Score(Command):

    def _parse_args(self, argv):
        parser = argparse.ArgumentParser(
            description='score corpus with a trained uniblock model',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            'corpus_path',
            type=str,
            help=(
                'path to noisy data, '
                'this is usually the training set'
            )
        )
        parser.add_argument(
            '--exp-path',
            type=str,
            default='./',
            help='where uniblock-related files are stored',
        )
        args = parser.parse_args(argv)
        return args

    def run(self):
        args = self.args
        uniblock_path = self._get_uniblock_path()
        feature = load(os.path.join(uniblock_path, 'feature.dump'))
        X = feature.get_feature_matrix(args.corpus_path)
        legal = load(os.path.join(uniblock_path, 'legal.dump'))
        mask = load(os.path.join(uniblock_path, 'mask.dump'))
        legalness = np.array([self._get_legalness(v, mask) for v in X])
        X = X[:, legal]
        bgm = load(os.path.join(uniblock_path, 'bgm.dump'))
        scores = bgm.score_samples(X)
        scores *= legalness
        self._log_scores(scores, args.corpus_path)
        self._log_stats(scores, args.corpus_path)

    @staticmethod
    def _get_legalness(vector, mask):
        if all(np.logical_or(vector, mask).astype(bool) == mask):
            return 1
        else:
            return 0


#########################################################################
# ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ 波浪 波浪 ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ #
#########################################################################


class Filter(Command):

    def _parse_args(self, argv):
        parser = argparse.ArgumentParser(
            description='filter lines with uniblock scores',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            'score_paths',
            type=str,
            nargs='+',
            help=(
                'paths to score files to remove lines from, '
                'order-sensitive, '
                'parallel score files supported, '
                '.filter files are written in the same directory '
                'as the corresponding score_paths'
            )
        )
        parser.add_argument(
            '--thres-abs',
            type=float,
            help=(
                'absolute threshold value, lines with scores '
                'lower than or equal to this value will be removed'
            ),
        )
        parser.add_argument(
            '--thres-rel',
            type=float,
            help=(
                'relative threshold of amount of data to remove, '
                'in range [0, 1]'
            ),
        )
        parser.add_argument(
            '--combine',
            type=str,
            default='min',
            choices=['min', 'max', 'avg', 'weighted_sum', 'none'],
            help='method to combine scores across parallel files',
        )
        parser.add_argument(
            '--lambdas',
            type=float,
            nargs='*',
            help=(
                'when --combine is set to weighted_sum, '
                'need lambdas for each file '
                '(for Machine Translation one lambda for source '
                'and one lambda for target), '
                'order-sensitive'
            ),
        )
        parser.add_argument(
            '--stat-paths',
            nargs='*',
            help=(
                'when --combine is set to none, training stat files '
                'are required, parallel sentences are removed when '
                'one of them has a score lower than the lowest score '
                'of the corresponding training corpus, order-sensitive'
            ),
        )
        args = parser.parse_args(argv)
        return args

    def run(self):
        args = self.args
        num_lines = self._check_num_lines()
        self.num_lines = num_lines
        if args.combine == 'none':
            assert args.stat_paths is not None, \
                'missing --stat-paths'
            scores = self._get_scores(num_lines)
            thresholds = self._get_thresholds()
            to_filter = np.any(scores < thresholds, axis=1)
        else:
            scores = self._get_scores(num_lines)
            if args.combine == 'weighted_sum':
                assert args.lambdas is not None, \
                    'missing --lambdas'
                assert len(args.score_paths) == len(args.lambdas), \
                    'unmatched score_paths and --lambdas'
                weights = np.array(args.lambdas)
                scores = np.sum(scores * weights, axis=1)
            else:
                combine = {
                    'min': np.min,
                    'max': np.max,
                    'avg': np.mean,
                }[args.combine]
                scores = combine(scores, axis=1)
            filter_method = self._get_filter_method()
            to_filter = filter_method(scores)
        self._filter(to_filter)

    def _filter(self, to_filter):
        for score_path in self.args.score_paths:
            score_dir = os.path.dirname(score_path)
            score_name = os.path.basename(score_path)
            filter_path = os.path.join(
                score_dir,
                score_name + '.filter'
            )
            with open(filter_path, 'w') as fo, \
                    open(score_path, 'r', encoding='utf-8') as fi:
                for i, line in enumerate(fi):
                    if to_filter[i]:
                        pass
                    else:
                        fo.write(self._strip_score(line))
        print(
            f'@ {sum(to_filter):.0f} / {self.num_lines} '
            f'≈ {sum(to_filter)/self.num_lines:3.2%} '
            f'lines filtered out'
        )

    @staticmethod
    def _strip_score(line):
        return line[11:]

    def _check_num_lines(self):
        args = self.args
        num_lines = get_num_lines(args.score_paths[0])
        for score_path in args.score_paths[1:]:
            assert (get_num_lines(score_path) == num_lines), \
                'non-parallel data in score_paths'
        return num_lines

    def _get_thresholds(self):
        score_paths = self.args.score_paths
        stat_paths = self.args.stat_paths
        assert (len(score_paths) == len(stat_paths)), \
            'unmatched score_paths and --stat-paths'
        thresholds = np.zeros(len(score_paths))
        for i, stat_path in enumerate(stat_paths):
            with open(stat_path, 'r', encoding='utf-8') as fi:
                line = fi.readline()  # min was written to the first line
                thresholds[i] = float(line.split(':')[0].strip())
        return thresholds

    def _get_scores(self, num_lines):
        score_paths = self.args.score_paths
        scores = np.zeros((num_lines, len(score_paths)))
        for j, score_path in enumerate(score_paths):
            with open(score_path, 'r', encoding='utf-8') as fi:
                for i, line in enumerate(fi):
                    score = float(line.split(':')[0].strip())
                    scores[i, j] = score
        return scores

    def _get_filter_method(self):
        abs_set = self.args.thres_abs is not None
        rel_set = self.args.thres_rel is not None
        assert \
            ((abs_set and not rel_set) or (not abs_set and rel_set)), \
            'ambiguous threshold, need either --thres-abs or --thres-rel'
        if abs_set:
            return self._absolute
        if rel_set:
            return self._relative

    def _absolute(self, scores):
        threshold = self.args.thres_abs
        to_filter = scores < threshold
        return to_filter

    def _relative(self, scores):
        threshold = self.args.thres_rel
        threshold = self.args.thres_rel
        cut_index = int(np.floor(threshold * len(scores)))
        to_filter = np.zeros(len(scores))
        to_filter[np.argsort(scores)[:cut_index]] = 1
        return to_filter


#########################################################################
# ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ 波浪 波浪 ᚛(・ω・)᚜ ᚛(・ω・)᚜ ᚛(・ω・)᚜ #
#########################################################################


def main():
    parser = argparse.ArgumentParser(
        description=(
            'uniblock, scoring and filtering corpus '
            'with Unicode block information (and more).'
        ),
        usage='uniblock.py <command> [<args>]',
    )
    parser.add_argument(
        'command',
        type=str,
        choices=['setup', 'train', 'score', 'filter'],
        help='command to execute',
    )
    args = parser.parse_args(sys.argv[1:2])
    command_class = {
        'setup': Setup,
        'train': Train,
        'score': Score,
        'filter': Filter,

    }[args.command]
    command = command_class(sys.argv[2:])
    command.run()


#########################################################################
# ⌝(・ω・)⌜ ⌝(・ω・)⌜ ⌝(・ω・)⌜ 托住 托住 ⌝(・ω・)⌜ ⌝(・ω・)⌜ ⌝(・ω・)⌜ #
#########################################################################
