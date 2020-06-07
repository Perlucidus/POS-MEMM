from abc import abstractmethod
from typing import Any, Collection, Dict, Iterable, Optional, Sequence, Set, Tuple
from heapq import nlargest
import pickle
from pathlib import Path
from time import time
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import logsumexp, softmax

Word = str
Affix = str
Tag = str
Context = Sequence[Word]
ContextTagging = Sequence[Tag]
History = Tuple[Tag, Tag, Context, int]


class MaximumEntropyMarkovModel:
    def __init__(self, lambda_: float = 0, beam: Optional[int] = None):
        """
        :param lambda_: Regularization parameter
        :param beam: Beam heuristic value for prediction
        """
        self.word_vocabulary: Optional[Set[Word]] = None
        self.tag_vocabulary: Optional[Set[Tag]] = None
        self.prefixes: Optional[Set[Affix]] = None
        self.suffixes: Optional[Set[Affix]] = None
        self.index: Optional[Collection[Tuple[History, Tag]]] = None
        self.feature_index: Optional[Dict[Any, Dict[Any, int]]] = None
        self.weights: Optional[np.ndarray] = None
        self.feature_cache: Dict[Tuple[History, Tag], Collection[int]] = {}
        self.lambda_ = lambda_
        self.beam = beam

    def fit(self, contexts: Iterable[Context], taggings: Iterable[ContextTagging]):
        """
        Fits the model using tagged contexts
        :param contexts: Train contexts
        :param taggings: Train taggings
        """
        start = time()
        self.build_index(contexts, taggings)
        cache_start = time()
        print('Caching train features')
        for history, tag in self.index:
            self.get_features(history)
        print(f'Train features cache done in {time() - cache_start:.2f}s')
        weights = np.zeros(self.num_features)
        # Minimize the loss function to find the optimal weights
        self.weights, _, result = fmin_l_bfgs_b(func=self.objective, x0=weights, maxiter=500, iprint=10)
        warn_flag = result['warnflag']
        if warn_flag == 0:
            print(f'Converged in {result["nit"]} iterations with {result["funcalls"]} objective function calls')
        elif warn_flag == 1:
            print('Failed to converge, too many iterations')
        elif warn_flag == 2:
            print('Failed to converge (fatal)')
            print(result['task'])
        print(f'Training done in {time() - start:.2f}s')

    def predict(self, contexts: Collection[Context]) -> Iterable[ContextTagging]:
        """
        Predicts tags for every context using Viterbi algorithm
        :param contexts: The contexts for which the tags will be predicted
        :return: Tags for every context
        """
        print(f'Predicting {len(contexts)} contexts with beam={self.beam}')
        return (self.viterbi(context) for context in contexts)  # Iterator

    def objective(self, weights: np.ndarray) -> Tuple[float, float]:
        """
        Computes the loss and gradient of the weights
        :param weights: Weights for every feature
        :return: Loss and gradient
        """
        start = time()
        loss = 0
        gradient = np.zeros_like(weights)
        loss -= (self.lambda_ / 2) * sum(np.square(weights))  # Loss Regularization
        gradient -= self.lambda_ * weights  # Gradient Regularization
        for idx, (history, tag) in enumerate(self.index):
            all_features = np.asarray(self.get_features(history), dtype=int)
            score = np.sum(weights[all_features], axis=1)
            loss += sum(weights[self.get_features(history, tag)])  # Loss Linear Term
            loss -= logsumexp(score)  # Loss Normalization Term
            gradient[self.get_features(history, tag)] += 1  # Empirical Counts
            # Expected Counts
            # Execute softmax on features of every (history, tag) pairs and copy the result to the corresponding
            # gradient entries (result is tiled for every gradient entry)
            gradient[all_features] -= np.tile(softmax(score)[np.newaxis].transpose(), all_features.shape[-1])
        print(f'Objective computed in {time() - start:.2f}s')
        return -loss, -gradient

    def build_index(self, contexts: Iterable[Context], taggings: Iterable[ContextTagging]):
        """
        Builds a vocabulary of words and tags
        :param contexts: Train contexts
        :param taggings: Train taggings
        """
        start = time()
        self.word_vocabulary = set()
        self.tag_vocabulary = set()
        self.index = []
        self.prefixes = set()
        self.suffixes = set()
        for context, tags in zip(contexts, taggings):
            for idx, (word, tag) in enumerate(zip(context, tags)):
                self.word_vocabulary.add(word)
                self.tag_vocabulary.add(tag)
                history: History = (
                    tags[idx - 2] if idx >= 2 else '*',
                    tags[idx - 1] if idx >= 1 else '*',
                    tuple(context),
                    idx
                )
                self.index.append((history, tag))
                for l in range(4):
                    if len(word) > l:
                        idx = l + 1
                        self.prefixes.add(word[:idx])
                        self.suffixes.add(word[-idx:])
        self.build_feature_index()
        print(f'Indexing done in {time() - start:.2f}s')

    def get_features(self, history: History, tag: Optional[Tag] = None):
        """
        Get or make features for the given history and optionally tag
        :param history: History
        :param tag: Tag
        :return: Collection of feature indexes whose feature predicates are true
        """
        if tag is None:
            return [self.get_features(history, t) for t in self.tag_vocabulary]
        cached = self.feature_cache.get((history, tag), None)
        if cached is None:
            cached = self.feature_cache[(history, tag)] = self.make_features(history, tag)
        return cached

    @abstractmethod
    def build_feature_index(self):
        """
        Builds an index for all features
        """
        raise NotImplementedError()

    @property
    def num_features(self):
        """
        :return: Number of features for every history and tag
        """
        return sum(len(index) for index in self.feature_index.values())

    @abstractmethod
    def make_features(self, history: History, tag: Tag):
        """
        Makes features for the given history and tag
        :param history: History
        :param tag: Tag
        :return: Collection of feature indexes whose feature predicates are true
        """
        raise NotImplementedError()

    def viterbi(self, context: Context):
        """
        Viterbi algorithm for predicting tags
        :param context: Context
        :return: Predicted tags for the context
        """
        def available_tags(k):
            return self.tag_vocabulary if k >= 0 else ('*',)

        pi = {-1: {('*', '*'): 1}}
        bp = {}
        for idx, word in enumerate(context):
            pi[idx] = {}
            for tag1 in available_tags(idx - 1):
                q = {}  # Softmax cache for all tag, tag2 with this tag1
                for tag2 in available_tags(idx - 2):
                    if (tag2, tag1) not in pi[idx - 1]:
                        continue  # May not be there due to beam heuristic
                    history = (tag2, tag1, tuple(context), idx)
                    s = softmax(np.sum(self.weights[np.asarray(self.get_features(history))], axis=1))
                    for tag, p in zip(self.tag_vocabulary, s):
                        q[(tag, tag2)] = p
                for tag in available_tags(idx):
                    probabilities = {
                        tag2: pi[idx - 1][(tag2, tag1)] * q[(tag, tag2)]
                        for tag2 in available_tags(idx - 2)
                        if (tag2, tag1) in pi[idx - 1]  # May not be there due to beam heuristic
                    }
                    if len(probabilities) == 0:  # Could happen due to beam heuristic
                        continue
                    bp[idx, tag1, tag], pi[idx][tag1, tag] = max(probabilities.items(), key=lambda x: x[-1])
            if self.beam is not None:  # Keep only beam largest
                pi[idx] = dict(nlargest(self.beam, pi[idx].items(), key=lambda x: x[-1]))
        n = len(context)
        tags = list(max(
            {(tag2, tag1): pi[n - 1][(tag2, tag1)] for (tag2, tag1) in pi[n - 1]}.items(),
            key=lambda x: x[-1]
        )[0])
        while len(tags) < n:
            tags.insert(0, bp[n - 1 - (len(tags) - 2), tags[0], tags[1]])
        return tags

    def save_model(self, path: Path):
        """
        Saves the model (without feature cache) to IO storage
        :param path: Path to the saved model
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as m:
            cache = self.feature_cache
            self.feature_cache = {}
            pickle.dump(self, m)
            self.feature_cache = cache


def load_model(path: Path) -> MaximumEntropyMarkovModel:
    """
    Loads a model from IO storage
    :param path: Path to the saved model
    """
    with path.open('rb') as m:
        model: MaximumEntropyMarkovModel = pickle.load(m)
        return model


def preprocess(path: Path) -> Tuple[Collection[Context], Collection[ContextTagging]]:
    """
    Preprocess data file
    :param path: Path to data file
    :return: Contexts with their corresponding tags (if there are any)
    """
    with path.open('r') as data:
        lines = data.read().splitlines()
        contexts = []
        taggings = []
        for line in lines:
            context, tags = [], []
            for part in line.split(' '):
                if '_' not in part:
                    context.append(part)
                    continue
                word, tag = part.split('_')
                context.append(word)
                tags.append(tag)
            contexts.append(context)
            taggings.append(tags)
        return contexts, taggings
