from memm import MaximumEntropyMarkovModel
from itertools import product


class Model1(MaximumEntropyMarkovModel):
    def build_feature_index(self):
        def add_feature(category, feature):
            self.feature_index[category][feature] = self.num_features

        self.feature_index = {
            'f100': {},
            'f101': {},
            'f102': {},
            'f103': {},
            'f104': {},
            'f105': {},
            'f106': {},
            'f107': {},
            'caps': {},
            'numeric': {},
            'hyphen': {},
        }
        for word, tag in product(self.word_vocabulary, self.tag_vocabulary):
            add_feature('f100', (word, tag))  # Ratnaparkhi f100
        for prefix, tag in product(self.prefixes, self.tag_vocabulary):
            add_feature('f101', (prefix, tag))  # Ratnaparkhi f101
        for suffix, tag in product(self.suffixes, self.tag_vocabulary):
            add_feature('f102', (suffix, tag))  # Ratnaparkhi f102
        for tag2, tag1, tag in product(self.tag_vocabulary, repeat=3):
            add_feature('f103', (tag2, tag1, tag))  # Ratnaparkhi f103
        for tag1, tag in product(self.tag_vocabulary, repeat=2):
            add_feature('f104', (tag1, tag))  # Ratnaparkhi f104
        for tag in self.tag_vocabulary:
            add_feature('f105', tag)  # Ratnaparkhi f105
        for word, tag in product(self.word_vocabulary, self.tag_vocabulary):
            add_feature('f106', (word, tag))  # Ratnaparkhi f106
            add_feature('f107', (word, tag))  # Ratnaparkhi f107
        for tag in self.tag_vocabulary:
            add_feature('caps', tag)  # Capital letters
            add_feature('numeric', tag)  # Numeric
            add_feature('hyphen', tag)  # Contains hyphen

    def make_features(self, history, tag):
        def get_feature_index(category, feature):
            return self.feature_index[category].get(feature, None)

        tag2, tag1, context, idx = history
        word = context[idx]
        features = set()
        features.add(get_feature_index('f100', (word, tag)))
        for i in range(4):
            affix_length = i + 1
            features.add(get_feature_index('f101', (word[:affix_length], tag)))
            features.add(get_feature_index('f102', (word[-affix_length:], tag)))
        features.add(get_feature_index('f103', (tag2, tag1, tag)))
        features.add(get_feature_index('f104', (tag1, tag)))
        features.add(get_feature_index('f105', tag))
        if idx > 0:
            features.add(get_feature_index('f106', (context[idx - 1], tag)))
        if idx < len(context) - 1:
            features.add(get_feature_index('f107', (context[idx + 1], tag)))
        if any(c.isupper() for c in word):
            features.add(get_feature_index('caps', tag))
        if any(c.isdigit() for c in word):
            features.add(get_feature_index('numeric', tag))
        if '-' in word:
            features.add(get_feature_index('hyphen', tag))
        features.discard(None)
        return list(features)
