from model1 import Model1


class Model2(Model1):
    def build_feature_index(self):
        super().build_feature_index()

    def make_features(self, history, tag):
        features = super().make_features(history, tag)
        return features
