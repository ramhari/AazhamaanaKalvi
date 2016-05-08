import numpy as np
import pandas as pd


class DataParser:
    def __init__(self):
        pass

    def get_next_batch(self, batch_size):
        # Input
        # batch_size: size of the training batch used

        # Output
        # features: [batch_size, feature_dim] nd-array
        # labels : [batch_size, num_classes] nd-array; Note: since we are using softmax loss function,
        # we need a 1-hot 2-d label vector even in the simple 2-class case
        features = np.random.randint(0, 256, [batch_size, 32, 32, 1])
        labels = np.random.randint(0, 2, [batch_size, 1])
        labels = np.hstack([labels, 1-labels])
        return features, labels
