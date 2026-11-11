import numpy as np


def softmax(logits):
    exp = np.exp(logits[0][0])
    exp_sum = np.sum(exp)
    probs = exp / exp_sum
    max_idx = np.argmax(probs)

    return probs, max_idx
