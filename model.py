import onnxruntime as ort
from to_tensor import image_to_tensor
from softmax import softmax

vocab = ["fresh", "other", "rotten"]

sess = ort.InferenceSession("models/is_banana_rotten.onnx")
inp = sess.get_inputs()[0]


def predict(image_bytes):
    x_norm = image_to_tensor(image_bytes)
    logits = sess.run(None, {inp.name: x_norm})
    probs, max_idx = softmax(logits)

    pred = vocab[max_idx]
    prob = round(float(probs[max_idx]), 4)

    return pred, prob
