from fastai.vision.all import *
import numpy as np

vocab = ['fresh', 'rotten']

bananas = DataBlock(
    blocks = (ImageBlock, CategoryBlock(vocab=vocab)),
    get_y=lambda o: vocab[0],
    splitter=lambda o: ([0], [0]),
    item_tfms = [Resize(224, method="squish")],
    batch_tfms = Normalize.from_stats(*imagenet_stats),
)

dls = bananas.dataloaders([np.zeros((224, 224, 3), dtype=np.uint8)], bs=1)
learn = vision_learner(dls, resnet18, pretrained=False)

learn.load('is_banana_rotten', with_opt=False)

def predict(image_bytes):
    image = PILImage.create(image_bytes)

    pred, idx, prob = learn.predict(image)

    return pred, round(float(prob[idx]), 4)


