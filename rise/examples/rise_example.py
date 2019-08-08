import os
import sys

import numpy as np
from keras import backend as K
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from matplotlib import pyplot as plt

sys.path.append("../../")
from rise.rise import RiseImageExplainer

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

K.set_learning_phase(0)

model = ResNet50()
input_size = (224, 224)


def load_img(path):
    _img = image.load_img(path, target_size=input_size)
    img_arr = image.img_to_array(_img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    return _img, img_arr


def class_name(idx):
    return decode_predictions(np.eye(1, 1000, idx))[0][0][1]


explainer = RiseImageExplainer()
explainer.fit(model.predict,
              input_size,
              500,
              8,
              0.5)

img, x = load_img(os.path.join(FILE_PATH, 'bullcar.jpg'))
sal = explainer.explain(x)

class_idxs = model.predict(x).argsort()[0][::-1]
fig, axes = plt.subplots(1, 5)
fig.set_size_inches(15, 6)
for i in range(5):
    class_idx = class_idxs[i]
    axes[i].set_title('Explanation for `{}`'.format(class_name(class_idx)))
    axes[i].axis('off')
    axes[i].imshow(img)
    axes[i].imshow(sal[class_idx], cmap='jet', alpha=0.5)

plt.show()
