import os
import sys
import warnings

from imgcat import imgcat

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from keras import backend as K
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras.preprocessing import image
from matplotlib import pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{SCRIPT_DIR}/..")

import rise

K.set_learning_phase(0)

model = MobileNetV2()
input_size = (224, 224)


def load_img(path):
    img = image.load_img(path, target_size=input_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


def class_name(idx):
    return decode_predictions(np.eye(1, 1000, idx))[0][0][1]


img, x = load_img(f'{SCRIPT_DIR}/bullcar_crop.png')
imgcat(img)

explainer = rise.RiseImageExplainer()
explainer.fit(model.predict,
              input_size,
              number_of_masks=1000,
              mask_granularity=8,
              mask_density=0.5,
              channels_last=True)

sal = explainer.explain(x)

probas = model.predict(x)
top_classes_idx = probas.argsort()[0][::-1]
top_probas = probas[0][top_classes_idx]

fig, axes = plt.subplots(1, 5)
fig.set_size_inches(15, 6)
for i in range(5):
    axes[i].set_title(f'Cls: "{class_name(top_classes_idx[i])}" {top_probas[i] * 100: .1f}%')
    axes[i].axis('off')
    axes[i].imshow(img)
    axes[i].imshow(sal[top_classes_idx[i]], cmap='jet', alpha=0.4)
plt.tight_layout()
imgcat(fig)
