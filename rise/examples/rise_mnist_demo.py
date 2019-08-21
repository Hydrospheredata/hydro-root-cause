import os
import sys
import warnings

from imgcat import imgcat

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from keras import backend as K
from matplotlib import pyplot as plt
from keras.models import load_model
import mnist

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{SCRIPT_DIR}/..")

import rise

K.set_learning_phase(0)

images = mnist.test_images()
model = load_model(f"{SCRIPT_DIR}/../../hs_demos/mnist/model.h5", compile=False)
input_size = (28, 28)
#
img = images[np.random.randint(0, images.shape[0])]

imgcat(img)

explainer = rise.RiseImageExplainer()
explainer.fit(lambda x: model.predict(x.reshape((-1, 28 * 28))),
              input_size,
              number_of_masks=1000,
              mask_granularity=8,
              mask_density=0.5,
              channels_last=True,
              single_channel=True)

sal = explainer.explain(img)

probas = model.predict(img.reshape((1, 28 * 28)))
top_classes_idx = probas.argsort()[0][::-1]
top_probas = probas[0][top_classes_idx]

fig, axes = plt.subplots(1, 5)
fig.set_size_inches(15, 6)
for i in range(5):
    axes[i].set_title(f'Cls: "{top_classes_idx[i]}" {top_probas[i] * 100: .1f}%')
    axes[i].axis('off')
    axes[i].imshow(img)
    axes[i].imshow(sal[top_classes_idx[i]], cmap='jet', alpha=0.4)
plt.tight_layout()
imgcat(fig)
