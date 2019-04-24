from typing import Dict, Tuple, Callable, TypeVar, Sequence

import numpy as np
from skimage.transform import resize

T = TypeVar('T')  # The result type of prediction_fn


class RiseImageExplainer:
    def __init__(self):
        pass

    def fit(self,
            prediction_fn: Callable[[np.array], Sequence[T]],
            input_size: Tuple[int, int],
            number_of_masks: int = 1000,
            mask_granularity: int = 7,
            mask_density: float = 0.5,
            single_channel=False) -> 'RiseImageExplainer':
        """

        :param single_channel: For B&W images
        :return: Explainer with paramters stored inside
        :rtype: RiseImageExplainer
        :param prediction_fn: function which takes np.array with shape (batch_size, input_size[0], input_size[1], 3)
         and returns predicted labels. In most cases this will be `model.predict` method.
        :param input_size: Tuple with image width and height in pixels
        :param class_names: Mapping from class_id to class name. class_id has to be the result type of prediction_fn
        :param number_of_masks: Number of masks generated and evaluated by the explainer
        :param mask_granularity: Parameter specifies how many cells will be in one masks. Mask is a [mask_granularity df mask_granularity]
        grid.
        :param mask_density: Parameter specifies how many cells will be present in the mask. More dense the matrix will be, more parts of
        the original image will be present after its multiplication with mask.
        """
        if not (0.0 < mask_density < 1.0):
            raise ValueError("Invalid mask_density, mask density belongs to (0, 1)")
        if number_of_masks < 1:
            raise ValueError("Invalid number of masks, please specify positive integer)")
        if not (1 < mask_granularity < min(input_size)):
            raise ValueError("Invalid mask_granularity, mask_granularity belongs to (1, min(input_size))")
        if min(input_size) < 1.0:
            raise ValueError("Invalid input size")

        self.number_of_masks = number_of_masks
        self.mask_granularity = mask_granularity
        self.mask_density = mask_density
        self.input_size = input_size
        self.prediction_fn = prediction_fn
        self.single_channel = single_channel
        self.masks = self._generate_masks()  # Is it valid to store the same masks for everything?

        return self

    def _generate_masks(self, ) -> np.array:
        """
        Generate np.array of masks. Each element of this float array is a binary mask. It will
        be multiplied by the image to hide some parts of an image
        :return:
        """
        cell_size = np.ceil(np.array(self.input_size) / self.mask_granularity)
        up_size = (self.mask_granularity + 1) * cell_size

        grid = np.random.rand(self.number_of_masks, self.mask_granularity, self.mask_granularity) < self.mask_density
        grid = grid.astype('float32')

        masks = np.empty((self.number_of_masks, *self.input_size))

        for i in range(self.number_of_masks):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect', )[x:x + self.input_size[0], y:y + self.input_size[1]]

        if self.single_channel:
            masks = masks.reshape(-1, *self.input_size)
        else:
            masks = masks.reshape(-1, *self.input_size, 1)

        return masks

    def explain(self, x, batch_size=20) -> np.array:
        """
        Multiply df by different masks and look at the prediction results.
        :param batch_size:
        :param x: Image represented as an np.array of shape (input_size[0], input_size[1], 3)
        :return: Array which contains sailency maps for each class. [Class_id -> Saliency map]
        """
        predictions = []
        # Make sure multiplication is being done for correct axes
        print(x.shape, self.masks.shape)
        masked = x * self.masks
        for i in range(0, self.number_of_masks, batch_size):
            # TODO check for situation then self.number_of_masks%batch_size!=0
            print(i)
            masked_x = masked[i:min(i + batch_size, self.number_of_masks)]
            predictions.append(self.prediction_fn(masked_x))
        predictions = np.concatenate(predictions)
        saliency_map = predictions.T.dot(self.masks.reshape(self.number_of_masks, -1)).reshape(-1, *self.input_size)
        saliency_map = saliency_map / self.number_of_masks / self.mask_density
        return saliency_map
