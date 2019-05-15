import unittest
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from anchor2.anchor2.utils import DiscretizerTransformer

import warnings


class TestTabularExplanations(unittest.TestCase):
    warnings.simplefilter("ignore")

    def test_dummy(self):
        data = np.zeros((200, 7))
        # np.random.seed(42)

        random_seeds = np.random.randn(200, 5)
        mean = np.random.randint(-5, 5, size=5)
        stds = (np.random.rand(5) - 0.5) * 10
        stds += 0.001  # To prevent items collapsing into constant on std == 0

        data[:, :5] = (random_seeds - mean) * stds
        data[:, 4] = 2  # Make some dummy column which is a constant

        data[:, 5] = np.random.randint(0, 2, size=200)
        data[:, 6] = 1 - data[:, 5]

        discretizer = DiscretizerTransformer()
        discretizer.fit(data, ordinal_indices=list(range(5)), oh_encoded_categories={"OH_example1": [5, 6]})

        transformed_data = discretizer.transform(data)
        self.assertEqual(transformed_data.shape[1], 6)
        self.assertEqual(data.shape[0], transformed_data.shape[0])
        restored_data = discretizer.inverse_transform(transformed_data)

        with self.subTest("test same shape"):
            # Check that shapes of data are the same
            self.assertEqual(restored_data.shape, data.shape)

        with self.subTest("test same categorical decoding"):
            # Check that categorical columns are decoded correctly
            self.assertTrue(np.all(restored_data[:, 5:] == data[:, 5:]))

        with self.subTest("test whether nans are present"):
            # Check that there are no NANs
            self.assertEqual(0, np.sum(restored_data != restored_data))
