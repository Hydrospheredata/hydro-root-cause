import unittest
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from anchor2.anchor2.utils import DiscretizerTransformer


class TestTabularExplanations(unittest.TestCase):

    def test_shape(self):
        data = np.zeros((200, 7))
        data[:, :5] = (np.random.rand(200, 5) - np.random.randint(-5, 5, size=5)) * np.random.randint(-5, 5, size=5)
        data[:, 5] = np.random.randint(0, 2, size=200)
        data[:, 6] = 1 - data[:, 5]

        discretizer = DiscretizerTransformer()
        discretizer.fit(data, ordinal_indices=list(range(5)), oh_encoded_categories={"OH_example1": [5, 6]})

        transformed_data = discretizer.transform(data)
        self.assertEqual(transformed_data.shape[1], 6)
        self.assertEqual(data.shape[0], transformed_data.shape[0])
        restored_data = discretizer.inverse_transform(transformed_data)

        # Check that shapes of data are the same
        self.assertEqual(restored_data.shape, data.shape)

        # Check that categorical columns are decoded correctly
        self.assertTrue(np.all(restored_data[:, 5:] == data[:, 5:]))
