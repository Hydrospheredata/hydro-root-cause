from unittest import TestCase, main as unittest_main

import numpy as np
from hypothesis import given, event
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers, tuples


class TestAnchor(TestCase):

    def test_sample(self):
        pass
        # TODO write test
        self.assertEqual(2, 2)

    @given(arrays(dtype=np.float32, shape=tuples(integers(0, 20000), integers(0, 200))))
    def test_float_dataset(self, x):
        # TODO write test
        assert x.shape[1] > -1






