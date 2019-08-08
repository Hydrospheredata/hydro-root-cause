import unittest
from utils import get_supported_endpoints
from contract import HSContract

with open("../hs_demos/adult/serving.yaml") as f:
    columnar_contract = HSContract.load(f)

with open("../hs_demos/adult_tensor/serving.yaml") as f:
    tensor_contract = HSContract.load(f)

with open("../hs_demos/adult_scalar/serving.yaml") as f:
    scalar_contract = HSContract.load(f)

with open("../hs_demos/mobilenet/serving.yaml") as f:
    image_contract = HSContract.load(f)


class TestSupportedEndpoints(unittest.TestCase):

    def test_columnar(self):
        supported_endpoints = get_supported_endpoints(columnar_contract)
        self.assertEqual(supported_endpoints, ['anchor'])

    def test_tensor(self):
        supported_endpoints = get_supported_endpoints(tensor_contract)
        self.assertEqual(supported_endpoints, ['anchor'])

    def test_scalar(self):
        supported_endpoints = get_supported_endpoints(scalar_contract)
        self.assertEqual(supported_endpoints, ['anchor'])

    def test_image_w_batch_size(self):
        supported_endpoints = get_supported_endpoints(image_contract)
        self.assertEqual(supported_endpoints, ['rise'])


if __name__ == '__main__':
    unittest.main()
