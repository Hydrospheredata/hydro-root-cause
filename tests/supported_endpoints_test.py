import os
import sys
import unittest

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{SCRIPT_DIR}/..")

from utils import get_method_support_statuses

# with open(f"{SCRIPT_DIR}/../hs_demos/adult_columnar/serving.yaml") as f:
#     columnar_contract = HSContract.load(f)
#
# with open(f"{SCRIPT_DIR}/../hs_demos/adult_tensor/serving.yaml") as f:
#     tensor_contract = HSContract.load(f)
#
# with open(f"{SCRIPT_DIR}/../hs_demos/adult_scalar/serving.yaml") as f:
#     scalar_contract = HSContract.load(f)
#
# with open(f"{SCRIPT_DIR}/../hs_demos/mobilenet_v2_035/serving.yaml") as f:
#     image_contract = HSContract.load(f)
#
# with open(f"{SCRIPT_DIR}/../hs_demos/mnist/serving.yaml") as f:
#     mnist_contract = HSContract.load(f)
#
#
# class TestSupportedEndpoints(unittest.TestCase):
#
#     def test_columnar(self):
#         supported_endpoints = get_method_support_statuses(columnar_contract)
#         self.assertEqual(supported_endpoints, ['anchor'])
#
#     def test_tensor(self):
#         supported_endpoints = get_method_support_statuses(tensor_contract)
#         self.assertEqual(supported_endpoints, ['anchor'])
#
#     def test_scalar(self):
#         supported_endpoints = get_method_support_statuses(scalar_contract)
#         self.assertEqual(supported_endpoints, ['anchor'])
#
#     def test_image_w_batch_size(self):
#         supported_endpoints = get_method_support_statuses(image_contract)
#         self.assertEqual(supported_endpoints, ['rise'])
#
#     def test_mnist_contract(self):
#         supported_endpoints = get_method_support_statuses(mnist_contract)
#         self.assertEqual(supported_endpoints, ['rise'])
#
#
# if __name__ == '__main__':
#     unittest.main()
