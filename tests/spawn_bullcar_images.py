# import os
# import sys
# import warnings
# from time import sleep
#
# warnings.simplefilter(action='ignore', category=FutureWarning)
#
# SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
#
# sys.path.append(f"{SCRIPT_DIR}/..")
#
# import numpy as np
# from imgcat import imgcat
# from client import HydroServingClient
# from keras.preprocessing import image
#
# import matplotlib
#
# matplotlib.use("module://imgcat")
#
# MODEL_NAME = "mobilenet_v2"
#
# hs_client = HydroServingClient("localhost:9090")
#
# mobile_net_035 = hs_client.get_model(MODEL_NAME, 1)
#
# print("Model: ", mobile_net_035)
#
#
# def load_img(path):
#     img = image.load_img(path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     return img, x
#
#
# img, img_arr = load_img(f"{SCRIPT_DIR}/../rise/examples/bullcar_crop.png")
# img_arr = img_arr.astype(np.float64)
#
# print("Explained image:")
# imgcat(img)
#
# print("Deploy mobilenet servable and send our image")
# mobilenet_servable = hs_client.deploy_servable(MODEL_NAME)
#
# try:
#     while True:
#         sleep(5)
#         prediction = mobilenet_servable(img_arr, _profile=True)['probabilities']
#         print("Ping...")
# finally:
#     mobilenet_servable.delete()
