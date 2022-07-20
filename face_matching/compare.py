import math
import cv2
import numpy as np
from face_matching.box_utils import url_to_image
import requests
import tensorflow as tf

# from face_compare.model import facenet_model, img_to_encoding

# load model
# model1 = model.facenet_model(input_shape=(3, 96, 96))

def get_embedding(url, model):
    image = url_to_image(url)
    boxes = model.detect(image)
    face = tf.make_tensor_proto(boxes)
    face = tf.make_ndarray(face)
    x,y,w,h = face[0]
    face = image[int(y):int(y)+int(h), int(x):int(x)+int(w)]
    return face

def img_to_encoding(image, facenet_url):
    # Resize for model
    factor_0 = 160 / image.shape[0]
    factor_1 = 160 / image.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (int(image.shape[1] * factor), int(image.shape[0] * factor))
    resized = cv2.resize(image, dsize)

    diff_0 = 160 - resized.shape[0]
    diff_1 = 160 - resized.shape[1]
    print(diff_0, diff_1)
    img_pixels = np.pad(resized, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')

    if img_pixels.shape[0:2] != (160, 160):
        img_pixels = cv2.resize(img_pixels, (160, 160))

    # img_pixels = tensorflow.keras.preprocessing.image.img_to_array(img) #what this line doing? must?
    
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    
    # img_pixels /= 255
    # img_pixels *= 255
    img_pixels = np.float64(img_pixels) / 127.5
    img_pixels -= 1
    
    payload = {'instances': img_pixels.tolist()}
    res = requests.post(facenet_url, json=payload)
    # embedding = model.predict(img_pixels)[0].tolist()
    embedding = res.json()['predictions']
    # resized = cv2.resize(image, (160, 160))
    # Swap channel dimensions
    # input_img = resized[...,::-1]
    # # Switch to channels first and round to specific precision.
    # input_img = np.around(np.transpose(input_img, (2,0,1))/255.0, decimals=12)
    # x_train = np.array([input_img])
    # embedding = model.predict_on_batch(x_train)
    return embedding
def findEuclideanDistance(source_representation, test_representation):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    # print("1:", euclidean_distance)
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    # print("2:", euclidean_distance)
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def distance_to_confident(face_distance, threshold):
    if face_distance > threshold:
        range = (1.64 - threshold)
        linear_val = (1.64 - face_distance) / (range * 2)
        return linear_val
    else:
        range = threshold
        linear_val = 1 - (face_distance / (range * 2))
        return linear_val + ((1 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))