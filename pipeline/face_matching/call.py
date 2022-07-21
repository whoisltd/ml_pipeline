from time import time
from flask import jsonify, request, json
import numpy as np
from pipeline.face_matching.compare import findEuclideanDistance, img_to_encoding, l2_normalize, get_embedding
from pipeline.face_matching.mtcnn import MTCNN
from pipeline.face_matching import app
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

pnet_url = 'http://localhost:8501/v1/models/p_net:predict'
rnet_url = 'http://localhost:8501/v1/models/r_net/versions/1:predict'
onet_url = 'http://localhost:8501/v1/models/o_net:predict'
facenet_url = 'http://localhost:8501/v1/models/face_net:predict'
@app.route('/api/v1/face_matching', methods=['POST'])
def api_face_matching():
    data = json.loads(request.data)
    url_image1 = data['url_image1']
    url_image2 = data['url_image2']
    # Time to load images
    
    pnet_url = data['pnet_url']
    rnet_url = data['rnet_url']
    onet_url = data['onet_url']
    facenet_url = data['facenet_url']

    start = time()
    mtcnn = MTCNN(pnet_url, rnet_url, onet_url)
    face1 = get_embedding(url_image1, mtcnn)
    face2 = get_embedding(url_image2, mtcnn)
    end = time()
    print('Time to load models: ', end - start)

    embedding_one = img_to_encoding(face1, facenet_url)
    embedding_two = img_to_encoding(face2, facenet_url)
    # Convert list to array
    embedding_one = np.array(embedding_one)
    embedding_two = np.array(embedding_two)
    # print(embedding_one.shape)
    # Calculate distance
    dist = findEuclideanDistance(l2_normalize(embedding_one), l2_normalize(embedding_two))

    # dist = np.linalg.norm(embedding_one - embedding_two)
    print(f'Distance between two images is {dist}')
    if dist > 1.05:
        print('These images are of two different people!')
    else:
        print('These images are of the same person!')
    
    return jsonify({'distance': dist})

if __name__ == '__main__':
    postdata(
    'https://funflow-sp.sgp1.digitaloceanspaces.com/upload/blob_1f04290d-e3a9-4fbd-be2b-6cc3dee40e06',
    'https://funflow-sp.sgp1.digitaloceanspaces.com/upload/blob_1f04290d-e3a9-4fbd-be2b-6cc3dee40e06')

