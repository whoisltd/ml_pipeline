import json
from flask import jsonify, request
import numpy as np

# from corner import corner
# from functions import url_to_image
# from text import text
# from ocr import extract_infos

from ocr import app, corner, extract_infos, text
from ocr.functions import url_to_image

THRESHOLD = 0.3
imageSize = 512
corner_url = 'http://localhost:8501/v1/models/corner:predict'
text_url = 'http://localhost:8501/v1/models/text:predict'
seq2seq_url = 'vietocr/seq2seqocr.pth'
targetSize = { 'w': imageSize, 'h': imageSize }
   
@app.route('/api/v1/ocr', methods=['POST'])
def postdata():
    data = json.loads(request.data)
    data = data['url']
    image = url_to_image(data)
    image = np.array(image)

    targetSize['h'] = image.shape[0]
    targetSize['w'] = image.shape[1]

    corner_img = corner(image, corner_url, THRESHOLD, targetSize)

    targetSize['h'] = corner_img.shape[0]
    targetSize['w'] = corner_img.shape[1]

    textt = text(corner_img, text_url, THRESHOLD, targetSize)

    test = extract_infos(seq2seq_url)

    data = test.OCR(corner_img, textt)
    
    return jsonify(data)