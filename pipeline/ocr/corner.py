import requests
import numpy as np
from pipeline.ocr.functions import align_image, process_output 

def corner(img, url_model, threshold, targetSize):
    payload = {'instances': [img.tolist()]}
    res = requests.post(url_model, json=payload)
    data= res.json()['predictions'][0]
    results = process_output('corner', data, threshold, targetSize)
    
    crop_img = align_image(img, results)
    crop_img = np.array(crop_img)

    return crop_img
    