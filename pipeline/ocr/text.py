import requests
from pipeline.ocr.functions import process_output

def text(img, url_model, threshold, targetSize):
    payload = {'instances': [img.tolist()]}
    res = requests.post(url_model, json=payload)
    data= res.json()['predictions'][0]
    results = process_output('text', data, threshold, targetSize)
    return results