from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import cv2

class ocr:
    def __init__(self, url_model):
        config = Cfg.load_config_from_name('vgg_seq2seq')
        config['weights'] = url_model
        config['cnn']['pretrained']=False
        config['device'] = 'cpu'
        config['predictor']['beamsearch']=False 
        self.predictor = Predictor(config)

    def OCR(self, img, data):
        boxes = list(data.values())
        labels = list(data.keys())

        dict = {'data':{}, 'confidence': {}}

        for i in range(len(boxes)):
            boxes[i] = sorted(boxes[i] , key=lambda k: [k[0], k[1]])
            box = boxes[i]
            for j in range(len(box)):
                text_img = img[int(box[j][1]):int(box[j][3]), int(box[j][0]):int(box[j][2])]
                cv2.imwrite('text_img' + str(i) + '.png', text_img)
                if labels[i] in dict['data']:
                    text, conf = self.predictor.predict(Image.fromarray(text_img), True)
                    dict['data'][labels[i]] = text + ', ' + dict['data'][labels[i]]
                    dict['confidence'][labels[i]] = (conf + dict['confidence'][labels[i]])/2
                else:
                    text, conf = self.predictor.predict(Image.fromarray(text_img), True)
                    dict['data'][labels[i]] = text
                    dict['confidence'][labels[i]] = conf
        return dict