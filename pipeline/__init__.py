from flask import Flask

app = Flask(__name__)

import pipeline.face_matching, pipeline.ocr