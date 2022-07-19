from flask import Flask

app = Flask(__name__)

from face_matching import call, box_utils, compare, mtcnn