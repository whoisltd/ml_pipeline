from flask import Flask

app = Flask(__name__)

from ocr import call, corner, extract_infos, functions, text