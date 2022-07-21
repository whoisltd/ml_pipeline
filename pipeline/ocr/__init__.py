from flask import Flask
from pipeline import app

from pipeline.ocr import call, corner, extract_infos, functions, text