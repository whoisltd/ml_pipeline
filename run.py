# from ocr import ocr
from pipeline import app

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)