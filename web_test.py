from flask import Flask, request, jsonify, render_template
import subprocess
import os
import cv2 as cv
import time

from PIL import Image
from io import BytesIO


camera = cv.VideoCapture(0)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/imgTest', methods=['POST'])
def tone_analysis():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    img_receive = request.files['image']
    img_byte = img_receive.read()
    im = Image.open(BytesIO(img_byte))
    im.save('./static/test.jpg', im.format)

    code_path = r'./'
    subprocess.run(['python', os.path.join(code_path, 'tone_analysis.py')])

    with open(os.path.join('./result/', 'result.txt'), 'r', encoding='cp949') as f:
        result = f.read()
        word = result

    return jsonify({"your_tone": word})

if __name__ == '__main__':
    app.run('0.0.0.0', port=2500, debug=True)