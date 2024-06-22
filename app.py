from flask import Flask, request, jsonify
from faceDetection import detect_face

app = Flask(__name__)

@app.route('/detect_face', methods=['POST'])
def detect_face_endpoint():
    image_url = request.json['image_url']
    result = detect_face(image_url)
    return jsonify({'face_detected': result})


if __name__ == '__main__':
    app.run(debug=True)

