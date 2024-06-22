# import cv2

# def detect_face(imagePath):
#     img = cv2.imread(imagePath)
#     if img is None:
#         print(f"Error: Unable to read image file {imagePath}")
#         return False
#     gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     face_classifier = cv2.CascadeClassifier(
#         cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#     )
#     face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
#     return len(face) > 0

# imagePath = './images/image6.jpg'
# result = detect_face(imagePath)
# if result:
#     print("Face Detected")
# else:
#     print("No Face Detected")


import cv2
import numpy as np
import urllib.request

def detect_face(image_url):
    req = urllib.request.urlopen(image_url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    return len(faces) > 0