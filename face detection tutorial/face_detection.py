import cv2
import numpy as np
import matplotlib.pyplot as plt

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_face(img):
    faceImg = img.copy()
    faceRec = faceCascade.detectMultiScale(faceImg, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faceRec:
        cv2.rectangle(faceImg, (x, y), (x+w, y+h), (255, 255, 255), 10)

    return faceImg

def detect_eyes(img):
    eyeImg = img.copy()
    eyeRec = eyeCascade.detectMultiScale(eyeImg, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in eyeRec:
        cv2.rectangle(eyeImg, (x, y), (x + w, y + h), (255, 255, 255), 10)
        
    return eyeImg

img = cv2.imread('image.jpg')
imgCopy1 = img.copy()
imgCopy2 = img.copy()
imgCopy3 = img.copy()

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

face = detect_face(imgCopy1)
plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite('face.jpg', face)

eyes = detect_eyes(imgCopy2)
plt.imshow(cv2.cvtColor(eyes, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite('eyes.jpg', eyes)

temp = detect_face(imgCopy3)
eyes_face = detect_eyes(temp)
plt.imshow(cv2.cvtColor(eyes_face, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite('face+eyes.jpg', eyes_face)