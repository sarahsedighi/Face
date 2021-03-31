"""
@author: Sara

Face and Eye Detection using Haar Cascade Classifier and MTCNN

"""

#%% Part 1: face and eye detection using Haar Cascade classifier

import cv2

# 1-1: face detection 

# you can find haarcascade_frontalface_default.xml here:
# https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('img.jpg')
gray_img = cv2.cvtColor(img, cv.COLOR_BGR2GRAY) # no need to make gray image

face_detect = face_cascade.detectMultiScale(gray_img, 1.1, 3)

print(face_detect) # returns [(x, y), (w, h)]

for (x, y, w, h) in face_detect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=5)
        
# or

# boxes = face_cascade.detectMultiScale(img)

# for box in boxes:
#     x1, y1, width, height = box
#     x2, y2 = x1 + width, y1 + height

#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Image', img)
cv2.waitKey(0)

#%%
# 1-2: eye detection  

eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

img = cv2.imread('img.jpg')
gray_img = cv2.cvtColor(img, cv.COLOR_BGR2GRAY)

eye_detect = eye_cascade.detectMultiScale(gray_img)

for (x, y, w, h) in eye_detect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=4)

cv2.imshow('Image', img)
cv2.waitKey(0)

#%%
# 1-3: face and eye detection

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

img = cv2.imread('img.jpg')
gray_img = cv2.cvtColor(img, cv.COLOR_BGR2GRAY)
face_detect = face_cascade.detectMultiScale(gray_img, 1.1, 4)
eye_detect = eye_cascade.detectMultiScale(gray_img)

for (x, y, w, h) in face_detect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness=3)

    for (ex, ey, ew, eh) in eye_detect:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), thickness=3)

cv.imshow('Image', img)
cv.waitKey(0)

#%%
# PART 2: face detection using MTCNN
    
import cv2
import mtcnn

face_detector = mtcnn.MTCNN()
img = cv2.imread('img.jpg')
conf_t = 0.99

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = face_detector.detect_faces(img_rgb)

print(results)
for res in results:
    x1, y1, width, height = res['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    confidence = res['confidence']
    if confidence < conf_t:
        continue
    key_points = res['keypoints'].values()

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
    cv2.putText(img, f'conf: {confidence:.3f}', (x1, y1), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)

    for point in key_points:
        cv2.circle(img, point, 5, (0, 255, 0), thickness=-1)

cv2.imshow('Image', img)
cv2.waitKey(0)    
    
    



















