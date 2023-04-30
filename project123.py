import pandas as pd
import matplotlib.pyplot as plp
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import ImageOps 
from PIL import Image
import keyboard

x = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
# print(pd.Series(y).value_counts())
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses = len(classes)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=2500,train_size=7500,random_state=0)
x_train_scale = x_train/255.0
x_test_scale = x_test/255.0

logreg = LogisticRegression(solver='saga',multi_class = 'multinomial')
logreg.fit(x_train_scale,y_train)

predict = logreg.predict(x_test_scale)
accuracy = accuracy_score(y_test, predict)

print('accuracy:',accuracy)

cam = cv2.VideoCapture(0)
z = True
while z:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    width, height = gray.shape
    upper_left = (int(width / 2 - 56), int(height / 2 - 56))
    bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
    cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
    image_pil = Image.fromarray(roi)
    gray_scale = image_pil.convert('L')

    resize_image = gray_scale.resize((28,28), Image.ANTIALIAS)
    image_inverted = ImageOps.invert(resize_image)

    pixel_filter = 20
    min_pixel = np.percentile(image_inverted, pixel_filter)

    cliped_image = np.clip(image_inverted-min_pixel,0,255)
    max_pixel = np.max(image_inverted)

    cv2.imshow('Output', gray)
    if(keyboard.is_pressed('q')):
        z = False
        cv2.waitKey(1)

frame.release()
cv2.destroyAllWindows()