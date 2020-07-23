#Importing Libraries
import cv2
import pytesseract
import os
import numpy as np
import pandas as pd
import flair
from flair.models import TextClassifier
from flair.data import Sentence
flair.cache_root = 'my/cache/H/flair/cache'
classifier = TextClassifier.load('en-sentiment')

#Importing Images
path = "H:\Hacker Earth - love is love\Data Files\Dataset"
cv_img = []

label = pd.read_csv(r"H:\Hacker Earth - love is love\Data Files\Test.csv")

for img_filename in label.Filename:
        img = cv2.imread(os.path.join(path,img_filename))
        cv_img.append(img) 


def img_preprocessing(img):
    
        #Resizing image (300DPI)
        img = cv2.resize(img, None, fx=1.85, fy=1.85, interpolation=cv2.INTER_CUBIC)
        
        #Color to GRAY
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #Blur Image
        img = cv2.medianBlur(img,3)
        
        # Apply Bilinear Threshold to Image
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,17,5)
            
        #Noise Reduction Dilate and Erode
        kernel = np.ones((3,3), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        return img
    
def img_to_text (img):
    #Image to Text
    config = r'--oem 1 --psm 3'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(img,config =config)
    return text
    
def sentiment_analysis(text):
        
    if not text.strip():
        return "Random"
    else:
        text = Sentence(text)
        classifier.predict(text)
        sentiment = str(text.labels[0]).split()[0]
    
        if sentiment == "POSITIVE":
            return "Positive"
        elif sentiment == "NEGATIVE":
            return "Negative"
        else:
            return "Random"
        
img = cv2.imread(r'H:\Hacker Earth - love is love\Data Files\Dataset\test761.jpg')

cv2.imshow("image", img)
cv2.waitKey(0)

img = img_preprocessing(img)

cv2.imshow("preprocessed",img)
cv2.waitkey(0)

text = img_to_text(img)
print(text)

sentiment = sentiment_analysis(text)
print(sentiment)

sentiments = []


submission = pd.DataFrame(label)
submission["Category"] = sentiments
submission.to_csv('H:\Hacker Earth - love is love\submission.csv',index = False ,float_format ='%.0f')