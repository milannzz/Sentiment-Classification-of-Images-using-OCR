#Importing Libraries
import cv2
import pytesseract
import os
import numpy as np
import pandas as pd

# Importing flair
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

def img_to_sentiment(img) :

    #Resizing image to 1600x9000
    img = cv2.resize(img, None, fx=1.85, fy=1.85, interpolation=cv2.INTER_CUBIC)
    
    def img_dark_font(img):
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
        
        text1 = ""
        
        contours,_ =cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours: 
            x, y, w, h = cv2.boundingRect(cnt) 
                
            # Cropping the text block for giving input to OCR 
            cropped = img[y:y + h, x:x + w] 
            
            #Image to Text
            config = r'--oem 1'
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            textc = pytesseract.image_to_string(cropped,config =config)
            text1 = text1 + " "+textc
            return text1
    
    def img_light_font(img) :
        img = ~img
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
        
        text1 = ""
        
        contours,_ =cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours: 
            x, y, w, h = cv2.boundingRect(cnt) 
                    
            # Cropping the text block for giving input to OCR 
            cropped = img[y:y + h, x:x + w] 
            
            #Image to Text
            config = r'--oem 1'
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            textc = pytesseract.image_to_string(cropped,config =config)
            text1 = text1 + " "+textc
            return text1
    
    text1 = img_dark_font(img)
    text2 = img_light_font(img)
    
    if len(text1) > len(text2):
        text = text1
    else:
        text = text2
        
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

sentiments = []
for img in cv_img:
    sentiment = img_to_sentiment(img)
    sentiments.append(sentiment)

submission = pd.DataFrame(label)
submission["Category"] = sentiments
submission.to_csv('H:\Hacker Earth - love is love\submission.csv',index = False ,float_format ='%.0f')