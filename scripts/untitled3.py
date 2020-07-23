#Importing Libraries
import cv2
import pytesseract
import os
import numpy as np
import pandas as pd
import flair

#Importing Images
path = "H:\Hacker Earth - love is love\Data Files\Dataset"
cv_img = []

label = pd.read_csv(r"H:\Hacker Earth - love is love\Data Files\Test.csv")

for img_filename in label.Filename:
        img = cv2.imread(os.path.join(path,img_filename))
        cv_img.append(img) 

def img_to_sentiment(img) :
    
    #Resizing image to 1600x9000
    img_H,img_W,_ = img.shape
    if img_H <= 1600 or img_W <=900:
        fx = 1600/img_H
        fy = 900/img_W
        img = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    
    def img_dark_font(img):
        #Color to GRAY
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #Blur Image
        img = cv2.medianBlur(img,3)
        
        # Apply Bilinear Threshold to Image
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,11,2)
            
        #Noise Reduction Dilate and Erode
        kernel = np.ones((3,3), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        #Image to Text
        config = r'--oem 1'
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        text1 = pytesseract.image_to_string(img,config =config)
        return text1
    
    def img_light_font(img) :
        img = ~img
        #Color to GRAY
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #Blur Image
        img = cv2.medianBlur(img,3)
        
        # Apply Bilinear Threshold to Image
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,11,2)
            
        #Noise Reduction Dilate and Erode
        kernel = np.ones((3,3), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        #Image to Text
        config = r'--oem 1'
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        text2 = pytesseract.image_to_string(img,config =config)
        return text2
    
    text1 = img_dark_font(img)
    text2 = img_light_font(img)
    
    
    
    #Sentimental Analysis With VADER
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    sentiment1 = sid.polarity_scores(text1)
    sentiment2 = sid.polarity_scores(text2)
    
    if sentiment1["compound"] > sentiment2["compound"] :
        sentiment = sentiment1
    else:
        sentiment = sentiment2
                     
    sentimentt = sentiment                      
    def get_sentiment(sentiment):
        posi = sentiment["pos"] 
        neg = sentiment["neg"]
        if posi == neg :
            return "Random"
        elif posi > neg :
            return "Positive"
        elif neg > posi :
            return "Negative"
        
    sentiment = get_sentiment(sentiment)
    return sentimentt,sentiment