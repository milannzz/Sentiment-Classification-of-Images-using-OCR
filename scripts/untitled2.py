import cv2
import pytesseract
import os
import glob

path = "H:\Hacker Earth - love is love\Data Files\Dataset"
cv_img = []
label = []

for img_filename in os.listdir(path):
        labell = img_filename
        label.append(labell)
        img = cv2.imread(os.path.join(path,img_filename))
        if img is not None:
            cv_img.append(img) 

def img_to_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
    
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    im2 = img.copy() 

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt) 
        # Drawing a rectangle on copied image 
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        # Cropping the text block for giving input to OCR 
        cropped = im2[y:y + h, x:x + w] 
        # Apply OCR on the cropped image 
        text = pytesseract.image_to_string(cropped)
    return text

corpus = []      
for img in cv_img:
    text  = img_to_text(img)
    corpus.append(text)

    
