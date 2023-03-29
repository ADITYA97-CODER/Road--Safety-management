import easyocr
from matplotlib import pyplot as plt
import cv2
text_reader = easyocr.Reader(['en']) #Initialzing the ocr
results = text_reader.readtext(cv2.imread('img.png') )
for (bbox, text, prob) in results:
    print(text)
plt.imshow(cv2.imread('img.png'))
plt.title("First Image")
plt.show()