import cv2
import numpy as np
import math

img = cv2.imread('digits.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

digits = [np.hsplit(row,100) for row in np.vsplit(img, 50)]
digits = np.array(digits)
print digits.shape
# convert list to matrix(rows=100, cols=50, dtype=(20x20 ndarray))
count = 0
labels = ['0','1','2','3','4','5','6','7','8','9']
for i in np.arange(50):
	for j in np.arange(100):
		fp = 'classification_training/' + labels[int(math.floor(count/500))] + '_' + str(count%500) + '.png'
		cv2.imwrite(fp, digits[i,j])
		count += 1
