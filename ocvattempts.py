import numpy as np
import cv2
import cv

# Image capture (webcam)
# ''' #
cap = cv2.VideoCapture(0)
# Capture first frame
ret, frame = cap.read()
frameNum = 0

while(1):
	# Show img
	cv2.imshow(("Frame %d" %frameNum), frame)

	k = cv2.waitKey(0) & 0xff
	if k == 27: # esc exits
		break
	elif k == ord('s'): # saves capture
		cv2.imwrite(("captures/%d.png" %frameNum), frame)
	elif k == ord('n'): # new capture
		cv2.destroyWindow(("Frame %d" %frameNum))
		ret, frame = cap.read()
		frameNum += 1

cap.release()
cv2.destroyAllWindows()

# Video capture
''' #
cap = cv2.VideoCapture(0)

fgbg = cv2.BackgroundSubtractorMOG()

while(1):
	ret, frame = cap.read()

	fgmask = fgbg.apply(frame)

	cv2.imshow('Background Subtracted', fgmask)
	cv2.imshow('Webcam', frame)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
'''