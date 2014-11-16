import cv2
import numpy as np
import datetime

def main():
	dt = datetime.datetime

	cap = cv2.VideoCapture(0)
	filt = cv2.BackgroundSubtractorMOG()
	frameNum = 0
	useMask = False

	while 1:
		ret, frame = cap.read()
		frameNum+=1
		mask = filt.apply(frame)

		if useMask:
			cv2.imshow('Webcam', mask)
		else:
			cv2.imshow('Webcam', frame)

		k = cv2.waitKey(1) & 0xff
		if k == 27: # esc exits
			break
		elif k == ord('s'): # saves capture
			fp = "captures/" + datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S") + ".png"
			if useMask:
				cv2.imwrite(fp, mask)
			else:
				cv2.imwrite(fp, frame)
		elif k == ord('a'): # change between b/w and norm
			useMask = not useMask
	cap.release()
	cv2.destroyAllWindows()

main()