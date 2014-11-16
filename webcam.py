import cv2
import numpy as np
import datetime
import matplotlib.pyplot as plt

def main():
	dt = datetime.datetime
	show_track = False

	cap = cv2.VideoCapture(0)
	color = np.random.randint(0,255,(100,3))

	feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
	lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
	
	mask = None
	while 1:
		if not show_track:
			ret, frame = cap.read()
			cv2.imshow('Webcam', frame)
		else:
			try:
				ret, frame = cap.read()
				frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
				good_new = p1[st == 1]
				good_old = p0[st == 1]

				for i,(new, old) in enumerate(zip(good_new,good_old)):
					a,b = new.ravel()
					c,d = old.ravel()
					cv2.circle(frame, (a,b), 5, color[i].tolist(), -1)		
				cv2.imshow('Webcam', cv2.add(frame, mask))
				old_gray = frame_gray.copy()
				p0 = good_new.reshape(-1,1,2)
			except: # the inevitable
				None
		k = cv2.waitKey(10) & 0xff
		if k == 27: # esc exits
			break
		elif k == ord('s'): # saves capture
			fp = "captures/" + datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S") + ".png"
			cv2.imwrite(fp, frame)
		elif k == ord('t'): # t for track
			show_track = not show_track
			if show_track:
				ret, old_frame = cap.read()
				old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
				p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
				mask = np.zeros_like(old_frame)
		elif k == 32: # left arrow
			cv2.moveWindow('Webcam', 100, 100)

	cap.release()
	cv2.destroyAllWindows()

main()