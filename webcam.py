''' wow such tight coupling
		such great progam
			   beautiful

	all it needs is threading'''
import cv2
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys

def main():
	global DONT_CHANGE
	global KILL
	global CHANGE_TRACKING
	DONT_CHANGE = 0
	KILL = 1
	CHANGE_TRACKING = 2
	display = LiveDisplay()
	motion_tracker = MotionTracker()
	while 1:
		newframe = display.getLatest()
		if not motion_tracker.show_tracking:
			display.showFrame(newframe)
		else:
			try:
				coord_pairs = motion_tracker.getTrackedFrame(newframe)
				display.showTrackers(coord_pairs)
				display.showFrame(newframe)
			except: # the inevitable
				print str(sys.exc_info()[0])
		result = display.action(cv2.waitKey(10) & 0xff)
		if result == KILL:
			break
		elif result == CHANGE_TRACKING:
			prev = display.getPrev()
			motion_tracker.changeTracking(prev)

class MotionTracker:
	def __init__(self):
		self.feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
		self.lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
		self.show_tracking = False
		self.old_gray = None
		self.p0 = None
		self.mask = None

	def getTrackedFrame(self, new_frame):	
		self.new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
		self.p1, self.st, self.err = cv2.calcOpticalFlowPyrLK(self.old_gray, self.new_gray, self.p0, None, **self.lk_params)
		self.good_new = self.p1[self.st == 1]
		self.good_old = self.p0[self.st == 1]

		self.ab = []
		for (new, old) in zip(self.good_new,self.good_old):
			self.a,self.b = new.ravel()
			self.ab.append((self.a,self.b))
		self.old_gray = self.new_gray.copy()
		self.p0 = self.good_new.reshape(-1,1,2)
		return self.ab

	def startTracking(self, old_frame):
		self.old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
		self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask = None, **self.feature_params)
		self.mask = np.zeros_like(old_frame)
	
	# return coordinates of the outer limits of the image to scan
	def getBoundaries(self):
		self.ab = np.array(self.ab)
		d = {'home_coords': self.ab}
		neighborhood = pd.DataFrame(d)
		'''for pt in self.ab:
			self.friends = []
			for friend in self.ab:
				if not np.array_equal(friend[1:3],pt[1:3]): # don't be narcissistic
					bff = [friend[1]-pt[1], friend[2]-pt[2]]
					friendzone = 1.396 <= math.atan(bff[0]/bff[1]) <= 1.745
					if(friendzone): # <3
						friends.append(friend[0])
						
		for pt in self.ab:'''

	def changeTracking(self, old_frame):
		self.show_tracking = not self.show_tracking
		if self.show_tracking:
			self.startTracking(old_frame)


class LiveDisplay:
	def __init__(self):
		self.cap = cv2.VideoCapture(0)
		self.frame = None
		self.prev_frame = None
		self.ret = None

	def getLatest(self):
		if self.frame is not None:
			self.prev_frame = self.frame.copy()
		self.ret, self.frame = self.cap.read()
		return self.frame

	def getPrev(self):
		if self.prev_frame is not None:
			return self.prev_frame

	def stopDisplay(self):
		try:
			self.cap.release()
			cv2.destroyAllWindows()
		except:
			print "Display was not created or there was a problem closing the window"

	def saveFrame(self, thisFrame):
		self.dt = datetime.datetime
		self.fp = "captures/" + dt.now().strftime("%Y-%m-%d %H_%M_%S") + ".png"
		cv2.imwrite(self.fp, thisFrame)

	def showFrame(self, displayFrame):
		cv2.imshow('Webcam', displayFrame)

	def showTrackers(self, coords):
		for point in coords:
			cv2.circle(self.frame, point, 5, (0,0,255), -1)
	
	def action(self, key):
		if key == 27: # esc exits
			self.stopDisplay()
			return KILL
		elif key == ord('s'): # saves capture
			self.saveFrame(self.frame)
		elif key == ord('t'): # t for track
			return CHANGE_TRACKING
		elif key == ord('c'): # c centers window
			cv2.moveWindow('Webcam', 100, 100)
		elif key == ord('x'): # x raises window to top, scales
			cv2.moveWindow('Webcam', 0,-1080)
			cv2.resizeWindow('Webcam',1920,1080)
		return DONT_CHANGE
main()