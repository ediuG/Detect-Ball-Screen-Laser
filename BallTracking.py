import numpy as np
import cv2
import cv2.cv as cv
from common import draw_str
__author__ = 'Guide-ce'

cap = cv2.VideoCapture(1)
cap.set(3, 720)
cap.set(4, 720)

# create mask for add effect
mask = np.zeros((600, 800, 3), np.uint8)
# frame + mask
new = np.zeros((600, 800, 3), np.uint8)
img = cv2.imread('mickey_mouse.png',-1)

def nothing(x):
	pass


def is_ball_increase(before, after):
	if after > before:
		print("increase")
		print("ball before = %d" % ball_before)
		print("ball count = %d" % ball_count)
		return True
	else:
		return False


class Ball(object):

	"""Track each ball in frame"""

	def __init__(self, position):
		super(Ball, self).__init__()
		self.position = position


green_color_hsv_min = cv.Scalar(40, 75, 77)
green_color_hsv_max = cv.Scalar(60, 185, 222)
red_color_hsv_min = cv.Scalar(0, 200, 120)
red_color_hsv_max = cv.Scalar(180, 256, 210)
orange_color_hsv_min = cv.Scalar(2, 165, 180)
orange_color_hsv_max = cv.Scalar(9, 256, 256)
blue_color_hsv_min = cv.Scalar(105, 160, 90)
blue_color_hsv_max = cv.Scalar(115, 256, 256)
yellow_color_hsv_min = cv.Scalar(20, 110, 154)
yellow_color_hsv_max = cv.Scalar(25, 235, 256)


cv2.namedWindow("Hough Circles Track bar")
cv2.createTrackbar(
	"Hough resolution", "Hough Circles Track bar", 1, 100, nothing)
cv2.createTrackbar(
	"Canny threshold", "Hough Circles Track bar", 75, 300, nothing)
cv2.createTrackbar(
	"Accumulator threshold", "Hough Circles Track bar", 45, 200, nothing)
cv2.createTrackbar("Min radius", "Hough Circles Track bar", 5, 50, nothing)
cv2.createTrackbar("Max radius", "Hough Circles Track bar", 100, 200, nothing)

ball_before = 0
new_ball = False
bounce = 0

while True:
	hough_resolution = cv2.getTrackbarPos(
		"Hough resolution", "Hough Circles Track bar")
	canny_threshold = cv2.getTrackbarPos(
		"Canny threshold", "Hough Circles Track bar")
	accumulator_threshold = cv2.getTrackbarPos(
		"Accumulator threshold", "Hough Circles Track bar")
	minRadius = cv2.getTrackbarPos("Min radius", "Hough Circles Track bar")
	maxRadius = cv2.getTrackbarPos("Max radius", "Hough Circles Track bar")

	ret, frame = cap.read()
	if ret:
		hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		hsv_blue = cv2.inRange(hsv_image, blue_color_hsv_min, blue_color_hsv_max)
		hsv_red = cv2.inRange(hsv_image, red_color_hsv_min, red_color_hsv_max)
		hsv_orange = cv2.inRange(hsv_image, orange_color_hsv_min, orange_color_hsv_max)
		hsv_green = cv2.inRange(hsv_image, green_color_hsv_min, green_color_hsv_max)
		hsv_yellow = cv2.inRange(hsv_image, yellow_color_hsv_min, yellow_color_hsv_max)
		green_yellow_hsv = cv2.bitwise_or(hsv_green, hsv_yellow)
		green_yellow_blue_hsv = cv2.bitwise_or(green_yellow_hsv, hsv_blue)
		# green_yellow_blue_red_hsv = cv2.bitwise_or(green_yellow_blue_hsv, hsv_red)
		green_yellow_blue_orange_hsv = cv2.bitwise_or(green_yellow_blue_hsv, hsv_orange)

		str_el = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
		open_morphed = cv2.morphologyEx(green_yellow_blue_orange_hsv, cv2.MORPH_OPEN, str_el)
		morphed = cv2.morphologyEx(open_morphed, cv2.MORPH_CLOSE, str_el)

		hsv_blur = cv2.GaussianBlur(morphed, (7, 7), 4, 4)
		cv2.imshow('blur', hsv_blur)
		circles = cv2.HoughCircles(hsv_blur, cv.CV_HOUGH_GRADIENT, 2, 720 / 9,
								   param1=canny_threshold, param2=accumulator_threshold, minRadius=minRadius,
								   maxRadius=maxRadius)
		ball_count = 0

		if circles is not None:
			circles = np.uint16(np.around(circles))
			for i in circles[0, :]:
				ball_count += 1
				# create ball object if ball's number is increase
				if is_ball_increase(ball_before, ball_count):
					ball = Ball((i[0], i[1]))
				# draw the outer circle
				cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
				# draw the center of the circle
				cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
				draw_str(frame, (i[0], i[1]), 'Pos {},{}'.format(i[0], i[1]))
				draw_str(frame, (i[0], i[1]+20), 'R {}'.format(i[2]))
				if i[2] > 25:
					new_ball = True
				if new_ball is True and i[2] < 15:
					# cv2.circle(mask, (i[0], i[1]), 20, (0, 0, 255), -1)
					x_offset = i[0] - 43
					y_offset = i[1] - 50
					# mask[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
					for c in range(0,3):
						mask[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1], c] = img[:, :, c] * (img[:, :, 3]/255.0) + mask[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1], c] * (1.0 - img[:, :, 3]/255.0)
					new_ball = False
					bounce += 1

		ball_before = ball_count
		draw_str(frame, (20, 20), 'ball count: %d' % ball_count)
		draw_str(frame, (20, 40), 'bounce: %d' % bounce)
		new = cv2.addWeighted(frame, 0.7, mask, 0.7, 0)
		cv2.imshow('new', new)
	else:
		continue
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
