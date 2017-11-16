import cv2 

cap = cv2.VideoCapture(0)
cap.set(3,320); # 3 = CV_CAP_PROP_FRAME_WIDTH
cap.set(4,240); # 4 = CV_CAP_PROP_FRAME_HEIGHT
while(True):
	# read from camera first

    ret, firsr_frame = cap.read()
    cv2.imshow("Video", firsr_frame)
	# exit to let tld use camera
    cv2.waitKey(1)
