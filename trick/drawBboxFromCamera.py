import cv2 

cap = cv2.VideoCapture(0)
cap.set(3,320); # 3 = CV_CAP_PROP_FRAME_WIDTH
cap.set(4,240); # 4 = CV_CAP_PROP_FRAME_HEIGHT

cv2.namedWindow("camera")
success,frame = cap.read()
cv2.rectangle(frame, (82,55), (182,190), (0,255,0), 5)	#img start_point final_point color
cv2.imshow("camera", frame)
cv2.waitKey(10000)
