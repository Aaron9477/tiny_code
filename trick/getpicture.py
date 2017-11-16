import cv2

capture = cv2.VideoCapture('/media/Ubuntu/Jetson/WYZ/messing/video_pos.mp4')
#frame_img = cv2.QueryFrame(capture)
success = True
cv2.namedWindow("Video")
while(success):
	success,frame = capture.read()
	cv2.imshow("Video", frame)
	cv2.waitKey(1)
