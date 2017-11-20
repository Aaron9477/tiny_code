import cv2

capture = cv2.VideoCapture("/home/zq610/WYZ/JD_contest/raw_video/1.mp4")
#frame_img = cv2.QueryFrame(capture)
cv2.namedWindow("Video")
success,frame = capture.read()
cv2.rectangle(frame, (460,293), (976,525), (0,255,0), 5)
cv2.imshow("Video", frame)
cv2.waitKey(0)
