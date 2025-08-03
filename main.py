import cv2
#loadimg the haar cascade classifier
frontal_face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
# starting video capture from the inbuilt webcam
capture=cv2.VideoCapture(0)
if not capture.isOpened():
    print("Error opening video stream")
    exit

while True:
    ret,frame=capture.read() 
    if not ret:
        print("Failed to capture video")  
        break
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Detecting faces in the frame
    faces=frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    # cv2.rectangle(image, start_point, end_point, color, thickness)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow("Face Detection- PRESS q to Quit", frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
    

capture.release()
cv2.destroyAllWindows()