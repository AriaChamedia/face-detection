import cv2
frontal_face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to access the camera")
    exit()
    
while True:
    ret, frame= cap.read()
    if not ret:
        print("Failed to capture video")
        break
    else:
        gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=frontal_face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minSize=(30, 30), minNeighbors=5,)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
        cv2.putText(frame,f"People count: {len(faces)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Face detection, PRESS 'q' to quit", frame)
        key=cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()
        
        
        