import cv2

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
webcam = cv2.VideoCapture(0)

while True:

    success, img = webcam.read()
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Turns IMG Grayscale


    faceCoords = trained_face_data.detectMultiScale(g_img) #Finds face coordinated

    for (x,y,w,h) in faceCoords:
        cv2.rectangle(img, (x,y), (x + w, y + h), (0,0,255), 3) #Draws rectangle

    cv2.imshow("Face Detector",img)
    key = cv2.waitKey(1)

    #Quit when q/Q is pressed
    if key == 81 or key == 113:
        break

webcam.release()