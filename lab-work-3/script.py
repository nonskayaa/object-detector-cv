import cv2
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

prev_time = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    smile_detected = False
    eyes_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        smile_roi_gray = roi_gray[int(h / 2):h, 0:w]
        smile_roi_color = roi_color[int(h / 2):h, 0:w]

        # Детекция улыбки
        smiles = smile_cascade.detectMultiScale(smile_roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        if len(smiles) > 0:
            smile_detected = True
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(smile_roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

        eyes_roi_gray = roi_gray[0:int(h / 2), 0:w]
        eyes_roi_color = roi_color[0:int(h / 2), 0:w]

        eyes = eye_cascade.detectMultiScale(eyes_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        eyes = [eye for eye in eyes if eye[2] < w // 4 and eye[3] < h // 4] 

        if len(eyes) >= 2:  
            eyes_detected = True
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(eyes_roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    if not smile_detected:
        cv2.putText(frame, "Smile :)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if not eyes_detected:
        cv2.putText(frame, "Open your eyes", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    current_time = time.time()
    time_difference = current_time - prev_time
    fps = 1 / time_difference
    prev_time = current_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Social Robot Interface", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
