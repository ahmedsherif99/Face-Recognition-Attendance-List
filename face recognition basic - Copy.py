import cv2
import face_recognition

imgElon = face_recognition.load_image_file('ahmed1.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
#imgTest = face_recognition.load_image_file('ahmed2.jpg')
#imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
cap = cv2.VideoCapture(0)
success, img = cap.read()
# img = captureScreen()
#imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgS)[0]
encodeTest = face_recognition.face_encodings(imgS)[0]
cv2.rectangle(img, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results, faceDis)
cv2.putText(img, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Ahmed Sherif', imgElon)
cv2.imshow('Ahmed Test', img)
cv2.waitKey(0)