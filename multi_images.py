from PIL import Image
import face_recognition

#load the image
images = face_recognition.load_image_file('./faces/donald trump.jpg')
#encode the image
encoded = face_recognition.face_encodings(images)[0]

for face_location in encoded:
    top, right, left, bottom = face_location

    face_image = images[top:bottom, left:right]