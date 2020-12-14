from PIL import Image
import numpy as np
import face_recognition

#load the image
images = face_recognition.load_image_file('./faces/donald trump.jpg')
#encode the image
encoded = face_recognition.face_locations(images)

for face_location in encoded:
    top, right, bottom, left =  face_location

    face_image = images[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()