import face_recognition

#load the image
images = face_recognition.load_image_file('./faces/donald trump.jpg')
#encode the image
encoded = face_recognition.face_encodings(images)[0]
# print(encoded)

#load the image
unknow_images = face_recognition.load_image_file('./test.jpg')
#encode the image
unknown_encoded = face_recognition.face_encodings(unknow_images)[0]
# print(unknown_encoded)

# Comparing the two images
results = face_recognition.compare_faces([encoded], unknown_encoded)
print(results)
if results[0] :
    print('This is a match')
else:
    print('This is not a match')