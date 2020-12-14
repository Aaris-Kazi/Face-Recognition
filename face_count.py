import face_recognition
# This program is to identify number of people in a image
images = face_recognition.load_image_file('./faces/elon musk.jpg')
locations = face_recognition.face_locations(images)
print(locations)
print(f'There are {len(locations)} people in the image') #This is to find number of people in the image