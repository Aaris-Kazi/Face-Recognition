from PIL import Image, ImageDraw
import face_recognition as fr
import numpy as np

img = fr.load_image_file('./faces/bill gates.jpg')
encoded = fr.face_encodings(img)[0]

img1 = fr.load_image_file('./faces/elon musk.jpg')
encoded1 = fr.face_encodings(img1)[0]

img2 = fr.load_image_file('./faces/donald trump.jpg')
encoded2 = fr.face_encodings(img2)[0]

known_faces_encoding = [
    encoded,
    encoded1,
    encoded2
]

known_faces = [
    'Bill Gates',
    'Elon Musk',
    'Donald Trump'
]

test_image = fr.load_image_file('./test.jpg')
test_location = fr.face_locations(test_image)
test_encoded = fr.face_encodings(test_image, test_location)

pil_image = Image.fromarray(test_image)
draw = ImageDraw.Draw(pil_image)

for (top, right, bottom, left) , face_encode in zip(test_location, test_encoded) :
    match = fr.compare_faces( known_faces_encoding,face_encode)
    name = "Unknown"
    
    print(match)
    if True in match :
        match_name = match.index(True)
        name = known_faces[match_name]
        print('Known')
        draw.rectangle(( (left, top), (right, bottom)), outline=(0,0,0))
        width, height =  draw.textsize(name)
        draw.rectangle(((left, bottom - height - 10), (right, bottom)), fill=(0,0,0),
        outline=(0,0,0))
        draw.text((left + 6, bottom - height - 5), name, fill= (255,255,255,255))
    else:
        print('Unknown')

del draw
pil_image.show()
