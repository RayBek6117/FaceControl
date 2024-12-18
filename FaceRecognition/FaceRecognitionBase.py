import face_recognition
from PIL import Image, ImageDraw

def face_rec():
    gal_face_img = face_recognition.load_image_file("img/Miras.jpeg")
    gal_face_location = face_recognition.face_locations(gal_face_img)

    justice_league_img = face_recognition.load_image_file("img/Adilbek.jpeg")
    justice_league_faces_locations = face_recognition.face_locations(justice_league_img)

    pil_img1 = Image.fromarray(gal_face_img)
    draw1 = ImageDraw.Draw(pil_img1)

    for(top, right, bottom, left) in gal_face_location:
        draw1.rectangle((left, top), (right,bottom), outline=(255, 255, 0), width=4)
    
    del draw1
    pil_img1.save("img/new_Miras.jpg")

def extracting_faces(img_path):
    count = 0
    faces = face_recognition.load_image_file(img_path)
    faces_locations = face_recognition.face_locations(faces)

    for face_location in faces_locations:
        top, right, bottom, left = face_location

        face_img = faces[top:bottom, left:right]
        pil_img = Image.fromarray(face_img)
        pil_img.save(f"img/{count}_face_img.jpg")
        count += 1

    return f"Found {count} face(s) in this photo"

def compare_faces(img1_path, img2_path):
    img1 = face_recognition.load_image_file(img1_path)
    img1_encodings = face_recognition.face_encodings(img1)[0]
    img2 = face_recognition.load_image_file(img2_path)
    img2_encodings = face_recognition.face_encodings(img2)[0]

    result = face_recognition.compare_faces([img1_encodings], img2_encodings)




def main():
    # extracting_faces("img/Adilbek.jpeg")
    compare_faces("img/Miras.jpeg")

if __name__ == "__main__":
    main()