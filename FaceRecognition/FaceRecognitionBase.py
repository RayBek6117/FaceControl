import face_recognition
import cv2
import os
from PIL import Image

# Function to load and encode faces from the "img" folder
def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []

    for file_name in os.listdir(known_faces_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):  # Process image files only
            img_path = os.path.join(known_faces_dir, file_name)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:  # Ensure face was found
                known_encodings.append(encodings[0])
                name = os.path.splitext(file_name)[0]  # Extract name from file name
                known_names.append(name)

    return known_encodings, known_names

# Real-time face recognition
def real_time_face_recognition(known_encodings, known_names):
    video_capture = cv2.VideoCapture(0)  # Open webcam (0 is the default camera)

    if not video_capture.isOpened():
        print("Error: Cannot open webcam.")
        return

    while True:
        ret, frame = video_capture.read()  # Capture a frame from the webcam
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Resize frame for faster face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Find faces in the frame
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare the face encoding with known faces
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            # Find the closest match if there are encodings
            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()  # Get index of the minimum distance
                if face_recognition.compare_faces([known_encodings[best_match_index]], face_encoding)[0]:
                    name = known_names[best_match_index]
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

            # Draw a rectangle and label on the frame
            top, right, bottom, left = face_location
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    video_capture.release()
    cv2.destroyAllWindows()


# Main function
def main():
    known_faces_dir = "img"
    known_encodings, known_names = load_known_faces(known_faces_dir)

    if not known_encodings:
        print("No known faces loaded. Please add images to the 'img' folder.")
        return

    print(f"Loaded {len(known_encodings)} known face(s). Starting real-time recognition...")
    real_time_face_recognition(known_encodings, known_names)

if __name__ == "__main__":
    main()
