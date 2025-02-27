# face_recognition/detect_faces.py
import cv2
import face_recognition


def detect_face(frame):
    """
    Detects faces in the given frame and returns their encodings.
    """
    # Convert the frame from BGR to RGB (required by face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    return face_encodings