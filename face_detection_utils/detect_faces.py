import cv2


def detect_face(frame):
    """
    Detects faces in the given frame using Haar Cascades.
    Returns face locations as (x, y, w, h).
    """
    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the frame to grayscale (required by Haar Cascades)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Convert face locations to (top, right, bottom, left) format
    face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
    return face_locations