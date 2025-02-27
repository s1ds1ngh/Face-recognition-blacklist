# main.py
import cv2
from face_detection_utils.detect_faces import detect_face
from face_detection_utils.utils import load_face_ids_from_db, compare_face_encodings


def main():
    # Load face IDs, names, and statuses from the database
    face_ids = load_face_ids_from_db()

    # Open the camera
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Detect faces in the frame
        face_encodings = detect_face(frame)

        for encoding in face_encodings:
            # Compare detected face with database entries
            name, status = compare_face_encodings(encoding, face_ids)

            if status == 'blacklisted':
                print(f"ALERT: Blacklisted person detected! Name: {name}")
            elif status == 'not_blacklisted':
                print(f"Person is not blacklisted. Name: {name}")
            else:
                print("Person not found in database.")

        # Display the frame
        cv2.imshow('Video Stream', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()