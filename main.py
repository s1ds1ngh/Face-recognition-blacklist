import cv2
from face_detection_utils.detect_faces import detect_face
from face_detection_utils.utils import load_face_ids_from_db, compare_face_encodings
import face_recognition


def main():
    # Load face IDs, names, and statuses from the database
    face_ids = load_face_ids_from_db()

    # Open the camera
    cap = cv2.VideoCapture(1)
    cap.set(3, 640)  # Set Width
    cap.set(4, 480)  # Set Height
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set frame rate

    frame_count = 0
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Flip camera vertically
        if not ret:
            print("Failed to capture frame")
            break

        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame
            continue

        # Detect faces in the frame using Haar Cascade
        face_locations = detect_face(frame)

        # Convert the frame to RGB for face encoding
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for (top, right, bottom, left) in face_locations:
            # Compute the face encoding for the detected face
            face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
            if len(face_encodings) == 0:
                continue  # Skip if no face encoding is found

            encoding = face_encodings[0]
            name, status = compare_face_encodings(encoding, face_ids)

            # Determine the label and color based on the status
            if status == 'blacklisted':
                label = f"ALERT: {name} (Blacklisted)"
                color = (0, 0, 255)  # Red
            elif status == 'not_blacklisted':
                label = f"{name} (Not Blacklisted)"
                color = (0, 255, 0)  # Green
            else:
                label = "Unknown"
                color = (255, 255, 255)  # White

            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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