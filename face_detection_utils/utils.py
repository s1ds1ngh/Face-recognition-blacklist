import sqlite3
import numpy as np
import face_recognition

def load_face_ids_from_db():
    """
    Loads all face IDs, names, and statuses from the database.
    """
    conn = sqlite3.connect('database/db.sqlite3')
    cursor = conn.cursor()

    cursor.execute('SELECT face_id, name, status FROM persons')
    rows = cursor.fetchall()

    conn.close()

    # Convert face IDs from hexadecimal strings to numpy arrays
    face_ids = [(np.frombuffer(bytes.fromhex(row[0]), dtype=np.float64), row[1], row[2]) for row in rows]
    return face_ids


def compare_face_encodings(detected_encoding, face_ids):
    """
    Compares the detected face encoding with stored face IDs.
    Returns the name and status if a match is found, otherwise None.
    """
    for stored_encoding, name, status in face_ids:
        # Compare encodings using Euclidean distance
        if face_recognition.api.compare_faces([stored_encoding], detected_encoding, tolerance=0.5)[0]:
            return name, status
    return None, None