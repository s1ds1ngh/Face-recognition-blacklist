# database/create_database.py
import sqlite3
import face_recognition
import numpy as np


def extract_face_encoding(image_path):
    """
    Extracts the face encoding from an image file.
    Returns the encoding as a hexadecimal string.
    """
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if len(face_encodings) == 0:
        raise ValueError(f"No face detected in {image_path}")

    # Convert the encoding to a hexadecimal string for storage
    return face_encodings[0].tobytes().hex()


# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('database/db.sqlite3')
cursor = conn.cursor()

# Create the 'persons' table with an additional 'name' column
cursor.execute('''
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    status TEXT NOT NULL
)
''')

# Extract face encodings from the images
try:
    blacklisted_face_id = extract_face_encoding('database/11zon_cropped (1).jpeg')  # Blacklisted person
    not_blacklisted_face_id = extract_face_encoding('database/img.png')  # Non-blacklisted person
except ValueError as e:
    print(e)
    exit(1)

# Insert sample data into the database with names
sample_data = [
    (blacklisted_face_id, 'Siddharth Singh', 'blacklisted'),  # Blacklisted person
    (not_blacklisted_face_id, 'Rahul Sharma', 'not_blacklisted')  # Non-blacklisted person
]

cursor.executemany('INSERT OR IGNORE INTO persons (face_id, name, status) VALUES (?, ?, ?)', sample_data)

# Commit changes and close connection
conn.commit()
conn.close()

print("Database created and populated successfully.")