
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime


def load_known_faces():
    
    ansh_image = face_recognition.load_image_file("faces/ansh.jpg")
    arushi_image = face_recognition.load_image_file("faces/arushi.jpg")
    donald_image = face_recognition.load_image_file("faces/donald.jpg")
    elon_image = face_recognition.load_image_file("faces/elon.jpg")

    known_encodings = [
        face_recognition.face_encodings(ansh_image)[0],
        face_recognition.face_encodings(donald_image)[0],
        face_recognition.face_encodings(arushi_image)[0],
        face_recognition.face_encodings(elon_image)[0],
    ]
    known_faces = ["Ansh", "Donald", "Arushi", "Elon"]
    return known_encodings, known_faces


def initialize_csv():
    """Initialize a CSV file for attendance."""
    now = datetime.now()
    current_date = now.strftime("%y-%m-%d")
    file = open(f"{current_date}.csv", "w+", newline="")
    line_writer = csv.writer(file)
    return file, line_writer


def recognize_faces(videorecord, known_encodings, known_faces, line_writer):
    
    students = known_faces.copy()
    while True:
        ret, frame = videorecord.read()
        if not ret:
            break

        fast_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(fast_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        frame_face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in frame_face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_faces[best_match_index]
                mark_attendance(name, students, line_writer, frame)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("a"):
            break


def mark_attendance(name, students, line_writer, frame):
    
    if name in students:
        current_time = datetime.now().strftime("%H:%M")
        line_writer.writerow([name, current_time])
        students.remove(name)

    cv2.putText(
        frame,
        f"{name} Present",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    
    videorecord = cv2.VideoCapture(0)
    known_encodings, known_faces = load_known_faces()
    file, line_writer = initialize_csv()

    try:
        recognize_faces(videorecord, known_encodings, known_faces, line_writer)
    finally:
        videorecord.release()
        cv2.destroyAllWindows()
        file.close()
        print("Attendance done for today.")
        print("Come back tomorrow to learn more!")


if __name__ == "__main__":
    main()
