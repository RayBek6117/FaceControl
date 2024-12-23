import face_recognition
import cv2
import csv
import ast  # Для преобразования строкового представления списков в списки
from datetime import datetime
import time

# Функция для загрузки известных лиц из CSV-файла
def load_known_faces_from_csv(csv_file):
    known_encodings = []
    known_names = []

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Пропускаем строку заголовков
        for row in reader:
            name, encoding_str = row
            encoding = ast.literal_eval(encoding_str)  # Преобразуем строку в список
            known_encodings.append(encoding)
            known_names.append(name)

    return known_encodings, known_names

# Функция для добавления записи в журнал
def log_entry_to_csv(name, action, csv_file):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")  # Текущая дата и время

    # Записываем в журнал каждое действие (приход или уход)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, action, current_time])

    print(f"Запись добавлена для {name}: {action} - {current_time}")

# Функция для распознавания лиц в реальном времени
def real_time_face_recognition(known_encodings, known_names, log_csv_file):
    video_capture = cv2.VideoCapture(0)  # Открываем веб-камеру (0 — стандартная камера)

    if not video_capture.isOpened():
        print("Ошибка: не удалось открыть веб-камеру.")
        return

    face_last_seen = {}  # Словарь для отслеживания последнего времени появления лица

    while True:
        ret, frame = video_capture.read()  # Считываем кадр с веб-камеры
        if not ret:
            print("Ошибка: не удалось захватить кадр.")
            break

        # Уменьшаем размер кадра для ускорения обработки
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Ищем лица на кадре
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Сравниваем текущую кодировку лица с известными
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            # Ищем ближайшее совпадение
            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()  # Индекс минимальной дистанции
                if face_recognition.compare_faces([known_encodings[best_match_index]], face_encoding, tolerance=0.4)[0]:
                    name = known_names[best_match_index]  # Имя найденного лица
                else:
                    name = "Неизвестный"
            else:
                name = "Неизвестный"

            # Если лицо распознано
            if name != "Неизвестный":
                current_time = time.time()

                if name not in face_last_seen or (current_time - face_last_seen[name]['time'] > 5):
                    # Если лицо появляется впервые или прошло больше 5 секунд
                    if name not in face_last_seen or face_last_seen[name]['action'] == 'Выход':
                        print(f"Распознано лицо: {name}. Подтвердите вход (y/n): ")
                        confirmation = input().strip().lower()
                        if confirmation == 'y':
                            log_entry_to_csv(name, 'Вход', log_csv_file)
                            face_last_seen[name] = {'time': current_time, 'action': 'Вход'}
                    elif face_last_seen[name]['action'] == 'Вход':
                        print(f"Распознано лицо: {name}. Подтвердите выход (y/n): ")
                        confirmation = input().strip().lower()
                        if confirmation == 'y':
                            log_entry_to_csv(name, 'Выход', log_csv_file)
                            face_last_seen[name] = {'time': current_time, 'action': 'Выход'}

            # Отображаем рамку вокруг лица и имя
            top, right, bottom, left = face_location
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Рисуем прямоугольник
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)  # Добавляем текст

        # Отображаем обработанный кадр
        cv2.imshow('Face Recognition', frame)

        # Выход из программы по нажатию клавиши 'qq'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождаем ресурсы камеры и закрываем окна
    video_capture.release()
    cv2.destroyAllWindows()

# Главная функция
def main():
    known_faces_csv = "known_faces.csv"  # Файл с кодировками лиц
    log_csv_file = "attendance_log.csv"  # Журнал посещений

    # Загружаем известные лица
    known_encodings, known_names = load_known_faces_from_csv(known_faces_csv)

    if not known_encodings:
        print("Не удалось загрузить известные лица. Сначала выполните обработку изображений.")
        return

    print(f"Загружено {len(known_encodings)} известных лиц. Запуск распознавания...")
    real_time_face_recognition(known_encodings, known_names, log_csv_file)

if __name__ == "__main__":
    main()
