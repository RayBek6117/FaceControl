import face_recognition
import os
import csv

# Функция для загрузки лиц и сохранения их кодировок в CSV-файл
def save_known_faces_to_csv(known_faces_dir, csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Encoding"])  # Записываем строку заголовков

        # Перебираем все файлы в указанной папке
        for file_name in os.listdir(known_faces_dir):
            # Обрабатываем только изображения
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(known_faces_dir, file_name)
                image = face_recognition.load_image_file(img_path)  # Загружаем изображение
                encodings = face_recognition.face_encodings(image)  # Извлекаем кодировки лица

                # Если лицо найдено, сохраняем его кодировку
                if encodings:
                    name = os.path.splitext(file_name)[0]  # Извлекаем имя из названия файла
                    encoding = encodings[0]
                    writer.writerow([name, list(encoding)])  # Записываем имя и кодировку в CSV

    print(f"Кодировки сохранены в {csv_file}.")

# Главная функция для обработки
def main():
    known_faces_dir = "img"  # Папка с изображениями
    csv_file = "known_faces.csv"  # Имя выходного CSV-файла

    save_known_faces_to_csv(known_faces_dir, csv_file)

if __name__ == "__main__":
    main()
