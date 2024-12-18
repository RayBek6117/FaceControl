# Face Control

Данный проект использует библиотеку face_recognition и OpenCV для выполнения распознавания лиц в реальном времени с помощью веб-камеры. Программа сравнивает лица, полученные с камеры, с уже закодированными лицами из папки img, и в случае совпадения выводит имя человека. Если соответствия не найдено, отобразится надпись "Unknown".

## Как это работает

В начале выполнения программы все изображения в папке img обрабатываются и кодируются. Каждое обнаруженное лицо преобразуется в числовой вектор (кодировку), уникально описывающий особенности лица.

Изображение с веб-камеры захватывается покадрово. Каждый кадр анализируется на наличие лиц, и найденные лица также кодируются.

Кодировка обнаруженного лица с камеры сравнивается с ранее сохраненными кодировками из папки img. Если расстояние между кодировкой нового лица и уже известной кодировкой достаточно мало, программа считает, что это один и тот же человек, и показывает его имя. Если подходящей пары нет, лицо считается "Unknown".

Вывод результата: Имя совпавшего лица отображается над прямоугольной рамкой вокруг лица на видеопотоке.

## Требования

1. Python 3.6+
2. face_recognition
3. OpenCV
4. Pillow

### Установить необходимые библиотеки можно при помощи pip:

```Terminal
pip install cmake
```
```Terminal
pip install dlib
```
```Terminal
pip install face-recognition
```
```Terminal
pip install opencv-python
```

## Примечание:

Если вы используете macOS или Linux и сталкиваетесь с проблемами при установке face_recognition, убедитесь, что у вас установлен dlib с необходимыми зависимостями. Подробности по установке можно найти в репозитории dlib или установить с помощью пакетного менеджера (например, brew на macOS).

- Windows: https://github.com/z-mahmud22/Dlib_Windows_Python3.x
- macOS:

```Terminal
$ brew install cmake
```

```Terminal
$ pip install cmake
```

```Terminal
$ brew install dlib
```

```Terminal
$ pip install dlib
```

**Или** sudo pip install dlib

```Структура проекта
.
├── img/
│   ├── person1.jpg
│   ├── person2.jpg
│   └── ...
├── FaceRecognitionBase.py
└── README.md
```

Положите изображения знакомых людей в папку img.
Убедитесь, что на каждом изображении присутствует только одно чётко различимое лицо.
**Имя файла (без расширения) будет использоваться как имя человека при его распознавании.**

## Запуск программы

Подготовьте изображения:

_Добавьте одно или несколько изображений людей в папку img._
Например, файл Ivan.jpg будет отображаться как "Ivan", если программа его узнает.
Запустите программу:

```Terminal
python FaceRecognitionBase.py
```

Программа загрузит и закодирует известные лица.
Откроется окно с потоком с вашей веб-камеры.
Если лицо будет распознано, над рамкой вокруг лица отобразится имя.
Неизвестные лица будут помечены как "Unknown".

### Выход из программы:

Нажмите клавишу _q_, чтобы завершить работу программы.

### Возможные проблемы

- Веб-камера не обнаружена:
  Проверьте, что у вас есть рабочая веб-камера и что программа имеет к ней доступ. Если у вас несколько камер, возможно, придется изменить строку cv2.VideoCapture(0) на cv2.VideoCapture(1) или другое число.

- Нет загруженных лиц:
  Если программа пишет "No known faces loaded", убедитесь, что:

- В папку img добавлены изображения лиц.
  Файлы имеют формат .jpg, .jpeg или .png.
  Слабая точность / проблемы с распознаванием:
  Попробуйте добавить несколько изображений одного и того же человека или использовать изображения более высокого качества и с хорошим освещением. Убедитесь, что лицо на изображениях направлено прямо в камеру.
  Можно также изменить размеры кадра или условия освещения.
