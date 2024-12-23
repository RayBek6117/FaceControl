import cv2
import mediapipe as mp 
import time
import keyboard

# Получение изображения с камеры. 0 - нулевой индекс камеры
# Можно указать видео "wdir/video.mp4"
cap = cv2.VideoCapture(0)

# Создаем класс для определения наличия рук в кадре
mpHands = mp.solutions.hands 
hands = mpHands.Hands()

# Создаем класс для изображения маркеров рук
mpDraw = mp.solutions.drawing_utils

pTime = 0 # previous time
cTime = 0 # current time

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    # Отображение рук в кадре
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
           # вывод координат landmarks
            for id, lm in enumerate(handLms.landmark):
               print(id, lm)
               # Координаты Для ручного отображения landmarks
               h, w, c = img.shape
               cx, cy = int(lm.x * w), int(lm.y * h)
               if id == 8:
                   cv2.circle(img, (cx, cy), 20, (255, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) 

    # Отображение fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 128, 0), 5)

    # Запуск приложения
    cv2.imshow("Image", img)
    cv2.waitKey(20)

    # Выход из приложения при нажатии пробела 
    if keyboard.is_pressed("space"):
        break

