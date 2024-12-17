import cv2             
import mediapipe as mp 
import time                
import keyboard       

# Получение изображения с камеры. 0 - нулевой индекс камеры
# Можно указать видео "[Название рабочей папки]/video.mp4"
cap = cv2.VideoCapture(0)

# Переменные для рассчета FPS
pTime = 0 # previous time
cTime = 0 # current time

# Создаем класс для изображения точек/полос на видео
mpDraw = mp.solutions.drawing_utils

# Объект для распознования сетки лица
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 2)  
# У FaceMesh() можно указать уверенность трекинга, 
# количество максимальных лиц в кадре... 

# Указываем режим отображения точек на лице
# circle_radius - радиус точек
# thickness - толщина соединяющих линий
drawSpec = mpDraw.DrawingSpec(thickness = 2, circle_radius = 1)

while True:
    # Создаем новое окно для вывода результатов
    success, img = cap.read() 
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:                        # Если лицо есть в кадре(присутствуют face_landmarks)
        for faceLms in results.multi_face_landmarks:
            # Рисуем на экране landmarks
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec, drawSpec)
            
            # mpFaceMesh.FACEMESH_CONTOURS - соединияет точки на лице
            # faceLms - список точек лица
            # img - изображение, где рисовать
            # drawSpec - параметры отображения

            # Определяем координаты точек на лице
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape                      # Получаем высоту, ширину, центр окна с отображением
                x, y = int(lm.x * iw), int(lm.y * ih)       # Определяем координаты landmark
                if id == 0:                                 # Выбираем конкретный ID landmark'и для вывода в консоль
                    print([id, x, y])



    

    # Рассчет и отображение fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 128, 0), 5)

    # Запуск демонстрации видео
    cv2.imshow("Image", img)
    cv2.waitKey(20)

    # Выход из приложения при нажатии пробела (не работает на macos) 
    if keyboard.is_pressed("space"):
        break