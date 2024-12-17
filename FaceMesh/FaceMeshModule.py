import cv2             
import mediapipe as mp 
import time                
import keyboard       

class FaceMeshDetector():
    def __init__(self, static_mode = False, maxFaces = 2, refineLms = False,
                 minDetectionCon = 0.5, minTrackCon = 0.5):
        self.static_mode = static_mode
        self.maxFaces = maxFaces
        self.refineLms = refineLms
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_mode, self.maxFaces,
                                                 self.refineLms, self.minDetectionCon,
                                                 self.minTrackCon)  
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = 2, circle_radius = 1)

    def findFaceMesh(self, img, draw = True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []
        if self.results.multi_face_landmarks:                        
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                        self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape                      
                    x, y = int(lm.x * iw), int(lm.y * ih)       
                    face.append([x, y])
                faces.append(face)

        return img, faces

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0 
    cTime = 0 

    detector = FaceMeshDetector()

    while True:
        success, img = cap.read() 

        img, faces = detector.findFaceMesh(img)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 128, 0), 5)

        cv2.imshow("Image", img)
        cv2.waitKey(20)

        if keyboard.is_pressed("space"):
            break

if __name__ == "__main__":
    main()