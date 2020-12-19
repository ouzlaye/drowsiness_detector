
# import de packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np 
import playsound 
import argparse
import imutils
import time
import cv2
import dlib

path= 'alarm.mp3'

def play_alarm(path):
    playsound.playsound(path)


def eye_aspect_ratio(eye):
    A= dist.euclidean(eye[1], eye[5])
    B= dist.euclidean(eye[2], eye[4])

    #calcul de la distance euclidienne entre les coordonnees horizontales de eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    #calculer la taille de l'aspect des yeux 
    eye_ratio = (A + B) / (2.0 * C)

    return eye_ratio


# construction du parseur d'arguments
app = argparse.ArgumentParser()
app.add_argument("-p", "--shape-predictor", required=True,
    help="chemin vers le facial landmarks predictor")
app.add_argument("-a", "--alarm", type=str, default=0,
    help="chemin vers le fichier son d'alarme")
app.add_argument("-w", "--webcam", type=int, default=0,
    help="index du webcam")
args = vars(app.parse_args())

eye_ar = 0.3
eye_ar_consec_frame = 48

#initilaisation du compteur 
COUNTER = 0
ALARM_ON = False

#initialise dlib face detector (HOG-based) 
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart , lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# commencer le streaming 
print("[INFO] commencement du streaming video...")
vs = VideoStream(src= args["webcam"]).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detection de la face 
    rect = detector(gray, 0)

    for rec in rect:
        shape = predictor(gray, rec)
        shape= face_utils.shape_to_np(shape)
        #extraction des left et right coordinate
        leftEye = shape[lStart: lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR  = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
          
        # la moyenne des l'aspect ratio
        ear = (leftEAR + rightEAR) / 2.0

        #calculer le convex pour le right et left eye pour visulaiser les contour de l'oiem
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0),1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


        # # vérifier si le rapport hauteur / largeur des yeux est inférieur au clignotement
        # seuil, et si c'est le cas, incrémenter le compteur de clignotement
        if ear < eye_ar:
            COUNTER+=1

            # si les yeix sont fermés pendant un temps suffisant
            if COUNTER >= eye_ar_consec_frame:
                #si l'alarme est eteint alors l'allumer
                if not ALARM_ON:
                    ALARM_ON = True

                    if args["alarm"] != "":
                        t = Thread(target=play_alarm, args=("alarm.mp3",))
                        t.deamon = True
                        t.start()


                #affiche l'alarme dans la fenetre
                
                cv2.putText(frame, "ALERTE SOMNOLENCE!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        #remettre le compteur zero si la durée de fermeture des yeux est insuffisante
        else:
            COUNTER=0
            ALARM_ON= False

        cv2.putText(frame, "ear: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)


    #afficher la video
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

        #arreter da la boucle avec la touche q
    if key == ord("q"):
         break

cv2.destroyAllWindows()
vs.stop



    
