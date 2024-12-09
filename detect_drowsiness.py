import cv2                                            # Importation de la bibliothèque OpenCV pour la capture vidéo et la détection d'objets
import numpy as np                                    # Importation de la bibliothèque NumPy pour la gestion des tableaux
from keras.models import load_model                   # Importation de la fonction pour charger un modèle pré-entraîné
from keras.preprocessing.image import img_to_array    # Importation de la fonction pour convertir une image en tableau
import pygame                                         # Importation de Pygame pour jouer des sons
from threading import Thread                          # Importation de la classe Thread pour exécuter des tâches simultanées

def start_alarm(sound):
    """Jouer le son de l'alarme en utilisant pygame"""
    pygame.mixer.init()             # Initialiser le module Pygame pour le son
    pygame.mixer.music.load(sound)  # Charger le fichier sonore
    pygame.mixer.music.play()       # Lancer la lecture du son

# Chargement des modèles et des cascades de détection
classes = ['Closed', 'Open']   # Définir les classes d'yeux : fermés et ouverts
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")    # Charger la cascade pour la détection des visages
left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")    # Charger la cascade pour l'œil gauche
right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")  # Charger la cascade pour l'œil droit
mouth_cascade = cv2.CascadeClassifier("data/haarcascade_mcs_mouth.xml")             # Charger la cascade pour la détection de la bouche

cap = cv2.VideoCapture(0)                  # Ouvrir la caméra pour capturer des images en temps réel 
model = load_model("drowiness_new7.h5")    # Charger le modèle de détection de somnolence pré-entraîné
count = 0                                  # Compteur pour compter les frames avec les yeux fermés
alarm_on = False                           # Indicateur pour savoir si l'alarme est activée
alarm_sound = "data/alarm.mp3"             # Fichier sonore de l'alarme
yowning_sound = "data/msg.mp3"             # Fichier sonore pour le bâillement
status1 = ''                               # Statut de l'œil gauche (fermé ou ouvert)
status2 = ''                               # Statut de l'œil droit (fermé ou ouvert)
yawn_count = 0                             # Compteur de bâillements détectés
yawn_alerted = False                       # Indicateur pour éviter des alertes répétées pour le bâillement
drowsiness_alerted = False                  # Indicateur pour éviter des alertes répétées de somnolence

while True:
    _, frame = cap.read()                                 # Capturer une image depuis la caméra
    height, width = frame.shape[:2]                       # Récupérer la hauteur et la largeur de l'image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        # Convertir l'image en niveaux de gris pour la détection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)   # Détecter les visages dans l'image
    
    for (x, y, w, h) in faces:                                              # Pour chaque visage détecté
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)        # Dessiner un rectangle autour du visage
        roi_gray = gray[y:y + h, x:x + w]                                   # Extraire la région d'intérêt (ROI) en niveaux de gris pour les yeux
        roi_color = frame[y:y + h, x:x + w]                                 # Extraire la ROI en couleur pour les yeux

        left_eye = left_eye_cascade.detectMultiScale(roi_gray)          # Détecter l'œil gauche dans la région d'intérêt            
        right_eye = right_eye_cascade.detectMultiScale(roi_gray)        # Détecter l'œil droit dans la région d'intérêt
        for (x1, y1, w1, h1) in left_eye:                               # Si un œil gauche est détecté
            cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)     # Dessiner un rectangle autour de l'œil gauche
            eye1 = roi_color[y1:y1 + h1, x1:x1 + w1]                                   # Extraire l'œil gauche
            eye1 = cv2.resize(eye1, (145, 145))                                        # Redimensionner l'œil à la taille attendue par le modèle
            eye1 = eye1.astype('float') / 255.0                                        # Normaliser les pixels entre 0 et 1
            eye1 = img_to_array(eye1)                                                  # Convertir l'image de l'œil en tableau numpy
            eye1 = np.expand_dims(eye1, axis=0)                                        # Ajouter une dimension pour que le modèle puisse prédire
            pred1 = model.predict(eye1)                                                # Prédiction de l'état de l'œil gauche
            status1 = np.argmax(pred1)                                                 # Obtenir le statut (0 : fermé, 1 : ouvert)
            break

        for (x2, y2, w2, h2) in right_eye:   # Si l'œil droit est détecté
            cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1) # Dessine un rectangle autour de l'œil droit
            eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]                               # Récupère l'image de l'œil droit
            eye2 = cv2.resize(eye2, (145, 145))                                    # Redimensionne l'œil à 145x145 pixels
            eye2 = eye2.astype('float') / 255.0                                    # Normalise les pixels de l'œil entre 0 et 1
            eye2 = img_to_array(eye2)                                              # Convertit l'image de l'œil en un tableau numpy
            eye2 = np.expand_dims(eye2, axis=0)                                    # Ajoute une dimension pour que le modèle puisse faire la prédiction
            pred2 = model.predict(eye2)                                            # Effectue la prédiction pour l'œil droit
            status2 = np.argmax(pred2)                                             # Récupère l'état de l'œil (0 = fermé, 1 = ouvert)
            break
  
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)                 # Détecte la bouche dans la ROI en niveaux de gris
        for (mx, my, mw, mh) in mouth:   # Si une bouche est détectée
            yawn_ratio = mh / mw         # Calcule le ratio hauteur/largeur de la bouche
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 1)   # Dessine un rectangle autour de la bouche
            if yawn_ratio > 0.6:     # Si la bouche est ouverte au-delà d'un seuil (bâillement)
                yawn_count += 1      # Incrémente le compteur de bâillements
                cv2.putText(frame, "Yawning Detected!", (10, height - 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)    # Affiche un message de bâillement
                if not yawn_alerted:       # Si l'alerte de bâillement n'a pas encore été donnée
                    yawn_alerted = True    # Lance un thread pour jouer le son du bâillement
                    t = Thread(target=start_alarm, args=(yowning_sound,))  
                    t.daemon = True
                    t.start()
                break
            else:
                yawn_count = 0         # Réinitialise le compteur si aucun bâillement n'est détecté
                yawn_alerted = False   # Réinitialise l'alerte de bâillement

            # Si la bouche est ouverte, afficher un message
            if mh > 40:    # Si la hauteur de la bouche est supérieure à 40 pixels, cela indique que la bouche est ouverte
                 # Affiche "Bouche ouverte" sur l'image à la position (10, height - 100), en rouge (0, 0, 255), avec une taille de police de 1 et une épaisseur de 2
                cv2.putText(frame, "Mouth Open", (10, height - 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)  

        
        if status1 == 2 and status2 == 2:    # Si les deux yeux sont fermés pendant plusieurs frames consécutives
            count += 1    # Incrémente le compteur de frames avec les yeux fermés
            cv2.putText(frame, "Eyes Closed, Frame count: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)   # Affiche le compteur
            if count >= 5 and not drowsiness_alerted:   # Si les yeux sont fermés pendant 5 frames consécutives
                cv2.putText(frame, "Drowsiness Alert!!!", (100, height - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)   # Affiche l'alerte de somnolence
                if not alarm_on:   # Si l'alarme n'a pas encore été activée
                    alarm_on = True
                    t = Thread(target=start_alarm, args=(alarm_sound,))    # Lance un thread pour jouer l'alarme de somnolence
                    t.daemon = True
                    t.start()
                drowsiness_alerted = True   # Marque l'alerte de somnolence comme activée
        else:
            cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)  # Affiche "Eyes Open" en vert si les yeux sont ouverts.
            count = 0                    # Réinitialise le compteur des yeux fermés.
            drowsiness_alerted = False    # Réinitialise l'alerte de somnolence.

        # Réinitialise l'alerte de bâillement si la bouche n'est plus ouverte
        if not yawn_alerted:
            yawn_count = 0 # Réinitialise le compteur de bâillements.

    cv2.imshow("Drowsiness and Yawning Detector", frame)  # Affiche l'image avec les annotations (yeux, bouche, alertes)

    if cv2.waitKey(1) & 0xFF == ord('q'):   # Si l'utilisateur appuie sur 'q', quitte la boucle
        break

cap.release()             # Libère la caméra pour qu'elle puisse être utilisée par d'autres applications
cv2.destroyAllWindows()   # Ferme toutes les fenêtres OpenCV ouvertes