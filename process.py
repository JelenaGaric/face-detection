# import libraries here
import matplotlib
from joblib import dump, load
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
from imutils import face_utils

matplotlib.rcParams['figure.figsize'] = 16,12
import matplotlib
import os
import numpy as np
import cv2 # OpenCV
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import dlib
import cv2

            ####### Funkcije ########

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def display_image(image):
    plt.imshow(image, 'gray')

            # transformisemo u oblik pogodan za scikit-learn
def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))


def train_or_load_facial_expression_recognition_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    ##### Ucitavanje postojece istrenirane neuronske mreze ako ne postoji trenira novu #####

    clf_svm = load('clf_svm.joblib')

    ######### Ako postoji utrenirana mreza! #########

    if clf_svm != None:
        model = clf_svm
        return model


    ######### Ako ne postoji utrenirana mreza! #########
    print("Trening mreze zapoceo.")
    # inicijalizacija dlib detektora (HOG)
    detector = dlib.get_frontal_face_detector()
    # ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # ulazi
    ulazi = []

    # izlazi
    izlazi = []

    for izlaz in train_image_labels:
        izlazi.append(izlaz)

    for img in train_image_paths:

        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detekcija svih lica na grayscale slici
        rects = detector(gray, 1)

        # iteriramo kroz sve detekcije korak 1.
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            # odredjivanje kljucnih tacaka - korak 2
            shape = predictor(gray, rect)
            # shape predstavlja 68 koordinata
            shape = face_utils.shape_to_np(shape)  # konverzija u NumPy niz
            ulazi.append(shape)
            #print("Dimenzije prediktor matrice: {0}".format(shape.shape))  # 68 tacaka (x,y)
            #print("Prva 3 elementa matrice")
            #print(shape[:3])

            # konvertovanje pravougaonika u bounding box koorinate
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # crtanje pravougaonika oko detektovanog lica
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # ispis rednog broja detektovanog lica
            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # crtanje kljucnih tacaka
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            plt.imshow(image)

    x_train = np.array(ulazi)
    y_train = np.array(izlazi)

    x_train = reshape_data(x_train)

    clf_svm = SVC(kernel='linear', probability=True)
    clf_svm = clf_svm.fit(x_train, y_train)

    dump(clf_svm, 'clf_svm.joblib')
    print("Trening mreze zavrsio.")

    model = clf_svm

    return model


def extract_facial_expression_from_image(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje ekspresije lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati ekspresiju.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <String>  Naziv prediktovane klase (moguce vrednosti su: 'anger', 'contempt', 'disgust', 'happiness', 'neutral', 'sadness', 'surprise'
    """

    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)

    # inicijalizacija dlib detektora (HOG)
    detector = dlib.get_frontal_face_detector()
    # ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    print("Predikcija:")

    # ulazi
    ulazi = []
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detekcija svih lica na grayscale slici
    rects = detector(gray, 1)

    # iteriramo kroz sve detekcije korak 1.
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        # odredjivanje kljucnih tacaka - korak 2
        shape = predictor(gray, rect)
        # shape predstavlja 68 koordinata
        shape = face_utils.shape_to_np(shape)  # konverzija u NumPy niz
        #print("Dimenzije prediktor matrice: {0}".format(shape.shape))  # 68 tacaka (x,y)
        #print("Prva 3 elementa matrice")
        #print(shape[:3])
        ulazi.append(shape)

        # konvertovanje pravougaonika u bounding box koorinate
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # crtanje pravougaonika oko detektovanog lica
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ispis rednog broja detektovanog lica
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # crtanje kljucnih tacaka
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    x_test = np.array(ulazi)
    x_test = reshape_data(x_test)

    try:
        y_test_pred = trained_model.predict(x_test)
        print(y_test_pred)       ########## ['anger'] ##########
        y_test_pred = str(y_test_pred)
        facial_expression = y_test_pred[2:-2]
        print(y_test_pred[2:-2])
    except:
        facial_expression = ""

    return facial_expression
