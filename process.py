# import libraries here
import math
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import matplotlib
import matplotlib.pyplot as plt
# prikaz vecih slika
matplotlib.rcParams['figure.figsize'] = 16,12

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def display_image(image):
    plt.imshow(image, 'gray')

# transformisemo u oblik pogodan za scikit-learn
def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

def train(train_image_paths, train_image_labels):

    #TODO probati sa SVM i probati sa rastojanjem od tezista i uglom!

    # inicijalizaclija dlib detektora (HOG)
    detector = dlib.get_frontal_face_detector()
    # ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # ulazi
    inputs = []
    # izlazi
    labels = []
    for label in train_image_labels:
        labels.append(label)

    for img_path in train_image_paths:
        img = load_image(img_path)
        gray = img.copy()
        # detekcija svih lica na grayscale slici
        rects = detector(gray, 1)
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
            inputs.append(shape)

            # konvertovanje pravougaonika u bounding box koorinate
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # crtanje pravougaonika oko detektovanog lica
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # ispis rednog broja detektovanog lica
            cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # crtanje kljucnih tacaka
            for (x, y) in shape:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            # display_image(img)
            # plt.show()

    x = np.array(inputs)
    y = np.array(labels)

    x_train = reshape_data(x)

    clf_svm = SVC(kernel='linear', probability=True)
    clf_svm = clf_svm.fit(x_train, y)

    y_train_pred = clf_svm.predict(x_train)

    dump(clf_svm, 'svm.joblib')

    return clf_svm

def train_or_load_facial_expression_recognition_model(train_image_paths, train_image_labels):
    clf_svm = load('svm.joblib')
    if clf_svm == None:
        clf_svm = train(train_image_paths, train_image_labels)

    return clf_svm


def extract_facial_expression_from_image(trained_model, image_path):
    detector = dlib.get_frontal_face_detector()
    # ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # ulazi
    inputs = []
    img = load_image(image_path)
    gray = img.copy()
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)  # konverzija u NumPy niz
        inputs.append(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for (x, y) in shape:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        display_image(img)
        plt.show()


    x_test = np.array(inputs)
    x_test = reshape_data(x_test)

    try:
        clf_svm = load('svm.joblib')
        y_test_pred = clf_svm.predict(x_test)

        if 'neutral' in y_test_pred:
            facial_expression = 'neutral'
        elif 'sadness' in y_test_pred:
            facial_expression = 'sadness'
        elif 'surprise' in y_test_pred:
            facial_expression = 'surprise'
        elif 'happiness' in y_test_pred:
            facial_expression = 'happiness'
        elif 'anger' in y_test_pred:
            facial_expression = 'anger'
        elif 'contempt' in y_test_pred:
            facial_expression = 'contempt'
        elif 'disgust' in y_test_pred:
            facial_expression = 'disgust'
    except:
        facial_expression = ""

    print(facial_expression)
    return facial_expression
