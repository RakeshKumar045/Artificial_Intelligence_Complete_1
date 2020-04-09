import cv2
import numpy as np
import os


# faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32,  minNeighbors=5)  # detectMultiScale returns rectangles


# This module contains all common functions that are called in tester.py file


# Given an image below function returns rectangle for face detected alongwith gray scale image
def faceDetection(test_img):
    print(test_img)
    test_img = test_img.astype('uint8')
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  # convert color image to grayscale
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Load haar classifier
    faces = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)  # detectMultiScale returns rectangles

    return faces, gray_img


# Given a directory below function returns part of gray_img which is face alongwith its label/ID
def labels_for_training_data(directory):
    faces = []
    faceID = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")  # Skipping files that startwith .
                continue

            id = os.path.basename(path)  # fetching subdirectory names
            img_path = os.path.join(path, filename)  # fetching image path
            print("img_path:", img_path)
            print("id:", id)
            test_img = cv2.imread(img_path)  # loading each image one by one
            if test_img is None:
                print("Image not loaded properly")
                continue
            faces_rect, gray_img = faceDetection(
                test_img)  # Calling faceDetection function to return faces detected in particular image
            if len(faces_rect) != 1:
                continue  # Since we are assuming only single person images are being fed to classifier
            (x, y, w, h) = faces_rect[0]
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from grayscale image
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces, faceID


# Below function trains haar classifier and takes faces,faceID returned by previous function as its arguments
def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    # face_recognizer = cv2.face_LBPHFaceRecognizer.create()
    face_recognizer.train(faces, np.array(faceID))

    return face_recognizer


# Below function draws bounding boxes around detected face in image
def draw_rect(test_img, face):
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=5)


# Below function writes name of person for detected label
def put_text(test_img, text, x, y):
    cv2.putText(test_img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 5, (255, 0, 0), 6)

#
#
# # This module takes images  stored in diskand performs face recognition
# test_image = cv2.imread('/Users/rakeshkumar/Desktop/ML_Imp/AI/FaceObjectDetectionl/TestImage/r1.jpg')  # test_img path
# faces_detected, gray_image = faceDetection(test_image)
# print("faces_detected:", faces_detected)
#
# # Uncomment belows lines when running this program first time.Since it svaes training.yml file in directory
# faces1,faceID1=labels_for_training_data('./TrainImage')
# face_recognizer=train_classifier(faces1,faceID1)
# face_recognizer.save('ObjectDetectionModel.yml')
# print("Save Success")
#
#
# # face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# # face_recognizer.read('ObjectDetectionModel.yml')  # use this to load training data for subsequent runs
#
#
#
# name = {0: "Random", 1: "Rakesh"}  # creating dictionary containing names for each label
#
# for face in faces_detected:
#     (x, y, w, h) = face
#     roi_gray = gray_image[y:y + h, x:x + h]
#     label, confidence = face_recognizer.predict(roi_gray)  # predicting the label of given image
#     print("confidence:", confidence)
#     print("label:", label)
#     draw_rect(test_image, face)
#     predicted_name = name[label]
#     if (confidence > 37):  # If confidence less than 37 then don't print predicted face text on screen
#         continue
#     put_text(test_image, predicted_name, x, y)
#
# resized_img = cv2.resize(test_image, (1000, 1000))
# cv2.imshow("face dtecetion tutorial", resized_img)
# cv2.waitKey(0)  # Waits indefinitely until a key is pressed
# cv2.destroyAllWindows()
