import cv2
import numpy as np
from keras.models import model_from_json


class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Surprise", "Sad", "Happy", "Neutral"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        print("Model loaded from disk")
        self.loaded_model.summary()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]


if __name__ == '__main__':
    pass

rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('../xml/haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_DUPLEX


def __get_data__():
    _, fr = rgb.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)
    return faces, fr, gray


def start_app(cnn):
    ix = 0
    while True:
        ix += 1
        faces, fr, gray_fr = __get_data__()
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]

            roi = cv2.resize(fc, (48, 48))
            pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            # Generating text and rectangle around the face
            cv2.putText(fr, pred, (x, y), font, 1, (25, 33, 112), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (16, 237, 233), 1)

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Filter', fr)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Providing the model_json and h5 file to recognize the data
    model = FacialExpressionModel("../model/trainedModel.json", "../model/trainModel.h5")
    start_app(model)
