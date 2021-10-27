import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pickle
import cv2
import imutils


from detect_align import detect_face, preprocess_face
from facenet import *
from connect_db import load_info
from verification import verify


def initialize_model():

    global facenet_model
    facenet_model=loadModel()

    global liveness_model
    liveness_model = tf.keras.models.load_model('models/liveness.model')

    global le
    le = pickle.loads(open('models/label_encoder.pickle', 'rb').read())
    """with open('l_e.pickle', 'wb') as file:
        file.write(pickle.dumps(le))"""



def recognition_liveness(ID,confidence=0.5):

    if not "facenet_model" in globals():

        initialize_model()

    name, embedding = load_info(ID)

    capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    while (capture.isOpened):

        ret, frame = capture.read()
        frame = imutils.resize(frame,width=800)

        if ret == False:
            break

        #cv2.rectangle(frame, (200, 50), (600, 450), (0, 0, 255), 3)
        cv2.imshow(f"{ID}", frame)



        key = cv2.waitKey(33)

        if key == 26:

            face = detect_face(frame, confidence)


            # some error occur here if my face is out of frame and comeback in the frame
            face_to_recog = face
            face = cv2.resize(face, (32, 32))

            face = face.astype('float') / 255.0
            face = tf.keras.preprocessing.image.img_to_array(face)
            face = np.expand_dims(face, axis=0)


            preds = liveness_model.predict(face)[0]
            j = np.argmax(preds)
            label_name = le.classes_[j]  # get label of predicted class
            label = f'{label_name}: {preds[j]:.4f}'
            print(label)

            if label_name == 'real' :
                target_img = preprocess_face(face_to_recog)
                target = facenet_model.predict(target_img)[0]
                result, distance = verify(embedding, target)
                label = f'[{ID}] {name}: {result}({distance})'
                print(label)

        elif key == 27:  # esc
            break


    capture.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':

    ID='BBB'
    recognition_liveness(ID)