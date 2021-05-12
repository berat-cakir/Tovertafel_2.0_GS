import numpy as np
import cv2
from imutils import face_utils
import dlib
from statistics import mode
from keras.models import load_model

###############################################################################
# Utils

def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


###############################################################################
# Emotion model class

class EmotionModel():

    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()

        emotion_model_path = 'emotion_model_weights/fer2013_mini_XCEPTION.110-0.65.hdf5'
        self.emotion_labels = get_labels('fer2013')
        self.emotion_classifier = load_model(emotion_model_path, compile=False)

        # hyper-parameters for bounding boxes shape
        self.frame_window = 10
        self.emotion_offsets = (20, 40)
        # getting input model shapes for inference
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
        # starting lists for calculating modes
        self.emotion_window = []

        self.annotations = {'bboxes': [], 'emotions_text': [], 'emotion_probabilities' : [], 'face_landmarks': [], 'colors': []}


    def _reset_annotations(self):
        self.annotations = {'bboxes': [], 'emotions_text': [], 'emotion_probabilities' : [], 'face_landmarks': [], 'colors': []}

    def _detect_faces(self, gray_frame):
        faces = self.face_detector(gray_frame, 0)
        return faces

    def predict_emotion(self, gray_frame):
        faces = self._detect_faces(gray_frame.copy())

        for rect in faces:
            face_coordinates = face_utils.rect_to_bb(rect)
            x1, x2, y1, y2 = apply_offsets(face_coordinates, self.emotion_offsets)
            gray_face = gray_frame[y1:y2, x1:x2].copy()

            try:
                gray_face = cv2.resize(gray_face, (self.emotion_target_size))
            except:
                continue

            #gray_face = cv2.resize(gray_face, (self.emotion_target_size))
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            emotion_prediction = self.emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = self.emotion_labels[emotion_label_arg]
            # self.emotion_window.append(emotion_text)

            # if len(self.emotion_window) > self.frame_window:
            #     self.emotion_window.pop(0)
            # try:
            #     emotion_mode = mode(self.emotion_window)
            # except:
            #     continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()
            self.annotations['colors'].append(color)

            # draw_bounding_box(face_coordinates, gray_frame, color)
            self.annotations['bboxes'].append(tuple(face_coordinates))

            # draw_text(face_coordinates, gray_frame, emotion_mode, color, 0, -45, 1, 1)
            self.annotations['emotions_text'].append(emotion_text)
            self.annotations['emotion_probabilities'].append(tuple(emotion_prediction))

        return self.annotations['emotions_text'], self.annotations['emotion_probabilities']



    def draw_annotations(self, overlay_image):

        bboxes = self.annotations['bboxes']
        for i, face_coordinates in enumerate(bboxes):
            x, y, w, h = face_coordinates
            color = self.annotations['colors'][i]
            overlay_image = cv2.rectangle(overlay_image, (x, y), (x + w, y + h), color, 2)
            emotion_text = self.annotations['emotions_text'][i]
            overlay_image = cv2.putText(overlay_image, emotion_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)

        self._reset_annotations()
        return overlay_image