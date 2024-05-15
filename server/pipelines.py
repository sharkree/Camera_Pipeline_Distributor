import threading
import time

import cv2
import numpy as np

import tensorflow as tf

from helpers import Circular_Array, swap


class Pipeline:
    is_active = True
    name = ''
    idx = -1

    loop_times = Circular_Array(10)

    x_crop = [0, 1]
    y_crop = [0, 1]

    def __init__(self, name, idx):
        self.name = name
        self.idx = idx

    def process_frame(self, img):
        pass

    def set_x_crop_range(self, vals):
        self.x_crop = vals

    def set_y_crop_range(self, vals):
        self.y_crop_min = vals

    def set_active(self, is_active=True):
        self.is_active = is_active

    def set_name(self, name):
        self.name = name

    def get_average_loop_time(self):
        return self.loop_times.get_average()


class EmptyPipeline(Pipeline):
    color = 255

    def __init__(self, name, idx):
        super(EmptyPipeline, self).__init__(name, idx)

    def set_color(self, color):
        self.color = color

    def process_frame(self, img):
        start = time.time()

        img = cv2.rectangle(img, (0, 0), (300, 300), (0, 0, self.color), 2)

        end = time.time()

        self.loop_times.add(end - start)

        return img


class NeuralPipeline(Pipeline):
    confidence_threshold = 1.

    classes = []
    model = ''

    sem = threading.Semaphore()

    def __init__(self, name, idx, classes, model, confidence_threshold):
        super(NeuralPipeline, self).__init__(name, idx)

        self.classes = classes
        self.model = model
        self.confidence_threshold = confidence_threshold

        self.interpreter = tf.lite.Interpreter(model_path=self.model)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def set_confidence(self, conf):
        self.confidence_threshold = conf

    def process_frame(self, img):
        self.sem.acquire()

        img_data = self.preprocess(img)

        self.interpreter.set_tensor(self.input_details[0]['index'], img_data)
        self.interpreter.invoke()

        confidences = self.interpreter.get_tensor(self.output_details[0]['index'])
        boxes = self.interpreter.get_tensor(self.output_details[1]['index'])

        confidences = np.squeeze(confidences)
        boxes = np.squeeze(boxes)

        img = self.postprocess(img, confidences, boxes)

        self.sem.release()

        return img

    def preprocess(self, img):
        img = cv2.resize(img, (300, 300))
        return np.expand_dims(np.array(img), axis=0).astype(np.float32)

    def postprocess(self, img, confidences, boxes):
        rows = len(confidences)

        for i in range(rows):
            if confidences[i] >= self.confidence_threshold:
                topx, topy, botx, boty = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

                topx, topy = swap(topx, topy)
                botx, boty = swap(botx, boty)

                topx *= 640
                botx *= 640
                topy *= 480
                boty *= 480

                img = cv2.rectangle(img, (int(topx), int(topy)), (int(botx), int(boty)), (0, 0, 255), 2)

        return img
