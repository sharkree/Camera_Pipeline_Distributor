import cv2
import json
from pipelines import *
from helpers import to_idx

return_idx = 1

processors = [None] * 5
processors[0] = EmptyPipeline("test", 0)
processors[1] = NeuralPipeline("cooker", 1, [0, 1], "32bit.tflite", 0.5)

def process_image(img):
    res = img

    for i in range(5):
        if processors[i] is not None:
            if return_idx == i:
                res = processors[i].process_frame(img.copy())
            else:
                processors[i].process_frame(img.copy())

    return res
