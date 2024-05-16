import time
from ast import literal_eval

import cv2
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from flask_socketio import emit, SocketIO
from flask import Flask, render_template

import processor

app = Flask(__name__, template_folder="templates")
socketio = SocketIO(app)


@socketio.on('image')
def image(data_image):
    data_image += "=" * ((4 - len(data_image) % 4) % 4)

    binaryImageData = base64.b64decode(data_image)
    imageStream = BytesIO(binaryImageData)

    try:
        with Image.open(imageStream) as pimg:
            frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

            frame = processor.process_image(frame)

            imgencode = cv2.imencode('.jpg', frame)[1]

            stringData = base64.b64encode(imgencode).decode('utf-8')
            b64_src = 'data:image/jpg;base64,'
            stringData = b64_src + stringData
            emit('response_back', stringData)

            pass
    except IOError:
        time.sleep(0.05)
        print("An error occurred while trying to open the image.")
        

@socketio.on('get_loop')
def get_loop():
    emit('loop', processor.get_loop_times())



# str contains the json of all the stuff that changed
@socketio.on('change_processing')
def change_processing(json_str):
    data = processor.json.loads(json_str)

    for key, value in data.items():
        if key == 'return_idx':
            processor.return_idx = value
            continue

        if key == 'pipelines':
            for idx, pipeline in data[key].items():
                for key1, value1 in pipeline.items():
                    if key1 == 'type':
                        # create new pipeline and update it
                        continue

                    if key1 == "color":
                        processor.processors[int(idx)].set_color(int(value1))

                    if key1 == "is_active":
                        processor.processors[int(idx)].set_active(bool(value1))
                        continue

                    if key1 == "name":
                        processor.processors[int(idx)].set_name(value1)
                        continue

                    if key1 == "x_crop":
                        processor.processors[int(idx)].set_x_crop_range(literal_eval(value1))
                        continue

                    if key1 == "y_crop":
                        processor.processors[int(idx)].set_y_crop_range(literal_eval(value1))

                    if key1 == "confidence":
                        processor.processors[int(idx)].set_confidence(float(value1))

                    # add remaining for other pipelines


@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    socketio.run(app, debug=True)
