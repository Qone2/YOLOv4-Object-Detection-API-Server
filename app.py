import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import time
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
import json
import requests

framework = 'tf'
weights_path = './checkpoints/yolov4-416'
size = 416
tiny = False
model = 'yolov4'
output_path = './detections/'
iou = 0.45
score = 0.25


class Flag:
    tiny = tiny
    model = model
    
    
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
FLAGS = Flag
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
input_size = size

# load model
if framework == 'tflite':
    interpreter = tf.lite.Interpreter(model_path=weights_path)
else:
    saved_model_loaded = tf.saved_model.load(weights_path, tags=[tag_constants.SERVING])

# Initialize Flask application
app = Flask(__name__)
print("loaded")


# API that returns JSON with classes found in images
@app.route('/detections/by-image-files', methods=['POST'])
def get_detections_by_image_files():
    images = request.files.getlist("images")
    image_path_list = []
    for image in images:
        image_name = image.filename
        image_path_list.append("./temp/" + image_name)
        image.save(os.path.join(os.getcwd(), "temp/", image_name))

    # create list for final response
    response = []

    # loop through images in list and run Yolov4 model on each
    for count, image_path in enumerate(image_path_list):
        # create list of responses for current image
        responses = []
        try:
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            image_data = cv2.resize(original_image, (input_size, input_size))
            image_data = image_data / 255.
        except cv2.error:
            # remove temporary images
            for name in image_path_list:
                os.remove(name)
            abort(404, "it is not an image file or image file is an unsupported format. try jpg or png")
        except Exception as e:
            # remove temporary images
            for name in image_path_list:
                os.remove(name)
            print(e.__class__)
            print(e)
            abort(500)

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        if framework == 'tflite':
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
            interpreter.set_tensor(input_details[0]['index'], images_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if model == 'yolov3' and tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            t1 = time.time()
            infer = saved_model_loaded.signatures['serving_default']
            batch_data = tf.constant(images_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
            t2 = time.time()
            print('time: {}'.format(t2 - t1))

        t1 = time.time()
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )
        t2 = time.time()
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        print('time: {}'.format(t2 - t1))
        for i in range(valid_detections[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))
            responses.append({
                "class": class_names[int(classes[0][i])],
                "confidence": float("{0:.2f}".format(np.array(scores[0][i]) * 100)),
                "box": np.array(boxes[0][i]).tolist()
            })
        response.append({
            "image": image_path_list[count][7:],
            "detections": responses
        })
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to allow detections for only people)
        # allowed_classes = ['person']

        image = utils.draw_bbox(original_image, pred_bbox, allowed_classes=allowed_classes)

        image = Image.fromarray(image.astype(np.uint8))

        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path + 'detection' + str(count) + '.png', image)

    # remove temporary images
    for name in image_path_list:
        os.remove(name)
    try:
        return Response(response=json.dumps({"response": response}), mimetype="application/json")
    except FileNotFoundError:
        abort(404)


# API that returns image with detections on it
@app.route('/image/by-image-file', methods=['POST'])
def get_image_by_image_file():
    image = request.files["images"]
    image_path = "./temp/" + image.filename
    image.save(os.path.join(os.getcwd(), image_path[2:]))

    try:
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.
    except cv2.error:
        # remove temporary image
        os.remove(image_path)
        abort(404, "it is not an image file or image file is an unsupported format. try jpg or png")
    except Exception as e:
        # remove temporary image
        os.remove(image_path)
        print(e.__class__)
        print(e)
        abort(500)

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    if framework == 'tflite':
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
        interpreter.set_tensor(input_details[0]['index'], images_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        if model == 'yolov3' and tiny == True:
            boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                            input_shape=tf.constant([input_size, input_size]))
        else:
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                            input_shape=tf.constant([input_size, input_size]))
    else:
        t1 = time.time()
        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        t2 = time.time()
        print('time: {}'.format(t2 - t1))

    t1 = time.time()
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score
    )
    t2 = time.time()
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    print('time: {}'.format(t2 - t1))
    for i in range(valid_detections[0]):
        print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                    np.array(scores[0][i]),
                                    np.array(boxes[0][i])))

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())

    # custom allowed classes (uncomment line below to allow detections for only people)
    # allowed_classes = ['person']

    image = utils.draw_bbox(original_image, pred_bbox, allowed_classes=allowed_classes)

    image = Image.fromarray(image.astype(np.uint8))

    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path + 'detection' + '.png', image)

    # prepare image for response
    _, img_encoded = cv2.imencode('.png', image)
    response = img_encoded.tostring()

    # remove temporary image
    os.remove(image_path)

    try:
        return Response(response=response, status=200, mimetype='image/png')
    except FileNotFoundError:
        abort(404)


# API that returns JSON with classes found in images from url list
@app.route('/detections/by-url-list', methods=['POST'])
def get_detections_by_url_list():
    image_urls = request.get_json()["images"]
    raw_img_list = []
    if not isinstance(image_urls, list):
        abort(400, "can't find image list")
    image_names = []
    custom_headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    }
    for i, image_url in enumerate(image_urls):
        image_name = "Image" + str(i + 1)
        image_names.append(image_name)
        try:
            resp = requests.get(image_url, headers=custom_headers)
            img_raw = np.asarray(bytearray(resp.content), dtype="uint8")
            img_raw = cv2.imdecode(img_raw, cv2.IMREAD_COLOR)
        except cv2.error:
            abort(404, "it is not image url or that image is an unsupported format. try jpg or png")
        except requests.exceptions.MissingSchema:
            abort(400, "it is not url form")
        except Exception as e:
            print(e.__class__)
            print(e)
            abort(500)
        raw_img_list.append(img_raw)

    # create list for final response
    response = []

    # loop through images in list and run Yolov4 model on each
    for count, raw_image in enumerate(raw_img_list):
        # create list of responses for current image
        responses = []

        original_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        if framework == 'tflite':
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
            interpreter.set_tensor(input_details[0]['index'], images_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if model == 'yolov3' and tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            t1 = time.time()
            infer = saved_model_loaded.signatures['serving_default']
            batch_data = tf.constant(images_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
            t2 = time.time()
            print('time: {}'.format(t2 - t1))

        t1 = time.time()
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )
        t2 = time.time()
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        print('time: {}'.format(t2 - t1))
        for i in range(valid_detections[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))
            responses.append({
                "class": class_names[int(classes[0][i])],
                "confidence": float("{0:.2f}".format(np.array(scores[0][i]) * 100)),
                "box": np.array(boxes[0][i]).tolist()
            })
        response.append({
            "image": image_names[count],
            "detections": responses
        })
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to allow detections for only people)
        # allowed_classes = ['person']

        image = utils.draw_bbox(original_image, pred_bbox, allowed_classes=allowed_classes)

        image = Image.fromarray(image.astype(np.uint8))

        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path + 'detection' + str(count) + '.png', image)

    try:
        return Response(response=json.dumps({"response": response}), mimetype="application/json")
    except FileNotFoundError:
        abort(404)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
