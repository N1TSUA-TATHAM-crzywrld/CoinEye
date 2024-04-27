from ast import While
from ctypes import resize
import cv2
import sys
import numpy as np
import time
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import datetime

#Continuous_10-19_
def build_model(valid):
    net = cv2.dnn.readNet("/home/atatham45/ChangeCounter/3-19-23_CoinEye.onnx")
    if valid:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.8

def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds

def load_capture(img):
        final_img = cv2.imread(img)
        final_capture = cv2.resize(final_img,(640, 640))
        return final_capture

def load_classes():
    class_list = []
    with open("/home/atatham45/ChangeCounter/classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()

def finalize_detections(input_image, output_data):
    class_ids = []
    confidences = []
    #pre_boxes = []
    boxes = []

    rows = output_data.shape[0]
    image_width, image_height, _ = input_image.shape
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.70:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > 0.5):
                confidences.append(confidence)
                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])
    return result_class_ids, result_confidences, result_boxes

def format_frame(frames):
        row, col, _ = frames.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frames
        return frames

colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

valid = len(sys.argv) > 1 and sys.argv[1] == "cuda"
net = build_model(valid)
capture = load_capture("/home/atatham45/ChangeCounter/static/model_viewer.png")
frames = capture
inputImage = format_frame(frames)
outs = detect(inputImage, net)
class_ids, confidences, boxes = finalize_detections(inputImage, outs[0])



app = Flask(__name__)
app.config['SECRET_KEY'] = 'gettheluuuudes'
bootstrap = Bootstrap(app)

@app.route('/', methods=['GET'])
def Home():
    return render_template('main_app.html')

@app.route('/', methods=['POST'])
def Final_Predictions():

    img_file = request.files['img_file']
    img_path = "./static/" + img_file.filename
    img_file.save(img_path)

    capture = load_capture(img_path)
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
    valid = len(sys.argv) > 1 and sys.argv[1] == "cuda"
    net = build_model(valid)
    class_list = load_classes()
    frames = capture
    inputImage = format_frame(frames)
    outs = detect(inputImage, net)

    detection_amount = []
    class_ids, confidences, boxes = finalize_detections(inputImage, outs[0])
    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
        color = colors[int(classid) % len(colors)]
        cv2.rectangle(frames, box, color, 1)
        cv2.rectangle(frames, (box[0], box[1] - 10), (box[0] + box[2], box[1]), color, -1)
        cv2.putText(frames, class_list[classid], (box[0], box[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, .3, (255,255,255))
        y = list(confidences)
        detection_amount.append(class_list[classid])
        inferance = zip(y, detection_amount)

    quarter = 0
    dime = 0
    nickel = 0
    penny = 0

    for coin in detection_amount:
        if coin == 'Penny':
            penny += 0.01
            continue
        if coin == 'Dime':
            dime += 0.10
            continue
        if coin == 'Nickel':
            nickel += 0.05
            continue
        if coin == 'Quarter':
            quarter += 0.25
            continue

    img_path2 = "./static/images/" + img_file.filename
    cv2.imwrite(img_path2, frames)
    #img_file.save(img_path2)
    image_results =  (img_path2)
    sum = penny + nickel + dime + quarter
    arr = list(str(sum))
    MaxLength = (arr)
    output = (MaxLength[0:5])
    FinalSum = ("".join(output))
    #FinalSum = len(detection_amount)




    return render_template('main_app.html', output = FinalSum, img_results = image_results)

@app.route('/contact', methods=['GET'])
def Contact():
    return render_template('Contact.html')

#if __name__ == '__main__':
    #app.run(host='0.0.0.0', debug=True)
