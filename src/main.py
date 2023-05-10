from LogService import LogService
from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO("yolov8n.pt")

def loadVideo(videoName):
    video = cv2.VideoCapture(videoName)
    if video is None:
        raise Exception("Não foi possível carregar o video")
    return video

def draw_boxes(frame, boxes):
    quantity_of_box_nearby = 0
    for i in range(len(boxes)):
        box = boxes[i]
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

        for j in range(i + 1, len(boxes)):
            box1 = boxes[i]
            box2 = boxes[j]
            # to calculate the distance from the objects, use the center of the box
            center_box1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
            center_box2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
            
            distance = np.sqrt((center_box1[0] - center_box2[0])**2 + (center_box1[1] - center_box2[1])**2)
            if distance < 200:
                quantity_of_box_nearby += 1
                cv2.line(frame, (int(center_box1[0]), int(center_box1[1])), (int(center_box2[0]), int(center_box2[1])), (0, 255, 0), 2)
    return quantity_of_box_nearby

def detect_people(frame):
    result = model(frame)[0]
    person_boxes = []

    for i in range(len(result)):
        cls = result[i].boxes.cls
        if cls == 0:
            person_boxes.append(result[i].boxes.xyxy[0].tolist())

    return person_boxes

def getFramesFromVideoByFrameRate(video, frameRate=60):
    frames = []
    frameCount = 1
    logService = LogService('http://localhost:8080/api/v1/device/1/log')
    while video.isOpened():
        ret, frame = video.read()
        if frameCount % frameRate == 0:
            if not ret and frameCount == 1:
                raise Exception("Não foi possível obter o frame")
            if not ret:
                break
            frames.append(frame)

            # detect person in the frame and draw
            boxes = detect_people(frame)
            near_objects = draw_boxes(frame, boxes)
            imS = cv2.resize(frame, (600, 900))
            cv2.imshow('frame', imS)
            cv2.waitKey(100)

            result =  {
                'number_of_objets': len(boxes),
                'number_of_near_objects': near_objects
            }
            logService.log(result)

        frameCount+=1
    return frames

videoPath = 'D:\\Pastas Windows\\Desktop\\tcc\\person_detector_yolo\\videos\\test.mp4'
video = loadVideo(videoPath)

frames = getFramesFromVideoByFrameRate(video, 15)

video.release()
cv2.destroyAllWindows()