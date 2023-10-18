import os
import time
from LogService import LogService
from ultralytics import YOLO
import numpy as np
import cv2

pixel_allowed_min_distance = 200

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%2.2f sec' % (te-ts), flush=True)
        print('%2.2f fps' % (1/(te-ts)), flush=True)
        return result
    return timed

def calculate_distance(frame, boxes):
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
            if distance < pixel_allowed_min_distance:
                quantity_of_box_nearby += 1
                cv2.line(frame, (int(center_box1[0]), int(center_box1[1])), (int(center_box2[0]), int(center_box2[1])), (0, 255, 0), 2)
    return quantity_of_box_nearby

model = YOLO("yolov8n.pt")

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
    logService = LogService('http://localhost:8080/device/1/log')
    
    while video.isOpened():
        ret, frame = video.read()
        if frameCount % frameRate == 0:
            if not ret and frameCount == 1:
                raise Exception("Não foi possível obter o frame")
            if not ret:
                break
            frames.append(frame)

            result, ims = process_frame(frame)

            logService.log(result)

            time.sleep(1.5)
            #cv2.imshow('frame', ims)
            #cv2.waitKey(1500)


        frameCount+=1
    return frames

@timeit
def process_frame(frame):
    boxes = detect_people(frame)
    near_objects = calculate_distance(frame, boxes)
    ims = cv2.resize(frame, (600, 900))
    result =  {
        'number_of_objets': len(boxes),
        'number_of_near_objects': near_objects
    }
    return result, ims

def loadVideo(videoName):
    video = cv2.VideoCapture(videoName)
    if video is None:
        raise Exception("Não foi possível carregar o video")
    return video

videoPath = os.path.join('videos', 'test.mp4')
video = loadVideo(videoPath)

frames = getFramesFromVideoByFrameRate(video, 5)

video.release()
cv2.destroyAllWindows()