import os
import cv2
import threading
import time
from LogService import LogService
import PersonDetector as p
import logging

logging.basicConfig(level=logging.INFO,
                    format='(%(threadName)-10s) %(message)s',
                    )

def getImagePath(imageName):
    return os.path.sep.join([".", "../images", imageName])

def loadImage(imageName):
    image = cv2.imread(getImagePath(imageName))
    if image is None or image.size == 0:
        raise Exception("Não foi possível carregar a imagem")
    return image

def loadVideo(videoName):
    video = cv2.VideoCapture(videoName)
    if video is None:
        raise Exception("Não foi possível carregar o video")
    return video

def captureFrame(frame):
    cv2.imwrite('../images/test.png', frame) # want to save frame here

def detectOnVideo(videoName):
    logService = LogService('http://168.119.178.10/api/v1/device/1/log')
    video  = loadVideo(videoName)
    frameCount = 1
    personDector = p.PersonDetector()
    while video.isOpened():
        ret, frame = video.read()
        if frameCount % 1 == 0:
            if not ret and frameCount == 1:
                raise Exception("Não foi possível obter o frame")
            if not ret:
                break
            personDector.execute(frame.copy())
            result = personDector.getLastResult()
            logService.log(result)
            logging.info(result)
            #personDector.draw()

            if cv2.waitKey(100) == ord('q'):
                captureFrame(frame)
                return "stop";

        frameCount+=1

    video.release()

def detectOnImage(imagePath):
    image = loadImage(imagePath)
    personDector = p.PersonDetector()
    personDector.execute(image)
    personDector.draw()
    cv2.waitKey(0)


def detectOnVideoInfinite():
    while True:
        result = detectOnVideo("../videos/test.mp4")
        if result == "stop":
            break; 

def detectOnVideoWithThreads(numberOfThreads = 1):
    threads = []
    for i in range(0, numberOfThreads):
        threads.append(threading.Thread(target=detectOnVideoInfinite))
    for thread in threads:
        thread.start()
        time.sleep(4)
    for thread in threads:
        thread.join()

def main():
    detectOnVideoWithThreads(1)
    #detectOnImage("../images/test.png")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()