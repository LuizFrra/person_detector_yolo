from typing import final
import Draw as drawer
import ImageExtract as extractor
import numpy as np
import cv2
import os
import time

class PersonDetector:

    def __init__(self):
        self.__debugEnable = True
        self.__net = self.__readNet()
        self.__lastInformation = {}

    def __getConfigPath(self):
        return "/config/yolo";

    def __getYoloConfigPath(self):
        return os.path.sep.join([".", self.__getConfigPath(), "yolov3.cfg"])
        
    def __getYoloWeightsPath(self):
        return os.path.sep.join([".", self.__getConfigPath(), "yolov3.weights"])

    def __readNet(self):
        if(self.__debugEnable):
            print("Loagind YOLOv3 Model ...")
        return cv2.dnn.readNetFromDarknet(self.__getYoloConfigPath(), self.__getYoloWeightsPath())

    def __getFinalOutputLayers(self):
        layerNames = self.__net.getLayerNames()
        return [layerNames[i[0] - 1] for i in self.__net.getUnconnectedOutLayers()]

    def __constructBlobForImage(self, image):
        return cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)

    def __forwardImageThroughYolo(self, image):
        imageBlob = self.__constructBlobForImage(image)
        self.__net.setInput(imageBlob)
        return self.__net.forward(self.__getFinalOutputLayers())

    def execute(self, image):
        self.__image = image
        start = time.time()
        self.__layerOutputs = self.__forwardImageThroughYolo(self.__image)
        end = time.time()
        imageExtractor = extractor.ImageExtract(self.__image, self.__layerOutputs)
        self.__lastInformation = imageExtractor.execute()
        if(self.__debugEnable):
            print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    def getLastResult(self):
        return {
            'number_of_objets': self.__lastInformation['number_of_objets'],
            'number_of_near_objects': self.__lastInformation['number_of_near_objects']
        }

    def draw(self):
        draw = drawer.Draw(self.__image)
        image = draw.execute(self.__lastInformation['result'])
        cv2.imshow("Image", image)