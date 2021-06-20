from typing import final
import Draw as d
import numpy as np
import cv2
import os
import time

class PersonDetector:

    def __init__(self):
        self.__debugEnable = True
        self.__net = self.__readNet()
        self.__imageNameToRead = ""

    def setImagePathToRead(self, str):
        self.__imageNameToRead = str

    def __getConfigPath(self):
        return "../config/yolo";

    def __getLabelsPath(self):
        return os.path.sep.join([".", self.__getConfigPath(), "coco.names"])

    def __getYoloConfigPath(self):
        return os.path.sep.join([".", self.__getConfigPath(), "yolov3.cfg"])
        
    def __getYoloWeightsPath(self):
        return os.path.sep.join([".", self.__getConfigPath(), "yolov3.weights"])

    def __readLabels(self):
        return open(self.__getLabelsPath()).read().strip().split("\n")

    def __readNet(self):
        if(self.__debugEnable):
            print("Loagind YOLOv3 Model ...")
        return cv2.dnn.readNetFromDarknet(self.__getYoloConfigPath(), self.__getYoloWeightsPath())

    def __getImagePath(self, imageName):
        return os.path.sep.join([".", "../images", imageName])

    def __getFinalOutputLayers(self):
        layerNames = self.__net.getLayerNames()
        return [layerNames[i[0] - 1] for i in self.__net.getUnconnectedOutLayers()]

    def __constructBlobForImage(self, image):
        return cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)

    def __forwardImageThroughYolo(self, image):
        imageBlob = self.__constructBlobForImage(image)
        self.__net.setInput(imageBlob)
        return self.__net.forward(self.__getFinalOutputLayers())

    def execute(self):
        imageFullPath = self.__getImagePath(self.__imageNameToRead)
        self.__image = cv2.imread(imageFullPath)
        start = time.time()
        self.__layerOutputs = self.__forwardImageThroughYolo(self.__image)
        end = time.time()
        if(self.__debugEnable):
            print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    
    def draw(self):
        draw = d.Draw(self.__image, self.__layerOutputs)
        image = draw.execute()
        cv2.imshow("Image", image)
        cv2.waitKey(0)