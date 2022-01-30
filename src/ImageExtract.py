import numpy as np
import cv2

from Object import Object
from scipy.spatial import distance as dist

class ImageExtract:
    def __init__(self, image, yoloOutput, pixelUnitMeter):
        self.__image = image
        self.__yoloOutput = yoloOutput
        self.__confidenceInput = 0.5
        self.__threshold = 0.6
        self.__safe_meter_distance_grather = 2
        self.__pixel_unit_meter = pixelUnitMeter #0.066

    
    def execute(self):
        (boxes, confidences) = self.__getBoxesAndConfidence()
        # Aplica um algoritimo de non-maximum supression : https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/ , esse algoritimo
        # remove caixa delimitadoras redundante, como por exemplo ter duas caixa delimitadora para o mesmo objeto detectado
        detections = cv2.dnn.NMSBoxes(boxes, confidences, self.__confidenceInput, self.__threshold)

        objects = []
        centroids = []

        if len(detections) > 0:
            for detection in detections.flatten():
                (x, y) = (boxes[detection][0], boxes[detection][1])
                (w, h) = (boxes[detection][2], boxes[detection][3])
                object = Object(w, h, x, y, confidences[detection])
                centroids.append(object.getCenterAsArray())
                objects.append(object)
        
        distances = []
        if len(centroids) > 1:
            distances = dist.cdist(centroids, centroids, metric="euclidean")
        
        numberOfObjetsWithOthersObjectNears = 0

        for personIndex in range(len(distances)):
            #numberOfObjectNear = len([x for x in distances[personIndex] if self.__unsafe_pixel_distance > x])
            numberOfObjectNear = len([x for x in distances[personIndex] if ((x * self.__pixel_unit_meter) > self.__safe_meter_distance_grather)])
            if numberOfObjectNear >= 1:
                numberOfObjetsWithOthersObjectNears = numberOfObjetsWithOthersObjectNears + 1
                objects[personIndex].setIsNear()

        numberOfObjects = len(objects)

        return {
            'result': objects,
            'number_of_objets': numberOfObjects,
            'number_of_near_objects': numberOfObjetsWithOthersObjectNears
        }

    def __getBoxesAndConfidence(self):
        boxes = []
        confidences = []
        (H, W, C) = self.__image.shape

        # yolo possui 3 saídas
        for output in self.__yoloOutput:
            # para cada detecção
            for detection in output:
                # obtem uma lista contendo a probabilidade de cada classes, no caso são 80 classes conforme o arquivo coco.names
                scores = detection[5:]
                # obtem o indice do maior valor em scores, ou seja a classe com maior probabilidade, esse valor retornado é exatamente o identificador da classe
                # uma vez que cada linha do arquivo coco.names corresponde a um identificador
                classId = np.argmax(scores)
                # obtém a probabilidade da classe ser o que a yolo acha ser
                confidence = scores[classId]
                
                # interessado apenas em detecção que seja de person, por isso o filtro por classId = 0
                if confidence > self.__confidenceInput and classId == 0:
                    # para cada detecção, a yolo retorna o centro x e y da caixa em que o objeto caixa está contindo, seguido pela largura e altura da caixa
                    # porém esses valores precisam ser convertidores para a escala original da imagem
                    (centerX, centerY, widht, height) = (detection[0:4] * np.array([W, H, W, H])).astype('int')
                    
                    # a partir dos centros é possível derivar a posição do topo da extrema esquerda em que a caixa está na imagem
                    x = int(centerX - (widht / 2))
                    y = int(centerY - (height / 2))

                    # adiciona a caixa detectada junto com as coordenadas ao array de caixas
                    boxes.append([x, y, int(widht), int(height)])
                    confidences.append(float(confidence))
        
        return boxes, confidences