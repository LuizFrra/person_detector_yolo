import numpy as np
import random
import cv2

from Object import Object
from scipy.spatial import distance as dist

class Draw:
    def __init__(self, image, yoloOutput):
        self.__image = image
        self.__yoloOutput = yoloOutput
        self.__confidenceInput = 0.5
        self.__threshold = 0.6
        self.__max_pixel_distance = 150
    
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
        
        distances = dist.cdist(centroids, centroids, metric="euclidean")
        print(distances)
        for personIndex in range(len(distances)):
            numberOfObjectNear = [x for x in distances[personIndex] if x < self.__max_pixel_distance]
            if len(numberOfObjectNear) > 1:
                objects[personIndex].setIsNear()

        return self.__draw(objects)

    def __draw(self, objects = []):
        for idx, object in enumerate(objects):
            cv2.rectangle(self.__image, (object.x, object.y), object.getRectangle(), object.color, 2)
            cv2.circle(self.__image, object.getCenter(), radius=5, color = [0, 0, 255], thickness=-1)
            text = "{}: {:.2f}".format("Person", object.confidence)
            cv2.putText(self.__image, text, object.getCenter(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, object.color, 2)
    
        return self.__image

    def __getCentroid(self, x, y, w, h):
        xCentroid = int(x + (w/2))
        yCentroid = int(y + (h/2))
        return xCentroid, yCentroid

    def __generateColor(self):
        maxPixelValue = 255
        color = lambda : [random.randint(0, maxPixelValue), random.randint(0, maxPixelValue), random.randint(0, maxPixelValue)]
        return color()

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
