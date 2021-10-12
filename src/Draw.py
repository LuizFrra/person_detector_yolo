import cv2

class Draw:
    def __init__(self, image):
        self.__image = image

    def execute(self, objects = []):
        for idx, object in enumerate(objects):
            cv2.rectangle(self.__image, (object.x, object.y), object.getRectangle(), object.color, 2)
            cv2.circle(self.__image, object.getCenter(), radius=5, color = [0, 0, 255], thickness=-1)
            text = "{}: {:.2f}".format("Person", object.confidence)
            cv2.putText(self.__image, text, object.getCenter(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, object.color, 2)
    
        return self.__image