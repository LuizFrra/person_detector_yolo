class Object:
    def __init__(self, width, height, x, y, confidence):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.xCentroid = int(self.x + (self.width/2))
        self.yCentroid = int(self.y + (self.height/2))
        self.xRect = self.x + self.width
        self.yRect = self.y + self.height
        self.isNear = False
        self.color = [0, 150, 0]
        self.confidence = confidence
    
    def getRectangle(self):
        return (self.xRect, self.yRect)
    
    def getCenter(self):
        return (self.xCentroid, self.yCentroid)

    def getCenterAsArray(self):
        return [self.xCentroid, self.yCentroid]
    
    def setIsNear(self):
        self.isNear = True
        self.color = [0, 0, 255]
