import os
import cv2
import PersonDetector as p

personDector = p.PersonDetector()

def getImagePath(imageName):
    return os.path.sep.join([".", "../images", imageName])

def loadImage(imageName):
    image = cv2.imread(getImagePath(imageName))
    if image is None or image.size == 0:
        raise Exception("Não foi possível carregar a imagem")
    return image

def calculateFrame(scene = 1, divisor = 50):
    if scene == 1: 
        return "00001"
    value =  1 + ((scene - 1) * divisor)
    valueStr = str(value)
    numberLength = len(valueStr)
    padZeros = "0" * (5 - numberLength)
    return padZeros + valueStr

def loadVideo(videoName):
    video = cv2.VideoCapture(videoName)
    if video is None:
        raise Exception("Não foi possível carregar o video")
    return video

def detectOnVideo(videoName):
    video  = loadVideo(videoName)
    frameCount = 1
    while video.isOpened():
        ret, frame = video.read()
        
        if frameCount % 10 == 0:
            if not ret and frameCount == 1:
                raise Exception("Não foi possível obter o frame")
            if not ret:
                break
            
            personDector.execute(frame)
            personDector.draw()

            if cv2.waitKey(10) == ord('q'):
                break

        frameCount+=1

    video.release()

def detectOnImage():
    for i in range(6, 15):
        print(calculateFrame(i + 1))
        imageName = "scene" + calculateFrame(i + 1) + ".png"
        image = loadImage(imageName)
        personDector.execute(image)
        personDector.draw()

def main():
    print("Hello World!")
    detectOnVideo("../videos/shibuya.mp4")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()