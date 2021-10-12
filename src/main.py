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

def loadVideo(videoName):
    video = cv2.VideoCapture(videoName)
    if video is None:
        raise Exception("Não foi possível carregar o video")
    return video

def captureFrame(frame):
    cv2.imwrite('../images/test.png', frame) # want to save frame here

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
            personDector.execute(frame.copy())
            print(personDector.getLastResult())
            personDector.draw()

            if cv2.waitKey(100) == ord('q'):
                captureFrame(frame)
                break

        frameCount+=1

    video.release()

def detectOnImage(imagePath):
    image = loadImage(imagePath)
    personDector.execute(image)
    personDector.draw()
    cv2.waitKey(0)

def main():
    detectOnVideo("../videos/test.mp4")
    #detectOnImage("../images/test.png")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()