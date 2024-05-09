import os
from PIL import Image
import numpy
from matplotlib import pyplot


class MyImage:
    def __init__(self, image: Image, name: str) -> None:
        self.image = image
        self.name = name

    def show(self):
        self.image.show()

    def getArray(self):
        return numpy.array(self.image.resize((90, 90)))


class ImageLoader:
    def __init__(self) -> None:
        self.storedImage = []
        self.inputImage = []

    def loadImage(self, dataDir: str):
        dirList = os.listdir(dataDir)
        for dirName in dirList:
            if dirName != "README":
                tmpList = []
                imageList = dirList = os.listdir(
                    os.path.join(dataDir, dirName))
                for imageName in imageList:
                    tmpImage = MyImage(image=Image.open(
                        os.path.join(dataDir, dirName, imageName)), name=dirName)
                    tmpList.append(tmpImage)
                self.storedImage.append(tmpList)

    def loadInput(self, dataDir: str):
        imageList = os.listdir(dataDir)
        for imageName in imageList:
            tmpImage = Image.open(os.path.join(dataDir,  imageName))
            self.inputImage.append(
                MyImage(image=tmpImage, name=imageName[:-4]))


class AIClassifier:
    def __init__(self, imageData: list, inputImage: list) -> None:
        self.imageData = imageData
        self.inputImgage = inputImage

    def analyze(self):
        result = []

        # euclidean distance
        for x in self.inputImgage:
            tmpX = []
            for name in self.imageData:
                tmpName = []
                for y in name:
                    tmpName.append([numpy.linalg.norm(y.getArray()-x.getArray(),2), y])
                tmpX.append(tmpName)
            result.append(tmpX)

        # find nearest
        tmp = []
        for x in result:
            tmpX = [9999999, '']
            for name in x:
                for y in name:
                    if tmpX[0] > y[0]:
                        tmpX[0] = y[0]
                        tmpX[1] = y[1]
            tmp.append(tmpX[1])
        result = tmp

        return result


class MyReport:
    def __init__(self, inputData: list, result: list) -> None:
        self.inputData = inputData
        self.result = result

    def report(self):

        accuracy = 0
        pyplot.figure('report')
        for i in range(0, len(self.inputData)):
            pyplot.subplot(len(self.inputData), 2, i*2+1)
            # pyplot.imshow(self.inputData[i].image)
            if i == 0:
                pyplot.title('input')
            # pyplot.text(100, 50,'id: '+ self.inputData[i].name)
            pyplot.text(0, 0,'id: '+ self.inputData[i].name)
            pyplot.gca().axes.xaxis.set_visible(False)
            pyplot.gca().axes.yaxis.set_visible(False)

            pyplot.subplot(len(self.inputData), 2, i*2+2)
            # pyplot.imshow(self.result[i].image)
            if i == 0:
                pyplot.title('output')
            # pyplot.text(100, 50, 'id: '+self.result[i].name)
            pyplot.text(0, 0, 'id: '+self.result[i].name)
            pyplot.gca().axes.xaxis.set_visible(False)
            pyplot.gca().axes.yaxis.set_visible(False)

            if self.inputData[i].name == self.result[i].name:
                accuracy += 1

        pyplot.figtext(0.05, 0.05, 'accuracy: ' + str(accuracy))
        pyplot.show()


if __name__ == "__main__":
    data = ImageLoader()
    data.loadImage('data')
    data.loadInput('input')

    myAI = AIClassifier(imageData=data.storedImage, inputImage=data.inputImage)
    result = myAI.analyze()

    report = MyReport(inputData=data.inputImage, result=result)
    report.report()
