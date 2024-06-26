import os
from PIL import Image
import numpy
from matplotlib import pyplot
import keras


class MyImage:
    def __init__(self, image: Image, name: str) -> None:
        self.image = image
        self.name = name

    def show(self):
        self.image.show()

    def getArray(self):
        return numpy.array(self.image)


class ImageLoader:

    def loadImage(self, dataDir: str):

        storedImage=[]

        dirList = os.listdir(dataDir)
        for dirName in dirList:
            if dirName != "README":
                imageList = dirList = os.listdir(
                    os.path.join(dataDir, dirName))
                for imageName in imageList:
                    tmpImage = MyImage(image=Image.open(
                        os.path.join(dataDir, dirName, imageName)), name=dirName)
                    storedImage.append(tmpImage)

        return storedImage

    def loadInput(self, dataDir: str):

        inputImage=[]

        imageList = os.listdir(dataDir)
        for imageName in imageList:
            tmpImage = Image.open(os.path.join(dataDir,  imageName))
            inputImage.append(
                MyImage(image=tmpImage, name=imageName[:-4]))

        return inputImage


class AIClassifier:
    def __init__(self, imageData: list, inputImage: list) -> None:
        self.imageData = imageData
        self.inputImgage = inputImage

    def analyze(self):
        result = []

        # euclidean distance
        for x in self.inputImgage:
            k = 5
            value = []
            for y in self.imageData:
                value.append(numpy.sum(numpy.square(x.getArray()-y.getArray())))

            index = numpy.argsort(value)

            # find nearest
            name={'s1':0,'s2':0,'s3':0,'s4':0,'s5':0,'s6':0,'s7':0,'s8':0,'s9':0,'s10':0,'s11':0,'s12':0,'s13':0,'s14':0,'s15':0,'s16':0,'s17':0,'s18':0,'s19':0,'s20':0,'s21':0,'s22':0,'s23':0,'s24':0,'s25':0,'s26':0,'s27':0,'s28':0,'s29':0,'s30':0,'s31':0,'s32':0,'s33':0,'s34':0,'s35':0,'s36':0,'s37':0,'s38':0,'s39':0,'s40':0}
            for i in range(0,len(index)):
                if index[i]<k:
                    name[self.imageData[i].name]+=1
            tmp=[0,'']
            for x in name:
                if tmp[0]<name[x]:
                    tmp=[name[x],x]

            result.append(tmp[1])

        return result

class KerasClassifier:
    def __init__(self, imageData: list, inputImage: list) -> None:
        self.imageData = imageData
        self.inputImgage = inputImage

    def analyze(self):

        train_images = []
        train_labels = []
        for image in self.imageData:
            train_images.append(image.getArray())
            train_labels.append(int(image.name[1:]))
        train_images = numpy.array(train_images)
        train_labels= numpy.array(train_labels)
        train_labels-= 1

        test_images = []
        test_labels = []
        for image in self.inputImgage:
            test_images.append(image.getArray())
            test_labels.append(int(image.name[1:]))
        test_images = numpy.array(test_images)
        test_labels= numpy.array(test_labels)
        test_labels-= 1

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(112, 92)),
            keras.layers.Dense(10000, activation='relu'),
            keras.layers.Dense(40,activation='softmax'),
        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        
        model.fit(train_images, train_labels, epochs=10, validation_data=(test_images,test_labels))

        prediction= model.predict(test_images)
        prediction= numpy.argmax(prediction,axis=1)

        result=[]
        for i in prediction:
            result.append('s'+str(i+1))
        return result

class MyReport:
    def __init__(self, inputData: list, result: list) -> None:
        self.inputData = inputData
        self.result = result

    def report(self):

        #true
        report=[]
        for i in range(0,len(self.inputData)):
            if self.inputData[i].name == self.result[i]:
                report.append([self.inputData[i],self.result[i]])

        row =len(report)
        pyplot.figure('true')
        for i in range(0, row):
            pyplot.subplot(row, 1, i+1)
            pyplot.imshow(report[i][0].image)
            pyplot.text(100, 50,'input: '+ report[i][0].name)
            pyplot.text(100, 100,'output: '+ report[i][1])
            pyplot.gca().axes.xaxis.set_visible(False)
            pyplot.gca().axes.yaxis.set_visible(False)


        pyplot.figtext(0.05,0.05,'count: '+str(len(report)))

        #faluse
        report=[]
        for i in range(0,len(self.inputData)):
            if self.inputData[i].name != self.result[i]:
                report.append([self.inputData[i],self.result[i]])

        row =len(report)
        pyplot.figure('faluse')
        for i in range(0, row):
            pyplot.subplot(row, 1, i+1)
            pyplot.imshow(report[i][0].image)
            pyplot.text(100, 50,'input: '+ report[i][0].name)
            pyplot.text(100, 100,'output: '+ report[i][1])
            pyplot.gca().axes.xaxis.set_visible(False)
            pyplot.gca().axes.yaxis.set_visible(False)


        pyplot.figtext(0.05,0.05,'count: '+str(len(report)))
        pyplot.show()



if __name__ == "__main__":

    data = ImageLoader()

    storedImage = data.loadImage('data')
    inputImage= data.loadInput('input')

    # myAI = AIClassifier(imageData=storedImage, inputImage=inputImage)
    myAI = KerasClassifier(imageData=storedImage, inputImage=inputImage)
    result = myAI.analyze()

    report = MyReport(inputData=inputImage, result=result)
    report.report()