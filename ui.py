import csv
import artificialNeuralNetwork


class Console:

    def __showMenu(self):
        def showError(error):
            print("\n", error, "\n")
            self.__showMenu()

        print("***************** ARTIFICIAL NEURAL NETWORK ******************\n")

        learningRate = input("Learning rate(d=0.01): ")
        noEpochs = input("No of epochs (d=100): ")

        if learningRate == "d":
            learningRate = 0.01
        if noEpochs == "d":
            noEpochs = 100

        try:
            learningRate = float(learningRate)
            noEpochs = int(noEpochs)
        except ValueError:
            showError("Incorrect values")

        if noEpochs < 1:
            showError("Incorrect noEpochs")

        with open('column_2C.csv', 'r') as csvfile:
            spamreader = csv.reader(csvfile)
            classificationDataTest = []
            classificationDataTrain = []
            i = 0
            for row in spamreader:
                if row[-1] == 'Abnormal':
                    row[-1] = 1
                else:
                    row[-1] = 0
                if i < 60:
                    classificationDataTest.append(row)
                else:
                    classificationDataTrain.append(row)
                i += 1

            artificialNeuralNetwork.run(classificationDataTrain, classificationDataTest, learningRate, noEpochs)
        print()

    def startUi(self):
        self.__showMenu()


