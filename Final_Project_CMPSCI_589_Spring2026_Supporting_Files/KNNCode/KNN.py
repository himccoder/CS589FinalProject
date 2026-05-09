import math
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn import datasets


def runTrial(trainData, testData, OnTestData, k):
    columnCnt = len(trainData[0])

    normalizer = MinMaxScaler()

    normalizer.fit(trainData)
    trainScaled = normalizer.transform(trainData)
    testScaled = normalizer.transform(testData)

    def euclidDistance(instance1, instance2):
        sum = 0
        for attribute in range(columnCnt - 1):
            difference = instance1[attribute] - instance2[attribute]
            sum += difference ** 2
        return sum ** 0.5

    def KNNAlgo(instance, k):
        differences = []
        for e in trainScaled:
            differences.append((e, euclidDistance(instance, e)))
        differences.sort(key=lambda x: x[1])
        topK = differences[:k]

        classToCount = {}
        for e in topK:
            label = e[0][columnCnt - 1]
            classToCount[label] = classToCount.get(label, 0) + 1

        return max(classToCount, key=classToCount.get)
    
    def calcStats():
        #Check performance
        fn = 0
        tp = 0
        fp = 0

        reals = []
        predictedVals = []
        for instance in testScaled if OnTestData else trainScaled:
            real = instance[columnCnt - 1]
            reals.append(real)
            predicted = KNNAlgo(instance, k)
            predictedVals.append(predicted)
        
        reals = np.array(reals)
        predictedVals = np.array(predictedVals)
        accuracy = np.mean(reals == predictedVals)

        f1_per_class = []
        for label in np.unique(reals):
            tp = np.sum((predictedVals == label) & (reals == label))
            fp = np.sum((predictedVals == label) & (reals != label))
            fn = np.sum((predictedVals != label) & (reals == label))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall ) / (precision + recall)  if (precision + recall) > 0 else 0
            f1_per_class.append(f1)

        return (accuracy, np.mean(f1_per_class))

    return calcStats()


def plotKToAccuracy(dataset):
    #Plotting Training Data
    means = []
    deviations = []
    NUM_TRIALS = 20
    for k in range(1, 52, 2):
        accuracies = []
        sum = 0
        for trial in range(NUM_TRIALS):
            #Change to False to test on training data
            trainData, testData = train_test_split(dataset, train_size=0.8, test_size=0.2)
            accuracy = runTrial(trainData, testData, True, k)[0]
            sum += accuracy
            accuracies.append(accuracy)
        mean = sum / NUM_TRIALS
        print(f"Mean accuracy of k={k}: {mean}")
        means.append(mean)

        sumDeviation = 0
        for trial in range(NUM_TRIALS):
            sumDeviation += (accuracies[trial] - mean) ** 2
        standardDeviation = math.sqrt(sumDeviation / NUM_TRIALS)
        deviations.append(standardDeviation)

    pyplot.errorbar(range(1, 52, 2), means, deviations, capsize=0.5)

    pyplot.title("KNN Algorithm accuracy over k on Digits Dataset")

    pyplot.xlabel("Value of k")
    pyplot.ylabel("Accuracy over testing data")

    pyplot.show()

def evaluateK(dataset, k):
    KNN_k = k
    foldCnt = 10

    # Stratifying - Group by label (supports any number of classes)
    class_groups = {}
    for row in dataset:
        label = row[len(dataset[0]) - 1]
        if label not in class_groups:
            class_groups[label] = []
        class_groups[label].append(row)

    # Make folds
    folds = []
    for _ in range(foldCnt):
        folds.append([])

    for label in class_groups:
        samples = class_groups[label]
        i = 0
        while len(samples) > 0:
            folds[i % foldCnt].append(samples.pop())
            i += 1

    for i in range(len(folds)):
        folds[i] = shuffle(folds[i])

    sumAccuracy = 0
    sumF1 = 0

    #For each fold:
    for testIdx in range(foldCnt):
        testSet = folds[testIdx]
        trainingSet = []
        for foldIdx in range(foldCnt):
            if foldIdx != testIdx:
                trainingSet.extend(folds[foldIdx])

        accuracy, F1 = runTrial(trainingSet, testSet, False, KNN_k)
        sumAccuracy += accuracy
        sumF1 += F1


    stratifiedAccuracy = sumAccuracy / foldCnt
    stratifiedF1 = sumF1 / foldCnt

    print(f"\nFor k = {KNN_k}")
    print(f"Accuracy: {stratifiedAccuracy}\nF1: {stratifiedF1}\n")


'''#Preprocessing
dataset = pd.read_csv("../datasets/parkinsons.csv", header=0)
dataset = shuffle(dataset)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1].to_numpy()

#Use [0, 3, 4, 5, 6, 8, 9, 10, 11, 12] for credit_approval
categorical_cols = []

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

if len(categorical_cols) > 0:
    X_cat = X.iloc[:, categorical_cols]
    X_num = X.drop(X.columns[categorical_cols], axis=1)

    X_cat_encoded = encoder.fit_transform(X_cat)

    X = np.hstack((X_cat_encoded, X_num.to_numpy()))
else:
    X = X.to_numpy()

dataset_encoded = np.hstack((X, y.reshape(-1, 1)))'''

#If using the numbers dataset
digits = datasets.load_digits(return_X_y=True)

digits_dataset_X = digits[0]
digits_dataset_y = digits[1]

merged_dataset = np.hstack((digits_dataset_X, digits_dataset_y.reshape(-1, 1)))
dataset_encoded = shuffle(merged_dataset)


plotKToAccuracy(dataset_encoded)

#evaluateK(dataset_encoded, 5)