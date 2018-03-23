import numpy as np
from collections import namedtuple

def accuracy(y, yhat):
    return np.mean(y == yhat)

def mse(y, yhat):
    return np.mean((y - yhat) ** 2) / 2

def smile_classifier(trainingFaces, trainingLabels):
    y = trainingLabels
    x = trainingFaces.T
    ones = np.ones((trainingFaces.shape[0], 1)).T
    tilde_x = np.vstack((x, ones))
    tilde_w = np.linalg.solve(np.dot(tilde_x, tilde_x.T), tilde_x).dot(y)

    w = tilde_w[:-1]
    b = tilde_w[-1]
    
    def predict(faces):
        yhats = np.dot(faces, w) + b
        return [0 if yhat <= 0.5 else 1 for yhat in yhats]
    
    Classifier =  namedtuple("Classifier", ["w", "b", "predict"])
    return Classifier(w, b, predict)

def detect_smile(trainingFaces, trainingLabels, testingFaces, testingLabels):
    clf = smile_classifier(trainingFaces, trainingLabels)

    print("Training MSE: %f" % mse(trainingLabels, clf.predict(trainingFaces)))
    print("Testing MSE: %f" % mse(testingLabels, clf.predict(testingFaces)))

    print()
    print("Training Accuracy: %f" % accuracy(trainingLabels, clf.predict(trainingFaces)))
    print("Testing Accurary: %f" % accuracy(testingLabels, clf.predict(testingFaces)))

def main():
    trainingFaces = np.load("trainingFaces.npy")
    trainingLabels = np.load("trainingLabels.npy")
    testingFaces = np.load("testingFaces.npy")
    testingLabels = np.load("testingLabels.npy")

    detect_smile(trainingFaces, trainingLabels, testingFaces, testingLabels)

if __name__ == "__main__":
    main()