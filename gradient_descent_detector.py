import numpy as np
from collections import namedtuple

def accuracy(y, yhat):
    return np.mean(y == yhat)

def toLabel(yhat):
    return yhat > 0.5

def mse(y, yhat):
    return 0.5 * np.mean((yhat - y) ** 2)

def yhat(X, w, b):
    return X.T.dot(w) + b

def gradient_w(X, y, w, b):
    return X.dot(yhat(X, w, b) - y) / X.shape[1]

def gradient_b(X, y, w, b):
    return np.mean(yhat(X, w, b) - y)

def error(X, y, w, b):
    return mse(y, yhat(X, w, b))

def smile_classifier(trainingFaces, trainingLabels):
    learning_rate = .005
    tolerance = .000001

    X = trainingFaces.T
    y = trainingLabels
    w = .01 * np.random.rand(X.shape[0])
    b = .01 * np.random.rand()

    prev_error = error(X, y, w, b)

    is_training = True

    while is_training:
        w = w - learning_rate * gradient_w(X, y, w, b)
        b = b - learning_rate * gradient_b(X, y, w, b)
        current_error = error(X, y, w, b)

        if abs(prev_error - current_error) < tolerance:
            is_training = False
        else:
            prev_error = current_error
    
    def predict(faces):
        return yhat(faces.T, w, b)
        
    Classifier =  namedtuple("Classifier", ["w", "b", "predict"])
    return Classifier(w, b, predict)

def detect_smile(trainingFaces, trainingLabels, testingFaces, testingLabels):

    print("Start training the model")
    clf = smile_classifier(trainingFaces, trainingLabels)

    print("Training MSE: %f" % mse(trainingLabels, clf.predict(trainingFaces)))
    print("Testing MSE: %f" % mse(testingLabels, clf.predict(testingFaces)))

    print()
    print("Training Accuracy: %f" % accuracy(trainingLabels, toLabel(clf.predict(trainingFaces))))
    print("Testing Accuracy: %f" % accuracy(testingLabels, toLabel(clf.predict(testingFaces))))

def main():
    trainingFaces = np.load("trainingFaces.npy")
    trainingLabels = np.load("trainingLabels.npy")
    testingFaces = np.load("testingFaces.npy")
    testingLabels = np.load("testingLabels.npy")
    detect_smile(trainingFaces, trainingLabels, testingFaces, testingLabels)

if __name__ == "__main__":
    main()