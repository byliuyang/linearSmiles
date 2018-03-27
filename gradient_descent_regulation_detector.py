import numpy as np
from collections import namedtuple

def accuracy(y, yhat):
    return np.mean(y == yhat)

def mse(y, yhat):
    return 0.5 * np.mean((yhat - y) ** 2)

def yhat(X, w, b):
    return X.T.dot(w) + b

def gradient_w(X, y, w, b, alpha):
    return X.dot(yhat(X, w, b) - y) / X.shape[1] + alpha * w

def gradient_b(X, y, w, b):
    return np.mean(yhat(X, w, b) - y)

def error_regulated(X, y, w, b, alpha):
    return error_unregulated(X, y, w, b) + 0.5 * alpha * w.T.dot(w)

def error_unregulated(X, y, w, b):
    return mse(y, yhat(X, w, b))

def smile_classifier(trainingFaces, trainingLabels):
    learning_rate = .005
    tolerance = .000001

    X = trainingFaces.T
    y = trainingLabels
    w = .01 * np.random.rand(X.shape[0])
    b = .01 * np.random.rand()
    alpha = 1.0

    prev_error = error_regulated(X, y, w, b, alpha)

    is_training = True

    while is_training:
        w = w - learning_rate * gradient_w(X, y, w, b, alpha)
        b = b - learning_rate * gradient_b(X, y, w, b)
        current_error = error_regulated(X, y, w, b, alpha)

        if abs(prev_error - current_error) < tolerance:
            is_training = False
        else:
            prev_error = current_error
    
    def predict(faces):
        predictions =  yhat(faces.T, w, b)
        return predictions > .5

    Classifier =  namedtuple("Classifier", ["w", "b", "predict"])
    return Classifier(w, b, predict)

def detect_smile(trainingFaces, trainingLabels, testingFaces, testingLabels):

    print("Start training the model")
    clf = smile_classifier(trainingFaces, trainingLabels)

    print("Unregulated Training MSE: %f" % mse(trainingLabels, clf.predict(trainingFaces)))
    print("Unregulated Testing MSE: %f" % mse(testingLabels, clf.predict(testingFaces)))

    print()
    print("Training Accuracy: %f" % accuracy(trainingLabels, clf.predict(trainingFaces)))
    print("Testing Accuracy: %f" % accuracy(testingLabels, clf.predict(testingFaces)))

def main():
    trainingFaces = np.load("trainingFaces.npy")
    trainingLabels = np.load("trainingLabels.npy")
    testingFaces = np.load("testingFaces.npy")
    testingLabels = np.load("testingLabels.npy")
    detect_smile(trainingFaces, trainingLabels, testingFaces, testingLabels)

if __name__ == "__main__":
    main()