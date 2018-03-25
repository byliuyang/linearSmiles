import numpy as np
from collections import namedtuple

def accuracy(y, yhat):
    return np.mean(y == yhat)

def mse(y, yhat):
    return 0.5 * np.mean((yhat - y) ** 2)

def yhat(X, w):
    return X.T.dot(w)

def gradient_w(X, y, w, b):
    return X.dot(yhat(X, w) - y)

def gradient_b(X, y, w, b):
    return 1

def error(X, y, w):
    return mse(y, yhat(X, w))

def smile_classifier(trainingFaces, trainingLabels):
    learning_rate = .000008
    tolerance = .00000001

    X = trainingFaces.T
    y = trainingLabels
    w = np.random.rand(X.shape[0])

    prev_error = error(X, y, w)

    is_training = True

    while is_training:
        w = w - learning_rate * gradient(X, y, w)
        current_error = error(X, y, w, b)

        if abs(prev_error - current_error) < tolerance:
            is_training = False
        else:
            prev_error = current_error
    
    def predict(faces):
        predictions =  yhat(faces.T, w)
        return [0 if prediction <= 0.5 else 1 for prediction in predictions]

    Classifier =  namedtuple("Classifier", ["w", "predict"])
    return Classifier(w, predict)

def detect_smile(trainingFaces, trainingLabels, testingFaces, testingLabels):

    print("Start training the model")
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