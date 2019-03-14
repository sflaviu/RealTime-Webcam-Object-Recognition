# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import imutils
import cv2
import os
import glob

from sklearn.externals import joblib


#compress image to vector of size [32*32]
def image_to_feature_vector(image, size=(32, 32)):

    return cv2.resize(image, size).flatten()


#construct histogram of image
def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    else:
        cv2.normalize(hist, hist)

    return hist.flatten()


if __name__ == "__main__":

    #set path to folders with learning images
    imagePaths = ["/home/flaviu/Desktop/Faculta/An3/SCS/Proiect/Pictures/Wrapper/",
                  "/home/flaviu/Desktop/Faculta/An3/SCS/Proiect/Pictures/Notebook/",
                  "/home/flaviu/Desktop/Faculta/An3/SCS/Proiect/Pictures/Hand/"]

    rawImages = []
    features = []
    labels = []



    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):

        filenames = glob.glob(str(imagePath) + "*.jpg")
        #images = [cv2.imread(img) for img in filenames]

        label = imagePath.split("/")[9]

        for filename in filenames:

            index = os.path.splitext(os.path.basename(filename))[0]

	    #label corresponds to class
            labels.append(label)

            img = cv2.imread(filename)
            #print(label)

	    #extract features
            pixels = image_to_feature_vector(img)
            hist = extract_color_histogram(img)

            rawImages.append(pixels)
            features.append(hist)


    rawImages = np.array(rawImages)
    features = np.array(features)
    labels = np.array(labels)

    #trainRI, testRI, trainRL, testRL = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
    #trainFeat, testFeat, trainLabels, testLabels = train_test_split(features, labels, test_size=0.25, random_state=42)


    neighbours = 3


    model = LogisticRegression()
    model.fit(rawImages, labels)

    #pred = model.predict(testRI)
    #acc=accuracy_score(pred,testRL)
    #print("Raw pixel accuracy:"+str(acc))

    joblib.dump(model, 'model_logistic_pixel.pkl')


    model = KNeighborsClassifier(neighbours)
    model.fit(features, labels)

    #acc = model.score(testFeat, testLabels)
    #print("Histogram accuracy: " + str(acc))

    joblib.dump(model, 'model_knn_histo.pkl')
