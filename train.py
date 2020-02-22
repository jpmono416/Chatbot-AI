
# import the necessary packages
from model import Model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import l2
from imutils import paths
import numpy as np
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-e", "--epochs", type=int, default=1, help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")


args = vars(ap.parse_args())

# initialize the set of labels from the CALTECH-101 dataset we are
# going to train our network on
LABELS = {"butterfly", "Leopards", "Motorbikes", "airplanes"}


"""
----- FAILED ATTEMPT TO INTERCONNECT MODULES -----
def testCNN(imagePath):
    # extract the class label from the filename
    label = imagePath.split("/")[1].split("\\")[0]

    # if the label of the current image is not part of of the labels
    # are interested in, then ignore the image
    if label not in LABELS:
        return "null"

    # load the image and resize it to be a fixed 96x96 pixels,
    # ignoring aspect ratio
    imageData = cv2.imread(imagePath)
    imageData = cv2.resize(imageData, (96, 96))

    # update the data and labels lists, respectively
    labels.append(label)
    
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(predictions)
    return predictions
"""

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
print("[INFO] loading images...")

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split("/")[1].split("\\")[0]


    # if the label of the current image is not part of of the labels
    # are interested in, then ignore the image
    if label not in LABELS:
        continue

    # load the image and resize it to be a fixed 96x96 pixels,
    # ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (96, 96))

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
trainX, testX, trainY, testY = train_test_split(data, labels,
                                                  train_size=0.75, test_size=0.25, stratify=labels, random_state=42)
# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=1e-4, decay=1e-4 / args["epochs"])
model = Model.build(width=96, height=96, depth=3,
                    classes=len(lb.classes_), reg=l2(0.0005))
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network for {} epochs...".format(
    args["epochs"]))
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,
                        epochs=args["epochs"])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print("It is " + str(predictions))
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

model.save("trained-model")

"""
-----   EXPERIMENTAL    -----
# plot the training loss and accuracy
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
"""