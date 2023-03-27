import os
import cv2
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.optimizers.optimizer_v1 import adam
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np


# ---------- PREPROCESSING DATA ----------
image_directory = r"D:\CHINTYA'S\Privacy\Skripsi\Image Classification G-Colab\Multilabel Classification Image\skin_dataset_multilabel_joined"

# Now let us read metadata to get our Y values (multiple lables)
df = pd.read_csv(r"D:\CHINTYA'S\Privacy\Skripsi\Image Classification G-Colab\Multilabel Classification Image\skin_dataset_metadata.csv")

SIZE = 200
X_dataset = []
for i in tqdm(range(df.shape[0])):
    img = tf.keras.utils.load_img(image_directory + r'\\'+ df['Id'][i] + '.jpg', target_size=(SIZE, SIZE, 3))
    img = tf.keras.utils.img_to_array(img)
    img = img / 255.
    X_dataset.append(img)

X = np.array(X_dataset)

# Id and Genre are not labels to be trained. So drop them from the dataframe.
# No need to convert to categorical as the dataset is already in the right format.
y = np.array(df.drop(['Id', 'Condition'], axis=1))

# divide dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.3, stratify=y, shuffle=True)

#print("Number of classes in the training set: ", len(np.unique(y_train)))
#print("Number of classes in the validation set: ", len(np.unique(y_test)))


# ---------- CNN MODEL ARCHITECTURE ----------
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='sigmoid'))

model.summary()

opt = Adam (learning_rate = 0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# ---------- TRAINING DATA ----------

#data augmentation for training data
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
train_generator = datagen.flow(X_train, y_train, batch_size=16, shuffle=True, subset=None)
validation_generator = datagen.flow(X_train, y_train, batch_size=16, subset=None)

print(len(train_generator))
print("done")


history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=32)

history = model.fit_generator(generator=train_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    steps_per_epoch = len(train_generator) / 32,
                    validation_steps = len(validation_generator) / 32,
                    epochs = 15,
                    workers=-1)

#if want to print the train generator
#batch = next(train_generator)
#print(batch)

'''

# ---------- PLOT THE LOSS AND ACCURACY ---------

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

'''

img = tf.keras.utils.load_img(r"D:\CHINTYA'S\Privacy\Skripsi\Image Classification G-Colab\Multilabel Classification Image\try_clearSkin.jpg", target_size=(SIZE, SIZE, 3))

img = tf.keras.utils.img_to_array(img)
img = img / 255.
plt.imshow(img)
img = np.expand_dims(img, axis=0)

classes = np.array(df.columns[2:])  # Get array of all classes
proba = model.predict(img)  # Get probabilities for each class
sorted_categories = np.argsort(proba[0])[:-7:-1]  # Get class names for top 5 categories

for i in range(min(7, len(sorted_categories) - 1)):
    print("{}".format(classes[sorted_categories[i]]) + " ({:.3})".format(proba[0][sorted_categories[i]]))

_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")


#https://www.kaggle.com/code/moghazy/guide-to-cnns-with-data-augmentation-keras
#https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/
