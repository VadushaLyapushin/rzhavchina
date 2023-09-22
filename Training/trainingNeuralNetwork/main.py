from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# открываем подготовленные наборы данных data.pickle и labels.pickle
with open("C:\\Users\\Admin\\Desktop\\data.pickle", 'rb') as f:
  data = pickle.load(f)
print("Data loaded")
with open("C:\\Users\\Admin\\Desktop\\labels.pickle", 'rb') as f:
  labels = pickle.load(f)
print("Labels loaded")

# разбиваем набор данных на тренировочную и валидационную выборку (указываем размер тестовой выборки)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=32)

# строим модель
model = Sequential()
inputShape = (256, 256, 3) # указываем размер фотографий, которые задали при подготовке изображений
chanDim = -1

model.add(Conv2D(8, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(16, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation("softmax"))
print ("End")



NUM_EPOCHS = 70 # задаём количество эпох обучение
BS = 32 # задаём размер мини-батча

opt = tf.keras.optimizers.Adam(
    learning_rate = 1e-3,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam")

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

# функция трансформирует изображения в процессе обучения (необходима, так как небольшой набор данных)
aug = ImageDataGenerator(
	rotation_range=5,
	zoom_range=0.05,
	width_shift_range=0.05,
	height_shift_range=0.05,
	shear_range=0.10,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")


classTotals = trainY.sum(axis=0)
print(classTotals)
classWeight = classTotals.max() / classTotals
print(classWeight/1)

# указываем путь куда будет сохраняться модель
checkpointer = ModelCheckpoint(filepath='C:\\Users\\Admin\\Desktop\\Best_Sign.h5', verbose=1, save_best_only=True)

H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
						validation_data=(testX, testY),
						steps_per_epoch=trainX.shape[0] // BS,
						epochs=NUM_EPOCHS,
                        shuffle=True,
                        class_weight={0: classWeight[0], # если классифицируем более 3 классов, добавляем данные в словарь
                                      1: classWeight[1], # например для 4 классов необходимо добавить 3: classWeight[3]
                                      2: classWeight[2]},
                        callbacks=[checkpointer])

predictions = model.predict(testX, batch_size = 16)

# выводим отчёт для оценки обучения модели (просматриваем результат валидационный выборки)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=("without", "surface", "deep")))

# график для оценки точности
N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Results")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("C:\\Users\\Admin\\Desktop\\Accuracy.png") # указываем путь куда сохранить график
print("End")

# график для оценки ошибки
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="vall_loss")
plt.title("Results")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig("C:\\Users\\Admin\\Desktop\\Loss.png") # указываем путь куда сохранить график
print("End")