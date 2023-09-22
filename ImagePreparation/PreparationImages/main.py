import cv2
from imutils import paths
import random
import pickle
import numpy as np

'''
код для подготовки изображений из репозитория https://github.com/VadushaLyapushin/rzhavchina.git

в папке ImagePreparation содержится папка Images c 3 исследуемыми классами изображений:
папка without - металл без коррозии
папка surface - металл с поверхностной коррозией
папка deep - металл с глубокой коррозией

ниже указываем пути к этим папкам из вашего клонированного репозитория
(для пользователей Windows необходимо обязательно экранировать бэкслеш \ в пути к папкам)
'''
ImagePathsWithoot = list(paths.list_images("C:\\cloneRzhavchina\\rzhavchina\\ImagePreparation\\Images\\without"))
print(len(ImagePathsWithoot))
ImagePathsSurface = list(paths.list_images("C:\\cloneRzhavchina\\rzhavchina\\ImagePreparation\\Images\\surface"))
print(len(ImagePathsSurface))
ImagePathsDeep = list(paths.list_images("C:\\cloneRzhavchina\\rzhavchina\\ImagePreparation\\Images\\deep"))
print(len(ImagePathsDeep))
ImagePaths = ImagePathsWithoot + ImagePathsSurface + ImagePathsDeep
random.shuffle(ImagePaths)
print(len(ImagePaths))

data = []
labels = []

for imagepath in ImagePaths:
  image = cv2.imread(imagepath)
  image = cv2.resize(image, (256, 256)) # указываем размер изображений, которые далее пойдёт на вход нейронной сети
  data.append(image)
  ''' 
  разделение split необходимо, чтобы в переменную label записать имя папки, к которому относится изображения
  к примеру, считали путь C:\cloneRzhavchina\rzhavchina\ImagePreparation\Images\without\corosion.jpg 
  переменной label присвоится значение without
  '''
  label = imagepath.split('\\')[-2] # раскомментировать строку, если пользователь Windows
  # label = imagepath.split('/')[-2] # раскомментировать, если пользователь Linux
  if label == "without":
    label = [1, 0, 0]
  elif label == "surface":
    label = [0, 1, 0]
  elif label == "deep":
    label = [0, 0, 1]
  labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# запись подготовленного набора данных (указываем путь куда сохранить подготовленные наборы)
with open("C:\\Users\\Admin\\Desktop\\data.pickle", 'wb') as f:
  pickle.dump(data, f)
print("Data seved")

with open("C:\\Users\\Admin\\Desktop\\labels.pickle", 'wb') as f:
  pickle.dump(labels, f)
print("Labels seved")