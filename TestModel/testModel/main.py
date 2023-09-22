import cv2
import numpy as np
from tensorflow.keras.models import load_model
from imutils import paths
from matplotlib import pyplot as plt


'''
по ссылки https://drive.google.com/drive/u/1/folders/1nbMG4Aq7nd6MkvRlFCl2jAjtSFSYPs2-
в папках модель 40 эпох и модель 60 эпох лежат файлы с названием Best_Sign.h5 с уже обученной моделью
'''

# загружаем обученную модель
model = load_model("C:\\Users\\Admin\\Desktop\\Best_Sign.h5")

# загружаем изображения для теста обученной модели из репозитория TestModel/testImages
ImagePaths = list(paths.list_images("C:\\cloneRzhavchina\\rzhavchina\\TestModel\\testImages"))

# преобразуем изображения в тот вид, на котором обучалась НС
data = []
for imagepath in ImagePaths:
  image = cv2.imread(imagepath)
  image = cv2.resize(image, (256, 256))
  data.append(image)
data = np.array(data, dtype="float") / 255.0
pred = model.predict(data)

# просматриваем получившейся результат
for i in range(0, len(pred), 1):
    if ((pred[i][0] > pred[i][1]) and (pred[i][0] > pred[i][2])):
        plt.title('Без ржавчины\n' + str(pred[i][0]) + '  ' + str(pred[i][1]) + '  ' + str(pred[i][2]))
    elif ((pred[i][1] > pred[i][0]) and (pred[i][1] > pred[i][2])):
        plt.title('Поверхностная коррозия\n' + str(pred[i][0]) + '  ' + str(pred[i][1]) + '  ' + str(pred[i][2]))
    elif ((pred[i][2] > pred[i][0]) and (pred[i][2] > pred[i][1])):
        plt.title('Глубокая коррозия\n' + str(pred[i][0]) + '  ' + str(pred[i][1]) + '  ' + str(pred[i][2]))
    print(pred[i][0], pred[i][1], pred[i][2])
    image = cv2.imread(ImagePaths[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


