import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
import tkinter as tk
import os

def Start_Program(Space):
    fig, axs = plt.subplots(2, 3, figsize=(8, 6))
    plt.gcf().canvas.get_tk_widget().master.wm_title(Space)
    dir_path = "Images"
    num_files = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])

    for i in range (1, num_files+1):
        filename = "Images/" +str(i)+ ".jpg"
        # Загрузка изображений
        img = cv2.imread(filename)

        # Преобразование в выбранное цветовое пространство, а также настройка диапазонов
        if Space == "HSV":
            # Преобразование в цветовое пространство HSV
            color_space = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Создание массивов для диапазона цветов
            lower = np.array([0, 20, 70])
            upper = np.array([15, 255, 255])

        elif Space == "YCbCr":
            # Преобразование в цветовое пространство YCrCb
            color_space = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            # Создание массивов для диапазона цветов
            lower = np.array([16, 133, 77])
            upper = np.array([235, 173, 127])

        elif Space == "I1I2I3":
            # Преобразование в цветовое пространство I1I2I3
            I1 = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3
            I2 = (img[:, :, 0] - img[:, :, 2]) / 1
            I3 = (2 * img[:, :, 1] - img[:, :, 0] - img[:, :, 2]) / 2
            color_space = cv2.merge((I1, I2, I3))

            # нормализация
            color_space = cv2.normalize(color_space, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            # Создание массивов для диапазона цветов
            lower = np.array([0, 100, 100])
            upper = np.array([255, 235, 255])

        else:
            print("Error")

        # Разделение на слои
        color_space_array = np.array(color_space)
        first_layer = color_space_array[:, :, 0]
        second_layer = color_space_array[:, :, 1]
        third_layer = color_space_array[:, :, 2]

        # Создание маски на основе диапазона цветов
        mask = cv2.inRange(color_space, lower, upper)

        # Применение маски к изображению
        res = cv2.bitwise_and(img, img, mask=mask)

        # Загрузка детектора Виолы-Джонса
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Применение детектора к изображению
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
        gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=7)

        # Рисование прямоугольников вокруг найденных лиц
        for (x, y, w, h) in faces:
            cv2.rectangle(res, (x, y), (x+w, y+h), (0, 255, 0), 2)

        for (x, y, w, h) in faces2:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Отображение изображений
        #axs[0, 1].imshow(cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB))
        axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title('Original')
        axs[0, 1].imshow(first_layer, cmap='gray')
        axs[0, 1].set_title('first layer')
        axs[0, 2].imshow(second_layer, cmap='gray')
        axs[0, 2].set_title('Second layer')
        axs[1, 0].imshow(third_layer, cmap='gray')
        axs[1, 0].set_title('Third layer')
        axs[1, 1].imshow(mask, cmap='gray')
        axs[1, 1].set_title('Color space  mask')
        axs[1, 2].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        axs[1, 2].set_title('Result')

        # Удаление делений на осях
        for ax in axs.flat:
            ax.axis('off')
        plt.pause(3)

    plt.show()



def Start_Program_HSV():
    space = "HSV"
    Start_Program(space)

def Start_Program_YCbCr():
    space = "YCbCr"
    Start_Program(space)

def Start_Program_I1I2I3():
    space = "I1I2I3"
    Start_Program(space)

#главное окно
root = tk.Tk()
root.geometry("480x250")
root.title("Цветовые пространства")

#Выбор цветогового пространства
BGR = tk.Label(root, text="Выберите одно из цветовых пространств",pady = 20)
BGR.pack()

#Кнопка выбора изображения
button = tk.Button(root, text="HSV", command=Start_Program_HSV, height=1, width=35)
button.pack(anchor=CENTER, pady = 10)
button.pack()

button = tk.Button(root, text="YCbCr", command=Start_Program_YCbCr, height=1, width=35)
button.pack(anchor=CENTER, pady = 10)
button.pack()

button = tk.Button(root, text="I1I2I3", command=Start_Program_I1I2I3, height=1, width=35)
button.pack(anchor=CENTER, pady = 10)
button.pack()

#главный цикл обработки событий
root.mainloop()