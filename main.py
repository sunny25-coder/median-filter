import json
from skimage.io import imread, imsave, imshow, show
from skimage.filters._median import median
from skimage.morphology.footprints import diamond
from skimage.exposure import histogram
import matplotlib.pyplot as plt
import numpy as np


# входные данные
input_data = {
    'path_file': 'MPAU/kotenok.jpg',
    'path_file_result': 'MPAU/kotenok_result.jpg',
    'footprint': 4,
    'mode': 'mirror',  # поведение скользяшего окна
}

# пишем данные в json-файл
with open('input_data.json', 'w') as file_data:
    json.dump(input_data, file_data)

# читаем данные из json-файла
with open('input_data.json') as js_file:
    js_data = json.load(js_file)

path = js_data['path_file']
path_result = js_data['path_file_result']
footprint_ = js_data['footprint']
mode_ = js_data['mode']


image = imread(path)

# выполняем медианную фильтрацию структурным элементом в виде ромба и сохраняем изображение
median_ = median(image, np.dstack((diamond(footprint_), diamond(footprint_), diamond(footprint_))), mode=mode_)
imsave(path_result, median_)

# подсчитываем значения по оси абсцисс и ординат для каждого из каналов (RGB) исходного и обработанного изображения
y_blue_im, x_blue_im = histogram(image[0])
y_green_im, x_green_im = histogram(image[1])
y_red_im, x_red_im = histogram(image[2])

y_blue_im_res, x_blue_im_res = histogram(median_[0])
y_green_im_res, x_green_im_res = histogram(median_[1])
y_red_im_res, x_red_im_res = histogram(median_[2])


# представляем результат на экран
fig = plt.figure(figsize=(11, 5))
plt.axis('off')  # убираем оси общего листа
plt.title('Результаты фильтрации')
fig.add_subplot(2, 2, 1)  # делим область на 2 строки и 2 столбца = 4 места для представления результатов
plt.axis('off')  # убираем оси исходного изображения
imshow(image)

fig.add_subplot(2, 2, 2)
plt.axis('off')  # убираем оси обработанного изображения
imshow(median_)

fig.add_subplot(2, 2, 3)
plt.ylabel('число отсчетов')
plt.xlabel('яркость')
# рисуем графики для 3х каналов (красный, синий, зеленый) тонкой непрерывной линией (для исходного изображения)
plt.plot(x_blue_im, y_blue_im, color='blue', linestyle = '-', linewidth=1)
plt.plot(x_green_im, y_green_im, color='green', linestyle = '-', linewidth=1)
plt.plot(x_red_im, y_red_im, color='red', linestyle = '-', linewidth=1)
plt.legend(['blue', 'green', 'red'])

fig.add_subplot(2, 2, 4)
# рисуем графики для 3х каналов (красный, синий, зеленый) тонкой непрерывной линией (для обработанного изображения)
plt.plot(x_blue_im_res, y_blue_im_res, color='blue', linestyle = '-', linewidth=1)
plt.plot(x_green_im_res, y_green_im_res, color='green', linestyle = '-', linewidth=1)
plt.plot(x_red_im_res, y_red_im_res, color='red', linestyle = '-', linewidth=1)
plt.legend(['blue', 'green', 'red'])
show()

