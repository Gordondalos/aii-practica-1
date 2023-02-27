def read_txt_files_from_directory_to_one_file(directory_path, to_file_path):
    import os
    count = 0
    for filename in os.listdir(directory_path):
        with open(os.path.join(directory_path, filename), "r") as file:
            lines = file.readlines() # читаем все строки в список lines
            for line in lines:
                with open(to_file_path, "a") as fileWrite:
                    ln = filename.split('.')[0] + ',' +line + '\n'
                    if ln and len(ln.rstrip()) > 0:
                        fileWrite.write(ln.rstrip().replace(' ', ',') + '\n')
        file.close()
    fileWrite.close()




# Получим кроп изображения

def getCrop(url, bbox):
    import cv2
    import random
    import string
    image = cv2.imread(url)
    x, y, w, h, conf, cls =  map(int, bbox)
    crop = image[y:h, x:w]
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    cropUrl = f'./crop/{url.split("/")[2].split(".")[0]}-{random_string}.jpg'
    cv2.imwrite(cropUrl, crop)
    return cropUrl




def getCenterAndPosition(bbox):
    import math
    x1, y1, x2, y2, conf, cls = bbox.tolist()
    # находим центр баудинг бокса
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    # Находим вектора позиции центра  - при условии что 640 на 640 это размер фото
    # Вектор A  - расстояние между началом координат 0 - 0 до центра ббокса
    # Вектор B  - расстояние между началом координат 0 - 640 до центра ббокса
    # Вектор C  - расстояние между началом координат 640 - 640 до центра ббокса
    # Вектор D  - расстояние между началом координат 640 - 0 до центра ббокса

    ax = 0
    ay = 0
    a = math.sqrt((center_x - ax)**2 + (center_y - ay)**2)

    bx = 0
    by = 0
    b = math.sqrt((center_x - bx)**2 + (center_y - by)**2)

    cx = 0
    cy = 0
    c = math.sqrt((center_x - cx)**2 + (center_y - cy)**2)

    dx = 0
    dy = 0
    d = math.sqrt((center_x - dx)**2 + (center_y - dy)**2)

    return center_x, center_y, a, b, c, d

def show_image(image, w=15, h=15):
    import cv2
    import matplotlib.pyplot as plt
    plt.figure(figsize=(w,h))
    plt.grid(False)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.close()

def show_image_by_url(url, w=15, h=15):
    import cv2
    import matplotlib.pyplot as plt
    image = cv2.imread(url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # преобразование порядка цветов

    plt.figure(figsize=(w,h))
    plt.grid(False)
    plt.imshow(image)
    plt.show()



def save_model_pickle(model, model_name='model'):
    import pickle
    with open(f'{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
        f.close()


def load_model_picke(model_path):
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        f.close()
        return model






def check_gpu():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Выводим список доступных GPU
        for gpu in gpus:
            print("Имя GPU:", gpu.name)
    else:
        print("GPU не найдены")


def move_image_to_directory(source_folder, destination_folder):

    import os
    import shutil

    source_folder = './corridors'
    destination_folder = './data_image'

    # Создаем папку назначения, если ее еще нет
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Обходим все файлы и подпапки в исходной папке
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Проверяем, является ли файл картинкой
            if file.endswith('.jpg'):
                # Получаем полный путь к файлу
                file_path = os.path.join(root, file)
                # Получаем имя файла
                file_name = os.path.basename(file_path)
                # Перемещаем файл в папку назначения
                shutil.move(file_path, os.path.join(destination_folder, file_name))