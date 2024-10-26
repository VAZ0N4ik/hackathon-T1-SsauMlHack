import os
from PIL import Image, ImageEnhance

# Функция для обработки изображений
def process_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            # Открываем изображение
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)

            # Повышаем контраст
            enhancer = ImageEnhance.Contrast(img)
            img_contrast = enhancer.enhance(2)  # Увеличьте значение для большей контрастности

            # Конвертируем в черно-белый формат
            img_bw = img_contrast.convert('L')

            # Сохраняем результат (можно изменить директорию для сохранения)
            img_bw.save(os.path.join("datasets\data_preprocessed\images/val", filename))
            print(1)

# Укажите путь к директории
directory_path = 'datasets/data/images/val/'
process_images(directory_path)