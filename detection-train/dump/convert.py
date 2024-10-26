import os
import json
import yaml

# Укажите путь к папке с JSON-файлами
dataset_path = 'JSON2YOLO-main\datasets'  # Замените на ваш путь
output_yaml_path = 'output.yaml'

# Список для хранения данных меток
annotations = []

# Чтение всех JSON-файлов в указанной директории
for filename in os.listdir(dataset_path):
    if filename.endswith('.json'):
        with open(os.path.join(dataset_path, filename), 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            annotations.extend(data)

# Подготовка данных для YOLO
classes = list(set(item['content'] for item in annotations if not item['signature']))
yolo_data = {
    "path": dataset_path,                # Путь к вашему датасету
    "train": "JSON2YOLO-main\imgs",             # Путь к обучающим изображениям
    "val": "JSON2YOLO-main\imgs",                # Путь к валидационным изображениям
    "nc": len(classes),                  # Кол-во классов
    "names": classes                      # Имена классов
}

# Запись в YAML файл
with open(output_yaml_path, 'w', encoding='utf-8') as yaml_file:
    yaml.dump(yolo_data, yaml_file, allow_unicode=True)

print("YAML датасет успешно создан.")