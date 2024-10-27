import contextlib
import json
from collections import defaultdict

import yaml

import cv2
import pandas as pd
from PIL import Image

from utils import *

def make_dirs():
    save_dir = Path("../output")  # Например, результирующая папка
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir

def convert_coco_json_1(json_dir="../coco/annotations/"):
    """Converts COCO JSON format to YOLO label format, with options for segments and class mapping."""
    save_dir = make_dirs()  # output directory
    
    class_names = set()
    labels_dir = save_dir / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Import json 
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")): 
        try: 
            im = Image.open("JSON2YOLO-main/imgs/" + json_file.name.replace(".json", ".png")) 
        except:
            im = Image.open("JSON2YOLO-main/imgs/" + json_file.name.replace(".json", ".jpg")) 
        
        width = im.width 
        height = im.height 
        fn = labels_dir / json_file.stem.replace("instances_", "")  # folder name 
        
        with open(json_file, encoding="UTF-8") as f: 
            data = json.load(f) 
        
        bboxes = [] 
        c = 1
        for ann in data:  # Data format typically has annotations key
            box = np.array(ann["coordinates"], dtype=np.float64) 
            box[1] = -(box[0] - box[1]) 
            box[0] = box[0] + (box[1] / 2) 
            box1 = box[0] 
            box2 = box[1] 
            box1[0] /= width 
            box2[0] /= width 
            box1[1] /= height 
            box2[1] /= height 
            cls = c # Привязка к id класса
            class_names.add(cls)
            box = [cls] + box1.tolist() + box2.tolist() 
            bboxes.append(box) 

            c+=1
        
        with open(fn.with_suffix(".txt"), "a") as file: 
            for bbox in bboxes:
                line = (*bbox,)
                file.write(("%g " * len(line)).rstrip() % line + "\n")

    # Создание файла YAML
    yolo_yaml = {
        "path": str(save_dir),  # Путь к вашему датасету
        "train": "train/images",  # Путь к папке с обучающими изображениями
        "nc": len(class_names),  # Количество классов
        "names": list(class_names)  # Имена классов
    }

    with open(save_dir / 'output.yaml', 'w', encoding='utf-8') as yaml_file:
        yaml.dump(yolo_yaml, yaml_file, allow_unicode=True)

if __name__ == "__main__":
    source = "T1"

    if source == 'T1':
        convert_coco_json_1('..\datasets')

    # zip results
    # os.system('zip -r ../coco.zip ../coco')