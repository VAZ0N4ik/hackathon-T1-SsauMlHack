import contextlib
import json
from collections import defaultdict

import yaml

import cv2
import pandas as pd
from PIL import Image

from utils import *

def convert_coco_json(json_dir="../coco/annotations/", use_segments=False, cls91to80=False):
    """Converts COCO JSON format to YOLO label format, with options for segments and class mapping."""
    save_dir = make_dirs()  # output directory
    coco80 = coco91_to_coco80_class()

    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):
        fn = Path(save_dir) / "labels" / json_file.stem.replace("instances_", "")  # folder name
        fn.mkdir()
        with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
        print(data)
        # Create image dict
        images = {"{:g}".format(x["id"]): x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:g}"]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                # Segments
                if use_segments:
                    if len(ann["segmentation"]) > 1:
                        s = merge_multi_segment(ann["segmentation"])
                        s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                    else:
                        s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                    s = [cls] + s
                    if s not in segments:
                        segments.append(s)

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (*(segments[i] if use_segments else bboxes[i]),)  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")


def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).

    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """
    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def convert_coco_json_1(json_dir="../coco/annotations/"): 
    """Converts COCO JSON format to YOLO label format, with options for segments and class mapping.""" 
    save_dir = make_dirs()  # output directory 
    print(os.listdir(json_dir)) 
 
    # Import json 
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")): 
        try: 
            im=Image.open("JSON2YOLO-main\imgs/"+json_file.name.replace(".json","")+".png") 
        except: 
            im=Image.open("JSON2YOLO-main\imgs/"+json_file.name.replace(".json","")+".jpg") 
        width=im.width 
        height=im.height 
        fn = Path(save_dir) / "labels" / json_file.stem.replace("instances_", "")  # folder name 
        #fn.mkdir() 
        with open(json_file, encoding="UTF-8") as f: 
            data = json.load(f) 
        bboxes=[] 
        for ann in data: 
            box = np.array(ann["coordinates"], dtype=np.float64) 
            box[1]=-(box[0]-box[1]) 
            box[0]=box[0]+(box[1]/2) 
            box1=box[0] 
            box2=box[1] 
            box1[0]/=width 
            box2[0]/=width 
            box1[1]/=height 
            box2[1]/=height 
            cls = int(ann["signature"]) 
            box = [cls] + box1.tolist()+ box2.tolist() 
            bboxes.append(box) 
        with open((fn).with_suffix(".txt"), "a") as file: 
            for i in range(len(bboxes)): 
                print (bboxes[i]) 
                line = (*bboxes[i],)# cls, box or segments 
                file.write(("%g " * len(line)).rstrip() % line + "\n")

def delete_dsstore(path="../datasets"):
    """Deletes Apple .DS_Store files recursively from a specified directory."""
    from pathlib import Path

    files = list(Path(path).rglob(".DS_store"))
    print(files)
    for f in files:
        f.unlink()


if __name__ == "__main__":
    source = "2"

    if source == "COCO":
        convert_coco_json(
            "JSON2YOLO-main\datasets",  # directory with *.json
            use_segments=True,
            cls91to80=True,
        )

    elif source == '2':
        convert_coco_json_1('JSON2YOLO-main\datasets')

    # zip results
    # os.system('zip -r ../coco.zip ../coco')
