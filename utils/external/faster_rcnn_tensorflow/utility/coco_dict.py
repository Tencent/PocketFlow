# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

class_names = [
    'back_ground', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']


classes_originID = {
    'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4,
    'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9,
    'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13,
    'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17,
    'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22,
    'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27,
    'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33,
    'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37,
    'kite': 38, 'baseball bat': 39, 'baseball glove': 40,
    'skateboard': 41, 'surfboard': 42, 'tennis racket': 43,
    'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48,
    'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53,
    'sandwich': 54, 'orange': 55, 'broccoli': 56, 'carrot': 57,
    'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61,
    'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65,
    'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73,
    'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77,
    'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81,
    'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86,
    'scissors': 87, 'teddy bear': 88, 'hair drier': 89,
    'toothbrush': 90}

originID_classes = {item: key for key, item in classes_originID.items()}
NAME_LABEL_MAP = dict(zip(class_names, range(len(class_names))))
LABEL_NAME_MAP = dict(zip(range(len(class_names)), class_names))

# print (originID_classes)



