import os

import cv2
from imutils import paths


image_paths = list(paths.list_images('../../datasets/animals'))

data = []
labels = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    label = image_path.split(os.path.sep)[-2]
    data.append(image)
    labels.append(label)

print(data)
print(labels)
