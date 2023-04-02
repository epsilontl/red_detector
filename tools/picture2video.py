import os
import cv2
import time


img_path = "/home/wds/Desktop/detection_results"
filelist = sorted(os.listdir(img_path))
fps = 30
size = (1280, 720)

file_path = "video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
videoWriter = cv2.VideoWriter(file_path, fourcc, fps, size)

for item in filelist:
    item = os.path.join(img_path, item)
    img = cv2.imread(item)
    videoWriter.write(img)

videoWriter.release()
