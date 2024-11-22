!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="GfoCaZXpeaozAglH6pMn")
project = rf.workspace("yolo-rrhld").project("square_pick")
version = project.version(2)
dataset = version.download("yolov11")