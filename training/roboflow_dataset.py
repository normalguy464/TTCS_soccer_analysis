from roboflow import Roboflow
rf = Roboflow(api_key="naaNScBVJ2rEsplDAuSQ")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov8")

project2 = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version2 = project2.version(14)
dataset2 = version2.download("yolov8")

project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(10)
dataset = version.download("yolov8")
                


                

