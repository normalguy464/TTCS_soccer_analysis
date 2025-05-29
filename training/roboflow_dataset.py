from roboflow import Roboflow
rf = Roboflow(api_key="naaNScBVJ2rEsplDAuSQ")

project2 = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project2.version(2)
dataset = version.download("yolov8")

project8 = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project8.version(8)
dataset = version.download("yolov8")
                
project10 = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project10.version(10)
dataset = version.download("yolov8")

project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(14)
dataset = version.download("yolov8")
                
                                
                


                

