from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8l.pt') 

    # Train model
    model.train(
        data='d:/TTCS_Soccer_analysis/combined_dataset/data.yaml',
        epochs=100,
        imgsz=640,  
        batch=8,    
        device='0',
        task='detect',
    )