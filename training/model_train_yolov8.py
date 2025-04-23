from ultralytics import YOLO

if __name__ == '__main__':
    # Sử dụng mô hình nhỏ hơn thay vì yolov8x
    model = YOLO('yolov8m.pt')  # hoặc yolov8n.pt nếu vẫn gặp lỗi OOM

    # Train model
    model.train(
        data='c:/Users/ACER/Documents/TTCS_Soccer_analysis/combined_dataset/data.yaml',
        epochs=10,
        imgsz=640,  # Giảm kích thước ảnh
        batch=8,    # Giảm batch size
        device='0',
        task='detect',
        patience=10, # Early stopping nếu không cải thiện sau 10 epochs
        # cache=False  # Tắt cache để giảm sử dụng bộ nhớ
    )