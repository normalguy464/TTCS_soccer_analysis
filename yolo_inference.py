from ultralytics import YOLO

# Kiểm tra xem GPU có sẵn không
import torch
print(f"GPU available: {torch.cuda.is_available()}")

# Load model
model = YOLO('final_model/best.pt').to('cuda')  # Chuyển model sang GPU

# Dự đoán với GPU
result = model.predict('input_video/08fd33_4.mp4', save=True, device='cuda')  # Chỉ định sử dụng GPU

print(result[0])
print('======================================')
for box in result[0].boxes:
    print(box)