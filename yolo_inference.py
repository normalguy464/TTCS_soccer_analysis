from ultralytics import YOLO

# Kiểm tra xem GPU có sẵn không
import torch
print(f"GPU available: {torch.cuda.is_available()}")

# Load model
model = YOLO('final_model/best.pt').to('cuda')  # Chuyển model sang GPU

# Key:
video_name = 'test (1).mp4' #new video mp4

# Dự đoán với GPU
result = model.predict(f'input_video/{video_name}', save=True, device='cuda')  # Chỉ định sử dụng GPU

print(result[0])
print('======================================')
for box in result[0].boxes:
    print(box)

# import torch
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))  # Nếu có nhiều GPU thì kiểm tra từng cái
# if torch.cuda.device_count() > 1:
#     print(torch.cuda.get_device_name(1))
