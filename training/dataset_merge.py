import os
import shutil
import yaml

# Thư mục chứa các datasets
dataset_dirs = [
    "football-players-detection-1",
    "football-players-detection-10",
    "football-players-detection-14"
]

# Tạo thư mục đầu ra
output_dir = "combined_dataset"
os.makedirs(output_dir, exist_ok=True)

# Tạo cấu trúc thư mục
for folder in ["train", "valid", "test"]:
    os.makedirs(os.path.join(output_dir, folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, folder, "labels"), exist_ok=True)

# Gộp các dataset
for i, dataset_dir in enumerate(dataset_dirs):
    print(f"Đang gộp dataset: {dataset_dir}")
    
    # Xử lý từng phân vùng (train/valid/test)
    for folder in ["train", "valid", "test"]:
        img_src_dir = os.path.join(dataset_dir, folder, "images")
        lbl_src_dir = os.path.join(dataset_dir, folder, "labels")
        
        img_dst_dir = os.path.join(output_dir, folder, "images")
        lbl_dst_dir = os.path.join(output_dir, folder, "labels")
        
        # Kiểm tra thư mục tồn tại
        if not os.path.exists(img_src_dir) or not os.path.exists(lbl_src_dir):
            continue
            
        # Sao chép ảnh và nhãn
        for img_file in os.listdir(img_src_dir):
            base_name = os.path.splitext(img_file)[0]
            ext = os.path.splitext(img_file)[1]
            
            # Tạo tên file duy nhất
            new_img_name = f"v{i+1}_{base_name}{ext}"
            
            # Sao chép ảnh
            shutil.copy(
                os.path.join(img_src_dir, img_file),
                os.path.join(img_dst_dir, new_img_name)
            )
            
            # Sao chép nhãn tương ứng
            lbl_file = f"{base_name}.txt"
            if os.path.exists(os.path.join(lbl_src_dir, lbl_file)):
                new_lbl_name = f"v{i+1}_{base_name}.txt"
                shutil.copy(
                    os.path.join(lbl_src_dir, lbl_file),
                    os.path.join(lbl_dst_dir, new_lbl_name)
                )

# Tạo data.yaml cho dataset tổng hợp
# Dùng data.yaml từ một trong các dataset
with open(os.path.join(dataset_dirs[0], "data.yaml"), "r") as f:
    data_yaml = yaml.safe_load(f)

# Cập nhật đường dẫn
data_yaml["path"] = os.path.abspath(output_dir)
data_yaml["train"] = "train/images"
data_yaml["val"] = "valid/images"
data_yaml["test"] = "test/images"

# Lưu file data.yaml mới
with open(os.path.join(output_dir, "data.yaml"), "w") as f:
    yaml.dump(data_yaml, f)

print(f"Hoàn tất! Dataset tổng hợp đã được lưu tại: {output_dir}")
print(f"File cấu hình: {os.path.join(output_dir, 'data.yaml')}")