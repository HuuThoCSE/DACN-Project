import os
import shutil

def rename_and_copy(src, dst):
    # Duyệt qua tất cả thư mục và file trong thư mục nguồn
    for root, dirs, files in os.walk(src):
        # Thay đổi tên thư mục có dấu cách thành dấu gạch dưới
        new_root = root.replace(" ", "_")
        
        # Tạo thư mục đích nếu chưa có
        dst_root = new_root.replace(src, dst)
        if not os.path.exists(dst_root):
            os.makedirs(dst_root)
        
        # Sao chép các file và thay đổi tên file có dấu cách
        for file in files:
            new_file_name = file.replace(" ", "_")
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_root, new_file_name)
            shutil.copy2(src_file, dst_file)
            
        # Đổi tên thư mục nếu cần
        if new_root != root:
            os.rename(root, new_root)

# Đường dẫn thư mục nguồn và đích
source_dir = '/home/ubuntu/huutho/dataset'
destination_dir = '/home/ubuntu/huutho/dataset_ok'

# Gọi hàm
rename_and_copy(source_dir, destination_dir)