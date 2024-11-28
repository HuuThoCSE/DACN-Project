import os
import shutil

# Đường dẫn tới thư mục gốc
root_directory = '/home/fit/square_pick-2'
# Đường dẫn tới thư mục đích để gom tất cả ảnh
target_directory = os.path.join(root_directory, 'all_images')

# Tạo thư mục đích nếu chưa tồn tại
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Duyệt qua tất cả các thư mục con và tệp trong thư mục gốc
for subdir, dirs, files in os.walk(root_directory):
    for file in files:
        # Kiểm tra nếu tệp có phần mở rộng là một trong các định dạng ảnh phổ biến
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
            file_path = os.path.join(subdir, file)
            target_path = os.path.join(target_directory, file)
            
            # Di chuyển tệp vào thư mục đích
            shutil.move(file_path, target_path)
            print(f"Moved: {file_path} -> {target_path}")

print("All images have been consolidated into the 'all_images' directory.")
