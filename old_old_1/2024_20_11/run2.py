import os
import shutil

# Đường dẫn tới thư mục chứa ảnh
image_directory = '/home/fit/square_pick-2/all_images'

# Lấy danh sách tất cả các tệp trong thư mục
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'))]

# Đổi tên từng tệp theo thứ tự
for index, file_name in enumerate(image_files):
    old_path = os.path.join(image_directory, file_name)
    new_name = f'image_{index + 1:04d}{os.path.splitext(file_name)[1]}'
    new_path = os.path.join(image_directory, new_name)
    os.rename(old_path, new_path)
    print(f"Renamed: {old_path} -> {new_path}")

print("All images have been renamed in the 'all_images' directory.")
