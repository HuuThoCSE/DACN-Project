import os
import csv

# Đường dẫn tới thư mục chứa ảnh
image_directory = '/home/fit/square_pick-2/all_images'

# Lấy danh sách tất cả các tệp trong thư mục
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'))]

# Đường dẫn tới file CSV để lưu kết quả
output_csv = '/home/fit/square_pick-2/renamed_files.csv'

# Kiểm tra và tạo mới file CSV nếu chưa tồn tại
file_exists = os.path.exists(output_csv)

with open(output_csv, mode='a', newline='', encoding='utf-8') as csv_file:  # Mở file ở chế độ append
    csv_writer = csv.writer(csv_file)
    
    # Nếu file chưa tồn tại, ghi tiêu đề cột
    if not file_exists:
        csv_writer.writerow(['Old Name', 'New Name'])
    
    # Đổi tên từng tệp theo thứ tự
    for index, file_name in enumerate(image_files):
        old_path = os.path.join(image_directory, file_name)
        new_name = f'image_{index + 1:04d}{os.path.splitext(file_name)[1]}'
        new_path = os.path.join(image_directory, new_name)
        os.rename(old_path, new_path)
        # Ghi thông tin tên cũ và tên mới vào file CSV
        csv_writer.writerow([file_name, new_name])
        print(f"Renamed: {file_name} -> {new_name}")

print(f"All images have been renamed in the 'all_images' directory.")
print(f"Details saved in {output_csv}.")
