from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from PIL import Image, UnidentifiedImageError
import os
import io
import time
import matplotlib.pyplot as plt

# Khởi tạo SparkSession với YARN
spark = SparkSession.builder \
    .appName("Resize and Upload Images to HDFS via YARN") \
    .master("yarn") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop-namenode:9000") \
    .getOrCreate()

# Đường dẫn cục bộ đến thư mục chứa ảnh
local_dir = "/home/fit/square_pick-2/all_images"
# Thư mục đích trên HDFS
hdfs_dir = "/user/fit/all_images"
# Thư mục đích cho ảnh đã xử lý trên HDFS
hdfs_processed_dir = "/user/fit/processed_all_images"

# Hàm để ghi từng file hoặc thư mục lên HDFS (chạy trên driver)
def save_to_hdfs(local_path, hdfs_path):
    if os.path.isdir(local_path):
        # Nếu là thư mục, tạo thư mục trên HDFS và tải các tệp bên trong
        dir_name = os.path.basename(local_path)
        hdfs_dir_path = f"{hdfs_path.rstrip('/')}/{dir_name}"
        spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration()).mkdirs(spark._jvm.org.apache.hadoop.fs.Path(hdfs_dir_path))
        print(f"Successfully created directory {hdfs_dir_path} on HDFS")
        # Lặp qua các tệp và thư mục con bên trong để tải lên
        for item in os.listdir(local_path):
            item_path = os.path.join(local_path, item)
            save_to_hdfs(item_path, hdfs_dir_path)
    else:
        # Nếu là tệp, tải tệp lên HDFS
        file_name = os.path.basename(local_path)
        hdfs_file_path = f"{hdfs_path.rstrip('/')}/{file_name}"
        hdfs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_file_path)
        with open(local_path, "rb") as f:
            output_stream = hdfs.create(hdfs_path, True)
            output_stream.write(f.read())
            output_stream.close()
        print(f"Successfully uploaded {file_name} to HDFS")

# Hàm để thay đổi kích thước ảnh
def resize_image(local_path, output_size=(256, 256)):
    try:
        # Mở ảnh từ đường dẫn cục bộ bằng Pillow
        with open(local_path, "rb") as f:
            img = Image.open(io.BytesIO(f.read()))
            img_resized = img.resize(output_size)
            # Tạo buffer để lưu ảnh dưới dạng nhị phân
            buffer = io.BytesIO()
            img_resized.save(buffer, format=img.format)
            return buffer.getvalue()
    except UnidentifiedImageError as e:
        print(f"Failed to identify image {local_path} for resizing. Error: {e}")
        return None

# Gọi hàm để tải toàn bộ thư mục và các tệp lên HDFS
save_to_hdfs(local_dir, hdfs_dir)

# Lặp qua từng tệp trong thư mục nguồn để thay đổi kích thước và tải lên thư mục đích
for root, _, files in os.walk(local_dir):
    for file in files:
        local_file_path = os.path.join(root, file)
        if not local_file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            print(f"Skipping non-image file: {local_file_path}")
            continue
        # Thay đổi kích thước ảnh
        resized_content = resize_image(local_file_path)
        if resized_content:
            # Tạo đường dẫn HDFS cho ảnh đã thay đổi kích thước
            relative_path = os.path.relpath(local_file_path, local_dir)
            hdfs_resized_path = f"{hdfs_processed_dir}/{relative_path}"
            # Tải ảnh đã thay đổi kích thước lên HDFS
            hdfs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
            hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_resized_path)
            with hdfs.create(hdfs_path, True) as output_stream:
                output_stream.write(resized_content)
            print(f"Successfully uploaded resized image {hdfs_resized_path} to HDFS")

# Dừng SparkSession
spark.stop()
