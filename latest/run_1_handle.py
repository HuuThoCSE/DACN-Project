from pyspark.sql import SparkSession
from pyspark.context import SparkContext
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Khởi tạo SparkSession
spark = SparkSession.builder \
    .appName("Process Images from HDFS") \
    .master("yarn") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://node1:9000") \
    .getOrCreate()

sc = SparkContext.getOrCreate()

# Đường dẫn trên HDFS
input_hdfs_dir = "/user/ubuntu/dacn/dataset"
output_hdfs_dir = "/user/ubuntu/dacn/processed_dataset"

# Hàm để tải ảnh từ HDFS về máy cục bộ
def download_images_from_hdfs(hdfs_dir, local_dir):
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_dir)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    for file_status in fs.listStatus(hdfs_path):
        file_path = file_status.getPath().toString()
        file_name = os.path.basename(file_path)
        local_file_path = os.path.join(local_dir, file_name)
        fs.copyToLocalFile(False, spark._jvm.org.apache.hadoop.fs.Path(file_path), spark._jvm.org.apache.hadoop.fs.Path(local_file_path))
        print(f"Downloaded {file_name} to {local_file_path}")

# Hàm để tải ảnh lên HDFS
def upload_images_to_hdfs(local_dir, hdfs_dir):
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    if not fs.exists(spark._jvm.org.apache.hadoop.fs.Path(hdfs_dir)):
        fs.mkdirs(spark._jvm.org.apache.hadoop.fs.Path(hdfs_dir))
    for file_name in os.listdir(local_dir):
        local_file_path = os.path.join(local_dir, file_name)
        hdfs_file_path = os.path.join(hdfs_dir, file_name)
        fs.copyFromLocalFile(False, True, spark._jvm.org.apache.hadoop.fs.Path(local_file_path), spark._jvm.org.apache.hadoop.fs.Path(hdfs_file_path))
        print(f"Uploaded {file_name} to {hdfs_file_path}")

# Hàm xử lý ảnh (ví dụ tiền xử lý và phân đoạn bằng K-Means)
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading {image_path}")
        return None
    
    # Chuyển sang không gian màu L*a*b*
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Áp dụng K-Means clustering
    pixel_values = lab_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    kmeans = cv2.kmeans(pixel_values, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(kmeans[2])
    labels = kmeans[1].flatten()
    segmented_image = centers[labels].reshape(image.shape)

    # Trả về ảnh đã xử lý
    return segmented_image

# Chương trình chính
def main():
    local_input_dir = "/tmp/dataset"
    local_output_dir = "/tmp/processed_dataset"

    # Tải ảnh từ HDFS về máy cục bộ
    download_images_from_hdfs(input_hdfs_dir, local_input_dir)

    if not os.path.exists(local_output_dir):
        os.makedirs(local_output_dir)

    # Xử lý từng ảnh
    for file_name in os.listdir(local_input_dir):
        local_file_path = os.path.join(local_input_dir, file_name)
        processed_image = process_image(local_file_path)
        if processed_image is not None:
            output_file_path = os.path.join(local_output_dir, file_name)
            cv2.imwrite(output_file_path, processed_image)
            print(f"Processed and saved {file_name} to {output_file_path}")

    # Tải ảnh đã xử lý lên HDFS
    upload_images_to_hdfs(local_output_dir, output_hdfs_dir)

if __name__ == "__main__":
    main()
