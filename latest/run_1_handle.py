from pyspark.sql import SparkSession
import sys
import os
import cv2
import numpy as np
import shutil
import logging
from datetime import datetime

# Cấu hình logging
log_file = "/tmp/processing_log.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def log_message(message):
    """Ghi log với thời gian hiện tại"""
    logging.info(message)


def init_spark():
    """Khởi tạo SparkSession"""
    spark = SparkSession.builder \
        .appName("Process Images from HDFS") \
        .getOrCreate()
    log_message("SparkSession initialized.")
    return spark


def download_images_from_hdfs(spark, hdfs_dir, local_dir):
    """Tải ảnh từ HDFS về thư mục cục bộ"""
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_dir)

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        log_message(f"Created local directory: {local_dir}")

    for file_status in fs.listStatus(hdfs_path):
        file_path = file_status.getPath().toString()
        file_name = os.path.basename(file_path)
        local_file_path = os.path.join(local_dir, file_name)

        if file_status.isDirectory():
            # Gọi đệ quy nếu là thư mục
            log_message(f"Entering directory: {file_path}")
            download_images_from_hdfs(spark, file_path, local_file_path)
        elif file_status.isFile():
            # Nếu là tệp, tải về
            log_message(f"Found file in HDFS: {file_path}")
            fs.copyToLocalFile(False, spark._jvm.org.apache.hadoop.fs.Path(file_path),
                               spark._jvm.org.apache.hadoop.fs.Path(local_file_path))
            log_message(f"Downloaded {file_name} to {local_file_path}")


def upload_images_to_hdfs(spark, local_dir, hdfs_dir):
    """Tải ảnh từ thư mục cục bộ lên HDFS"""
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_dir)

    if not fs.exists(hdfs_path):
        fs.mkdirs(hdfs_path)
        log_message(f"Created HDFS directory: {hdfs_dir}")

    for file_name in os.listdir(local_dir):
        local_file_path = os.path.join(local_dir, file_name)
        hdfs_file_path = os.path.join(hdfs_dir, file_name)

        fs.copyFromLocalFile(False, True,
                             spark._jvm.org.apache.hadoop.fs.Path(local_file_path),
                             spark._jvm.org.apache.hadoop.fs.Path(hdfs_file_path))
        log_message(f"Uploaded {file_name} to {hdfs_file_path}")


def process_image(image_path):
    """Xử lý ảnh bằng K-Means Clustering"""
    if not os.path.exists(image_path):
        log_message(f"File not found: {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        log_message(f"Error reading {image_path}: File might be corrupted or not an image.")
        return None

    # Chuyển sang không gian màu L*a*b*
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Áp dụng K-Means clustering
    pixel_values = lab_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    _, labels, centers = cv2.kmeans(pixel_values, 3, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
                                    10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels].reshape(image.shape)

    return segmented_image


def delete_directory_from_hdfs(spark, hdfs_dir):
    """Xóa thư mục trên HDFS nếu tồn tại"""
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_dir)

    if fs.exists(hdfs_path):
        fs.delete(hdfs_path, True)  # True để xóa đệ quy các tệp và thư mục con
        log_message(f"Deleted HDFS directory: {hdfs_dir}")
    else:
        log_message(f"HDFS directory does not exist: {hdfs_dir}")


def main():
    """Chương trình chính"""
    if len(sys.argv) < 3:
        log_message("Usage: spark-submit script.py <input_hdfs_dir> <output_hdfs_dir>")
        sys.exit(1)

    input_hdfs_dir = sys.argv[1]
    output_hdfs_dir = sys.argv[2]

    local_input_dir = "/tmp/dataset"
    local_output_dir = "/tmp/processed_dataset"

    # Xóa các thư mục tạm nếu đã tồn tại
    if os.path.exists(local_input_dir):
        shutil.rmtree(local_input_dir)
        log_message(f"Deleted existing directory: {local_input_dir}")

    if os.path.exists(local_output_dir):
        shutil.rmtree(local_output_dir)
        log_message(f"Deleted existing directory: {local_output_dir}")

    # Khởi tạo Spark
    spark = init_spark()

    # Xóa thư mục trên HDFS nếu có
    delete_directory_from_hdfs(spark, output_hdfs_dir)  # Xóa thư mục output HDFS cũ trước khi tải ảnh mới lên

    # Tải ảnh từ HDFS về
    download_images_from_hdfs(spark, input_hdfs_dir, local_input_dir)

    # Tạo thư mục tạm cho ảnh đã xử lý
    if not os.path.exists(local_output_dir):
        os.makedirs(local_output_dir)
        log_message(f"Created output directory: {local_output_dir}")

    # Xử lý từng ảnh
    for file_name in os.listdir(local_input_dir):
        local_file_path = os.path.join(local_input_dir, file_name)
        processed_image = process_image(local_file_path)
        if processed_image is not None:
            output_file_path = os.path.join(local_output_dir, file_name)
            cv2.imwrite(output_file_path, processed_image)
            log_message(f"Processed and saved {file_name} to {output_file_path}")

    # Tải ảnh đã xử lý lên HDFS
    upload_images_to_hdfs(spark, local_output_dir, output_hdfs_dir)
    log_message("Image processing completed.")


if __name__ == "__main__":
    main()
