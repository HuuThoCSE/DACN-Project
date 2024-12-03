from pyspark.sql import SparkSession
import os
import sys
import logging
import shutil

# =========================
# Cấu hình và Ghi log
# =========================

# Cấu hình logging
log_file = "/tmp/upload_log.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def log_message(message):
    """Ghi log với thời gian hiện tại."""
    logging.info(message)

# =========================
# Khởi tạo Spark Session
# =========================

def init_spark():
    """Khởi tạo SparkSession."""
    spark = SparkSession.builder \
        .appName("Upload Images to HDFS") \
        .master("yarn") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://node1:9000") \
        .getOrCreate()
    log_message("SparkSession đã được khởi tạo.")
    return spark

# =========================
# Các chức năng thao tác với HDFS
# =========================

def delete_directory_from_hdfs(spark, hdfs_dir):
    """Xóa thư mục trên HDFS nếu tồn tại."""
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_dir)

    if fs.exists(hdfs_path):
        fs.delete(hdfs_path, True)  # True để xóa đệ quy các tệp và thư mục con
        log_message(f"Đã xóa thư mục HDFS: {hdfs_dir}")
    else:
        log_message(f"Thư mục HDFS không tồn tại: {hdfs_dir}")

def upload_images_to_hdfs(spark, local_dir, hdfs_dir):
    """Tải ảnh từ thư mục cục bộ lên HDFS (bao gồm cả cấu trúc thư mục)."""
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    
    # Duyệt qua tất cả các tệp trong thư mục cục bộ
    for root, dirs, files in os.walk(local_dir):
        for file_name in files:
            if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                local_file_path = os.path.join(root, file_name)
                
                # Tính đường dẫn tương đối để bảo toàn cấu trúc thư mục
                relative_path = os.path.relpath(local_file_path, local_dir)
                hdfs_file_path = os.path.join(hdfs_dir, relative_path).replace("\\", "/")  # Tương thích với Windows

                # Tạo thư mục con trên HDFS nếu cần
                hdfs_file_dir = os.path.dirname(hdfs_file_path)
                hdfs_file_dir_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_file_dir)
                if not fs.exists(hdfs_file_dir_path):
                    fs.mkdirs(hdfs_file_dir_path)
                    log_message(f"Đã tạo thư mục con trên HDFS: {hdfs_file_dir}")

                # Định nghĩa đường dẫn tệp trên HDFS
                hdfs_path_file = spark._jvm.org.apache.hadoop.fs.Path(hdfs_file_path)
                
                try:
                    # Upload tệp lên HDFS
                    fs.copyFromLocalFile(False, True, spark._jvm.org.apache.hadoop.fs.Path(local_file_path), hdfs_path_file)
                    log_message(f"Đã tải lên thành công {local_file_path} vào {hdfs_file_path}")
                except Exception as e:
                    log_message(f"Lỗi khi tải lên {local_file_path} vào {hdfs_file_path}: {e}")

def main():
    """Chương trình chính để tải lên ảnh từ thư mục cục bộ lên HDFS."""
    if len(sys.argv) < 3:
        print("Cách sử dụng: spark-submit upload_script.py <local_dir> <hdfs_dir>")
        sys.exit(1)

    local_dir = sys.argv[1]
    hdfs_dir = sys.argv[2]

    # Kiểm tra thư mục cục bộ tồn tại
    if not os.path.exists(local_dir):
        log_message(f"Thư mục cục bộ không tồn tại: {local_dir}")
        sys.exit(1)
    
    # Khởi tạo Spark
    spark = init_spark()

    # Xóa thư mục HDFS nếu muốn làm sạch trước khi tải lên
    delete_directory_from_hdfs(spark, hdfs_dir)

    # Gọi hàm upload
    upload_images_to_hdfs(spark, local_dir, hdfs_dir)

    # Dừng SparkSession
    spark.stop()
    log_message("SparkSession đã dừng.")

if __name__ == "__main__":
    main()