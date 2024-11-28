from pyspark.sql import SparkSession
import os

# Khởi tạo SparkSession
spark = SparkSession.builder \
    .appName("Upload Images to HDFS") \
    .master("yarn") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://node1:9000") \
    .getOrCreate()

# Đường dẫn cục bộ đến thư mục chứa ảnh
local_dir = "/home/ubuntu/huutho/dataset"
# Đường dẫn trên HDFS để lưu dataset
hdfs_path = "/user/ubuntu/dacn/dataset"

# Khởi tạo hệ thống tệp HDFS
hdfs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())

# Hàm để upload ảnh từ thư mục cục bộ lên HDFS (bao gồm cả cấu trúc thư mục)
def upload_images_to_hdfs(local_path, hdfs_path):
    if os.path.isdir(local_path):
        for root, dirs, files in os.walk(local_path):
            # Tạo cấu trúc thư mục tương ứng trên HDFS
            relative_path = os.path.relpath(root, local_path)
            hdfs_target_dir = os.path.join(hdfs_path, relative_path)
            hdfs.mkdirs(spark._jvm.org.apache.hadoop.fs.Path(hdfs_target_dir))
            
            for file_name in files:
                local_file_path = os.path.join(root, file_name)
                if file_name.endswith(".jpg"):
                    hdfs_file_path = os.path.join(hdfs_target_dir, file_name)
                    # Upload file lên HDFS
                    with open(local_file_path, "rb") as f:
                        output_stream = hdfs.create(spark._jvm.org.apache.hadoop.fs.Path(hdfs_file_path), True)
                        output_stream.write(f.read())
                        output_stream.close()
                    print(f"Successfully uploaded {local_file_path} to {hdfs_file_path}")

# Gọi hàm upload
upload_images_to_hdfs(local_dir, hdfs_path)

# Dừng SparkSession
spark.stop()