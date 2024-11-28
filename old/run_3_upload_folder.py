from pyspark.sql import SparkSession
import os

# Khởi tạo SparkSession với YARN
spark = SparkSession.builder \
    .appName("Upload Data to HDFS via YARN") \
    .master("yarn") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop-namenode:9000") \
    .getOrCreate()

# Đường dẫn cục bộ đến thư mục chứa ảnh
local_dir = "/home/fit/square_pick-2/all_images"
# Thư mục đích trên HDFS
hdfs_dir = "/user/fit"

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

# Gọi hàm để tải toàn bộ thư mục và các tệp lên HDFS
save_to_hdfs(local_dir, hdfs_dir)

# Dừng SparkSession
spark.stop()