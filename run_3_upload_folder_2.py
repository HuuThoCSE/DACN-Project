from pyspark.sql import SparkSession
import os

# Khởi tạo SparkSession với YARN
spark = SparkSession.builder \
    .appName("Upload Files to HDFS via YARN") \
    .master("yarn") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop-namenode:9000") \
    .getOrCreate()

# Đường dẫn cục bộ đến thư mục chứa ảnh và file CSV
local_dir = "/home/fit/square_pick-2"
# Thư mục đích trên HDFS
hdfs_dir = "/user/fit"

# Hàm để ghi file lên HDFS
def save_file_to_hdfs(local_file, hdfs_path):
    hdfs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    hdfs_file_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_path)
    with open(local_file, "rb") as f:
        output_stream = hdfs.create(hdfs_file_path, True)
        output_stream.write(f.read())
        output_stream.close()
    print(f"Successfully uploaded {local_file} to {hdfs_path}")

# Hàm upload toàn bộ thư mục hoặc file
def upload_to_hdfs(local_path, hdfs_path):
    if os.path.isdir(local_path):
        # Nếu là thư mục, tạo thư mục trên HDFS
        dir_name = os.path.basename(local_path)
        hdfs_target_dir = f"{hdfs_path.rstrip('/')}/{dir_name}"
        hdfs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        hdfs.mkdirs(spark._jvm.org.apache.hadoop.fs.Path(hdfs_target_dir))
        print(f"Created directory {hdfs_target_dir} on HDFS")
        
        # Lặp qua các file và thư mục con để tải lên
        for item in os.listdir(local_path):
            item_path = os.path.join(local_path, item)
            upload_to_hdfs(item_path, hdfs_target_dir)
    else:
        # Nếu là file, upload trực tiếp lên HDFS
        file_name = os.path.basename(local_path)
        hdfs_file_path = f"{hdfs_path.rstrip('/')}/{file_name}"
        save_file_to_hdfs(local_path, hdfs_file_path)

# Gọi hàm upload để tải thư mục lên HDFS
upload_to_hdfs(local_dir, hdfs_dir)

# Dừng SparkSession
spark.stop()