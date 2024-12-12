from pyspark.sql import SparkSession
from hdfs import InsecureClient
import os

# Tạo SparkSession
spark = SparkSession.builder.appName("UploadImagesToHDFS").getOrCreate()

# Đường dẫn cục bộ đến thư mục chứa ảnh
local_dir = "/home/huuthocse/square_pick-2"
# URL và thư mục đích HDFS
hdfs_url = "http://namenode_host:50070"  # Thay bằng địa chỉ WebHDFS
hdfs_dir = "/datasets/"

# Đọc toàn bộ file trong thư mục cục bộ
image_rdd = spark.sparkContext.binaryFiles(f"file://{local_dir}/*")

# Hàm để ghi file lên HDFS
def save_to_hdfs(file_data, hdfs_url, hdfs_dir):
    from hdfs import InsecureClient  # Khởi tạo `InsecureClient` bên trong hàm
    file_path, file_content = file_data
    file_name = os.path.basename(file_path)  # Lấy tên file từ đường dẫn
    hdfs_file_path = os.path.join(hdfs_dir, file_name)  # Đường dẫn đầy đủ trên HDFS
    
    # Khởi tạo client HDFS trong mỗi worker
    client = InsecureClient(hdfs_url, user='huuthocse')
    
    # Ghi dữ liệu lên HDFS
    client.write(hdfs_file_path, file_content, overwrite=True)

# Sử dụng `lambda` để truyền thêm đối số `hdfs_url` và `hdfs_dir`
image_rdd.foreach(lambda file_data: save_to_hdfs(file_data, hdfs_url, hdfs_dir))

# Dừng SparkSession
spark.stop()
