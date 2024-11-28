from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from PIL import Image, UnidentifiedImageError
import os
import io
import shutil

# Khởi tạo SparkSession với YARN
spark = SparkSession.builder \
    .appName("Resize and Upload Images to HDFS via YARN") \
    .master("yarn") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop-namenode:9000") \
    .getOrCreate()

# Thư mục tạm cục bộ để lưu ảnh
temp_local_dir = "/tmp/processed_images"
# Đường dẫn file CSV chứa thông tin ảnh trên HDFS
hdfs_csv_path = "/user/fit/renamed_files.csv"
# Thư mục đích cho ảnh đã xử lý trên HDFS
hdfs_processed_dir = "/user/fit/processed_all_images"

# Tạo thư mục tạm nếu chưa tồn tại
os.makedirs(temp_local_dir, exist_ok=True)

# Đọc file CSV từ HDFS để lấy đường dẫn ảnh
csv_df = spark.read.csv(hdfs_csv_path, header=True)
image_paths = csv_df.select("New Name").rdd.flatMap(lambda x: x).collect()

# Hàm tải file từ HDFS về cục bộ
def download_from_hdfs(hdfs_path, local_path):
    hdfs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    hdfs_file_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_path)

    if hdfs.exists(hdfs_file_path):
        input_stream = hdfs.open(hdfs_file_path)
        try:
            with open(local_path, "wb") as f:
                data_input_stream = spark._jvm.java.io.DataInputStream(input_stream)
                buffer = bytearray()
                b = data_input_stream.read()
                while b != -1:
                    buffer.append(b)
                    b = data_input_stream.read()
                f.write(buffer)
            print(f"Downloaded {hdfs_path} to {local_path}")
        finally:
            input_stream.close()

# Hàm upload file lên HDFS
def upload_to_hdfs(local_path, hdfs_path):
    hdfs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    hdfs_file_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_path)
    with open(local_path, "rb") as f:
        output_stream = hdfs.create(hdfs_file_path, True)
        output_stream.write(f.read())
        output_stream.close()
    print(f"Uploaded {local_path} to {hdfs_path}")

# Hàm thay đổi kích thước ảnh
def resize_image(input_path, output_path, output_size=(256, 256)):
    try:
        with open(input_path, "rb") as f:
            img = Image.open(io.BytesIO(f.read()))
            img_resized = img.resize(output_size)
            img_resized.save(output_path, format=img.format)
            print(f"Resized image saved to {output_path}")
    except UnidentifiedImageError as e:
        print(f"Failed to identify image {input_path} for resizing. Error: {e}")

# Duyệt qua tất cả đường dẫn ảnh trong file CSV
for image_name in image_paths:
    hdfs_image_path = f"{hdfs_processed_dir}/{image_name}"
    local_file_path = os.path.join(temp_local_dir, image_name)
    resized_file_path = os.path.join(temp_local_dir, f"resized_{image_name}")

    # Tải ảnh từ HDFS về thư mục tạm
    download_from_hdfs(hdfs_image_path, local_file_path)

    # Resize ảnh và lưu vào thư mục tạm
    resize_image(local_file_path, resized_file_path)

    # Tải ảnh đã resize lên HDFS
    hdfs_resized_path = f"{hdfs_processed_dir}/{image_name}"
    upload_to_hdfs(resized_file_path, hdfs_resized_path)

    # Xóa file resized khỏi thư mục tạm
    os.remove(local_file_path)
    os.remove(resized_file_path)
    print(f"Cleaned up temporary files for {image_name}")

# Xóa thư mục tạm sau khi hoàn thành
shutil.rmtree(temp_local_dir)
print("Cleaned up temporary directory.")

# Dừng SparkSession
spark.stop()
