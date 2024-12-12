from pyspark.sql import SparkSession
from pyspark import SparkFiles
from PIL import Image
import io
import os

# Tạo SparkSession
spark = SparkSession.builder.appName("ImageResizing").getOrCreate()

# Kích thước mới cho ảnh
new_size = (100, 100)  # Ví dụ thu nhỏ ảnh về 100x100

# Đường dẫn HDFS cho thư mục chứa ảnh
path_hadoop = 'hdfs:///master:9000'
input_dir = path_hadoop+"hdfs:///path/to/input/images"
output_dir = "hdfs:///path/to/output/resized_images"

# Hàm xử lý thu nhỏ ảnh
def resize_image(data):
    img_path, img_data = data
    try:
        # Mở ảnh từ byte array
        img = Image.open(io.BytesIO(img_data))
        img = img.resize(new_size, Image.ANTIALIAS)
        
        # Chuyển đổi lại thành byte array sau khi thay đổi kích thước
        output = io.BytesIO()
        img.save(output, format="JPEG")
        return (img_path, output.getvalue())
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Đọc các file từ HDFS
image_rdd = spark.sparkContext.binaryFiles(input_dir)

# Áp dụng hàm resize_image để xử lý thu nhỏ ảnh
resized_images = image_rdd.map(resize_image).filter(lambda x: x is not None)

# Lưu ảnh đã xử lý trở lại HDFS
def save_image(data):
    img_path, img_data = data
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    with open(output_path, "wb") as f:
        f.write(img_data)

resized_images.foreach(save_image)

# Dừng SparkSession
spark.stop()
