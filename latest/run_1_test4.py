
import os
import sys
import cv2
import shutil
from pyspark.sql import SparkSession

def init_spark():
    """Khởi tạo SparkSession."""
    spark = SparkSession.builder.appName("SimpleImageProcessing").getOrCreate()
    print("SparkSession đã được khởi tạo.")
    return spark

def list_files_in_hdfs(spark, hdfs_dir):
    """Liệt kê tất cả các tệp trong thư mục HDFS."""
    fs = spark._jsc.hadoopConfiguration().get("fs.defaultFS")
    hdfs_path = os.path.join(fs, hdfs_dir)
    files = []
    try:
        file_status = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jsc.hadoopConfiguration()
        ).listStatus(spark._jvm.org.apache.hadoop.fs.Path(hdfs_dir))
        for status in file_status:
            path = status.getPath().toString()
            if not status.isDirectory():
                files.append(path)
        print(f"Tìm thấy {len(files)} tệp trong {hdfs_dir}.")
        return files
    except Exception as e:
        print(f"Lỗi khi liệt kê tệp: {str(e)}")
        return []

def process_image(image_path):
    """Xử lý ảnh đơn giản: đọc ảnh và chuyển sang ảnh xám."""
    try:
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
        return None

def download_image_from_hdfs(spark, hdfs_path, local_path):
    """Tải ảnh từ HDFS về thư mục cục bộ."""
    try:
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jsc.hadoopConfiguration()
        )
        fs.copyToLocalFile(False, spark._jvm.org.apache.hadoop.fs.Path(hdfs_path),
                           spark._jvm.org.apache.hadoop.fs.Path(local_path))
        return True
    except Exception as e:
        print(f"Lỗi khi tải xuống tệp {hdfs_path}: {str(e)}")
        return False

def upload_image_to_hdfs(spark, local_path, hdfs_path):
    """Tải ảnh từ thư mục cục bộ lên HDFS."""
    try:
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jsc.hadoopConfiguration()
        )
        fs.copyFromLocalFile(False, True,
                             spark._jvm.org.apache.hadoop.fs.Path(local_path),
                             spark._jvm.org.apache.hadoop.fs.Path(hdfs_path))
        return True
    except Exception as e:
        print(f"Lỗi khi tải lên tệp {hdfs_path}: {str(e)}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Cách sử dụng: spark-submit script.py <input_hdfs_dir> <output_hdfs_dir>")
        sys.exit(1)

    input_hdfs_dir = sys.argv[1]
    output_hdfs_dir = sys.argv[2]

    temp_local_dir = "/tmp/image_processing"
    if not os.path.exists(temp_local_dir):
        os.makedirs(temp_local_dir)

    spark = init_spark()
    files = list_files_in_hdfs(spark, input_hdfs_dir)

    for hdfs_file in files:
        file_name = os.path.basename(hdfs_file)
        local_input_path = os.path.join(temp_local_dir, file_name)
        local_output_path = os.path.join(temp_local_dir, f"processed_{file_name}")
        hdfs_output_path = os.path.join(output_hdfs_dir, f"processed_{file_name}")

        # Tải ảnh từ HDFS về
        if download_image_from_hdfs(spark, hdfs_file, local_input_path):
            # Xử lý ảnh
            processed_image = process_image(local_input_path)
            if processed_image is not None:
                # Lưu ảnh đã xử lý cục bộ
                cv2.imwrite(local_output_path, processed_image)
                # Tải ảnh đã xử lý lên HDFS
                upload_image_to_hdfs(spark, local_output_path, hdfs_output_path)
                print(f"Đã xử lý và tải lên {hdfs_output_path}")
            else:
                print(f"Không thể xử lý ảnh {hdfs_file}")
            # Xóa tệp cục bộ
            os.remove(local_input_path)
            if os.path.exists(local_output_path):
                os.remove(local_output_path)
        else:
            print(f"Không thể tải xuống ảnh {hdfs_file}")

    spark.stop()
    print("Đã hoàn thành xử lý.")

if __name__ == "__main__":
    main()