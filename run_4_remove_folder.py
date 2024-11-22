from pyspark.sql import SparkSession
from py4j.java_gateway import java_import

# Khởi tạo SparkSession với YARN
spark = SparkSession.builder \
    .appName("Delete HDFS Directory using YARN") \
    .master("spark://spark-master:7077") \
    .master("yarn") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop-namenode:9000") \
    .config("spark.yarn.access.hadoopUser", "fit") \
    .getOrCreate()

# Hàm xóa thư mục trên HDFS
def delete_hdfs_directory(spark, hdfs_path):
    """
    Xóa một thư mục trên HDFS thông qua API HDFS của Spark.

    :param spark: Đối tượng SparkSession
    :param hdfs_path: Đường dẫn thư mục trên HDFS cần xóa (ví dụ: /path/to/dir)
    """
    try:
        hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
        java_import(spark._jvm, "org.apache.hadoop.fs.FileSystem")
        java_import(spark._jvm, "org.apache.hadoop.fs.Path")

        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
        path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_path)

        # Kiểm tra xem thư mục có tồn tại không
        if fs.exists(path):
            # Xóa thư mục (đệ quy)
            fs.delete(path, True)
            print(f"Thư mục '{hdfs_path}' đã được xóa thành công.")
        else:
            print(f"Thư mục '{hdfs_path}' không tồn tại.")
    except Exception as e:
        print(f"Lỗi khi xóa thư mục: {e}")


# Ví dụ sử dụng
delete_hdfs_directory(spark, "/user/fit/processed_all_images")
