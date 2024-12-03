from pyspark.sql import SparkSession
import sys
import os
import cv2
import numpy as np
import shutil
import logging
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import random
import time

# Chỉ nhập matplotlib nếu bạn cần hiển thị ảnh cục bộ
# Nếu chạy trong môi trường không giao diện, hãy cân nhắc loại bỏ hoặc bình luận các chức năng hiển thị
import matplotlib.pyplot as plt

# =========================
# Cấu hình và Ghi log
# =========================

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
    """Ghi log với thời gian hiện tại."""
    logging.info(message)


# =========================
# Khởi tạo Spark Session
# =========================

def init_spark():
    """Khởi tạo SparkSession."""
    spark = SparkSession.builder \
        .appName("Process Images from HDFS with Advanced Segmentation") \
        .getOrCreate()
    log_message("SparkSession đã được khởi tạo.")
    return spark


# =========================
# Các chức năng thao tác với HDFS
# =========================

def download_images_from_hdfs(spark, hdfs_dir, local_dir):
    """Tải ảnh từ HDFS về thư mục cục bộ."""
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_dir)

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        log_message(f"Đã tạo thư mục cục bộ: {local_dir}")

    for file_status in fs.listStatus(hdfs_path):
        file_path = file_status.getPath().toString()
        file_name = os.path.basename(file_path)
        local_file_path = os.path.join(local_dir, file_name)

        if file_status.isDirectory():
            # Gọi đệ quy nếu là thư mục
            log_message(f"Đang vào thư mục: {file_path}")
            download_images_from_hdfs(spark, file_path, local_file_path)
        elif file_status.isFile():
            # Tải tệp về
            log_message(f"Tìm thấy tệp trong HDFS: {file_path}")
            fs.copyToLocalFile(False, spark._jvm.org.apache.hadoop.fs.Path(file_path),
                               spark._jvm.org.apache.hadoop.fs.Path(local_file_path))
            log_message(f"Đã tải xuống {file_name} vào {local_file_path}")


def upload_images_to_hdfs(spark, local_dir, hdfs_dir):
    """Tải ảnh từ thư mục cục bộ lên HDFS."""
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_dir)

    if not fs.exists(hdfs_path):
        fs.mkdirs(hdfs_path)
        log_message(f"Đã tạo thư mục HDFS: {hdfs_dir}")

    for root, _, files in os.walk(local_dir):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            # Giữ nguyên cấu trúc thư mục trong HDFS
            relative_path = os.path.relpath(local_file_path, local_dir)
            hdfs_file_path = os.path.join(hdfs_dir, relative_path).replace("\\", "/")  # Tương thích với Windows

            fs.copyFromLocalFile(False, True,
                                 spark._jvm.org.apache.hadoop.fs.Path(local_file_path),
                                 spark._jvm.org.apache.hadoop.fs.Path(hdfs_file_path))
            log_message(f"Đã tải lên {file_name} vào {hdfs_file_path}")


def delete_directory_from_hdfs(spark, hdfs_dir):
    """Xóa thư mục trên HDFS nếu tồn tại."""
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_dir)

    if fs.exists(hdfs_path):
        fs.delete(hdfs_path, True)  # True để xóa đệ quy các tệp và thư mục con
        log_message(f"Đã xóa thư mục HDFS: {hdfs_dir}")
    else:
        log_message(f"Thư mục HDFS không tồn tại: {hdfs_dir}")


# =========================
# Các hàm xử lý hình ảnh
# =========================

def contour_based_roi(image):
    """Phát hiện đường viền để tìm Vùng Quan Tâm (ROI)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    result = cv2.bitwise_and(image, image, mask=mask)

    return result


def preprocess_image_with_contours(image):
    """Tiền xử lý ảnh với phương pháp contour và làm mờ Gaussian."""
    roi_image = contour_based_roi(image)
    blurred_image = cv2.GaussianBlur(roi_image, (5, 5), 0)
    return blurred_image


def convert_to_lab(image):
    """Chuyển đổi ảnh sang không gian màu L*a*b*."""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    return lab_image


def kmeans_segmentation(image, k=3):
    """Áp dụng phân cụm K-Means để phân đoạn ảnh."""
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=300)
    labels = kmeans.fit_predict(pixel_values)

    segmented_image = kmeans.cluster_centers_[labels]
    segmented_image = segmented_image.reshape(image.shape).astype(np.uint8)

    labels = labels.reshape(image.shape[:2])

    return segmented_image, labels, kmeans.cluster_centers_


def select_best_cluster(image, labels, kmeans_centers):
    """Chọn cluster tốt nhất dựa trên giá trị trung bình của các trung tâm cluster."""
    cluster_means = np.mean(kmeans_centers, axis=1)
    cluster_idx = np.argmax(cluster_means)
    return select_cluster(image, labels, kmeans_centers, cluster_idx)


def select_cluster(image, labels, kmeans_centers, cluster_idx):
    """Chọn một cluster cụ thể từ ảnh."""
    selected_cluster = np.zeros_like(image)
    mask = labels == cluster_idx
    selected_cluster[mask] = image[mask]
    return selected_cluster


def postprocess_segmented_image(segmented_image):
    """Áp dụng các phép toán hình thái học để cải thiện kết quả phân đoạn."""
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 500
    filtered_image = np.zeros_like(closed_image)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(filtered_image, [contour], -1, 255, thickness=cv2.FILLED)

    return filtered_image


def process_image(image_path):
    """
    Pipeline xử lý ảnh toàn diện:
    - Tiền xử lý với contour
    - Chuyển đổi không gian màu
    - Phân đoạn K-Means
    - Chọn cluster tốt nhất
    - Post-processing
    """
    if not os.path.exists(image_path):
        log_message(f"Không tìm thấy tệp: {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        log_message(f"Lỗi đọc {image_path}: Tệp có thể bị hỏng hoặc không phải là ảnh.")
        return None

    try:
        # Tiền xử lý
        preprocessed_image = preprocess_image_with_contours(image)

        # Chuyển đổi sang không gian màu L*a*b*
        lab_image = convert_to_lab(preprocessed_image)

        # Phân đoạn K-Means
        segmented_image, labels, centers = kmeans_segmentation(lab_image, k=3)

        # Chọn cluster tốt nhất
        selected_cluster = select_best_cluster(image, labels, centers)

        # Post-processing
        post_processed = postprocess_segmented_image(selected_cluster)

        # Tùy chọn: Kết hợp mask post-processed với ảnh gốc nếu cần hiển thị hoặc xử lý thêm

        return post_processed

    except Exception as e:
        log_message(f"Exception khi xử lý {image_path}: {e}")
        return None


# =========================
# Hàm xử lý chính
# =========================

def main():
    """Chương trình chính để xử lý ảnh từ HDFS và tải ảnh đã xử lý lên lại."""
    if len(sys.argv) < 3:
        log_message("Cách sử dụng: spark-submit script.py <input_hdfs_dir> <output_hdfs_dir>")
        sys.exit(1)

    input_hdfs_dir = sys.argv[1]
    output_hdfs_dir = sys.argv[2]

    local_input_dir = "/tmp/dataset"
    local_output_dir = "/tmp/processed_dataset"

    # Xóa các thư mục cục bộ hiện có nếu tồn tại
    if os.path.exists(local_input_dir):
        shutil.rmtree(local_input_dir)
        log_message(f"Đã xóa thư mục hiện có: {local_input_dir}")

    if os.path.exists(local_output_dir):
        shutil.rmtree(local_output_dir)
        log_message(f"Đã xóa thư mục hiện có: {local_output_dir}")

    # Khởi tạo Spark
    spark = init_spark()

    # Xóa thư mục đầu ra hiện có trên HDFS
    delete_directory_from_hdfs(spark, output_hdfs_dir)

    # Tải ảnh từ HDFS về thư mục cục bộ
    download_images_from_hdfs(spark, input_hdfs_dir, local_input_dir)

    # Tạo thư mục đầu ra cục bộ
    if not os.path.exists(local_output_dir):
        os.makedirs(local_output_dir)
        log_message(f"Đã tạo thư mục đầu ra: {local_output_dir}")

    # Xử lý từng ảnh
    for root, _, files in os.walk(local_input_dir):
        for file_name in files:
            input_file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(input_file_path, local_input_dir)
            output_file_path = os.path.join(local_output_dir, relative_path)

            # Đảm bảo rằng thư mục con đầu ra tồn tại
            output_subdir = os.path.dirname(output_file_path)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
                log_message(f"Đã tạo thư mục con: {output_subdir}")

            # Xử lý ảnh
            processed_image = process_image(input_file_path)
            if processed_image is not None:
                # Lưu ảnh đã xử lý
                cv2.imwrite(output_file_path, processed_image)
                log_message(f"Đã xử lý và lưu {file_name} vào {output_file_path}")
            else:
                log_message(f"Không thể xử lý {file_name}")

    # Tải ảnh đã xử lý lên lại HDFS
    upload_images_to_hdfs(spark, local_output_dir, output_hdfs_dir)
    log_message("Đã hoàn tất xử lý ảnh và tải lên HDFS.")

    # Dừng SparkSession
    spark.stop()
    log_message("SparkSession đã dừng.")


if __name__ == "__main__":
    main()