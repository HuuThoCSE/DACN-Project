from pyspark.sql import SparkSession
import sys
import os
import cv2
import numpy as np
import shutil
import logging
from sklearn.cluster import KMeans

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
        .master("yarn") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://node1:9000") \
        .getOrCreate()
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


def list_files_in_hdfs(spark, hdfs_dir):
    """Liệt kê tất cả các tệp trong thư mục HDFS (bao gồm cả thư mục con)."""
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_dir)
    files = []

    def traverse(path):
        for file_status in fs.listStatus(path):
            file_path = file_status.getPath()
            if file_status.isDirectory():
                traverse(file_path)
            elif file_status.isFile():
                files.append(file_path.toString())

    traverse(hdfs_path)
    return files


def download_image_from_hdfs(spark, hdfs_file_path, local_file_path):
    """Tải một ảnh từ HDFS về đường dẫn cục bộ."""
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_file_path)
    local_path = spark._jvm.org.apache.hadoop.fs.Path(local_file_path)

    try:
        fs.copyToLocalFile(False, hdfs_path, local_path)
        log_message(f"Đã tải xuống {hdfs_file_path} vào {local_file_path}")
        return True
    except Exception as e:
        log_message(f"Lỗi khi tải xuống {hdfs_file_path} vào {local_file_path}: {e}")
        return False


def upload_image_to_hdfs(spark, local_file_path, hdfs_file_path):
    """Upload một ảnh từ đường dẫn cục bộ lên HDFS."""
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_file_path)
    local_path = spark._jvm.org.apache.hadoop.fs.Path(local_file_path)

    # Tạo thư mục con trên HDFS nếu cần
    hdfs_file_dir = os.path.dirname(hdfs_file_path)
    hdfs_file_dir_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_file_dir)
    if not fs.exists(hdfs_file_dir_path):
        fs.mkdirs(hdfs_file_dir_path)
        log_message(f"Đã tạo thư mục con trên HDFS: {hdfs_file_dir}")

    try:
        fs.copyFromLocalFile(False, True, local_path, hdfs_path)
        log_message(f"Đã tải lên thành công {local_file_path} vào {hdfs_file_path}")
        return True
    except Exception as e:
        log_message(f"Lỗi khi tải lên {local_file_path} vào {hdfs_file_path}: {e}")
        return False


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
        return None, None, None

    image = cv2.imread(image_path)
    if image is None:
        log_message(f"Lỗi đọc {image_path}: Tệp có thể bị hỏng hoặc không phải là ảnh.")
        return None, None, None

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
        ground_truth_mask = postprocess_segmented_image(selected_cluster)

        # Return the processed images
        return segmented_image, selected_cluster, ground_truth_mask

    except Exception as e:
        log_message(f"Exception khi xử lý {image_path}: {e}")
        return None, None, None


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

    # Thư mục tạm thời để lưu ảnh tải xuống và xử lý
    temp_download_dir = "/tmp/temp_download"
    temp_upload_dir = "/tmp/temp_upload"

    # Tạo thư mục tạm nếu chưa tồn tại
    for temp_dir in [temp_download_dir, temp_upload_dir]:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            log_message(f"Đã tạo thư mục tạm: {temp_dir}")

    # Khởi tạo Spark
    spark = init_spark()

    # Lấy danh sách tất cả các tệp ảnh trong thư mục HDFS đầu vào
    hdfs_files = list_files_in_hdfs(spark, input_hdfs_dir)
    log_message(f"Tìm thấy {len(hdfs_files)} tệp ảnh trong HDFS: {input_hdfs_dir}")

    # Xử lý từng ảnh một
    for hdfs_file in hdfs_files:
        if not hdfs_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            log_message(f"Bỏ qua tệp không phải ảnh: {hdfs_file}")
            continue

        # Tính đường dẫn tương đối để bảo toàn cấu trúc thư mục
        relative_path = os.path.relpath(hdfs_file, input_hdfs_dir)
        category_dir = os.path.dirname(relative_path)  # Thư mục con (e.g., benign)
        base_name = os.path.basename(relative_path)
        name, ext = os.path.splitext(base_name)

        # Đường dẫn HDFS cho thư mục con
        hdfs_output_category_dir = os.path.join(output_hdfs_dir, category_dir).replace("\\", "/")

        # Đường dẫn cục bộ cho các tệp ảnh
        local_file_path = os.path.join(temp_download_dir, base_name)
        original_processed_local_file_path = os.path.join(temp_upload_dir, f"{name}_original{ext}")
        segmented_local_file_path = os.path.join(temp_upload_dir, f"{name}_segmented_kmeans{ext}")
        selected_cluster_local_file_path = os.path.join(temp_upload_dir, f"{name}_selected_cluster{ext}")
        ground_truth_mask_local_file_path = os.path.join(temp_upload_dir, f"{name}_ground_truth_mask{ext}")

        # Tải ảnh từ HDFS về thư mục tạm
        download_success = download_image_from_hdfs(spark, hdfs_file, local_file_path)
        if not download_success:
            log_message(f"Không thể tải xuống tệp: {hdfs_file}")
            continue

        # Sao chép ảnh gốc vào thư mục upload với hậu tố _original
        try:
            shutil.copy(local_file_path, original_processed_local_file_path)
            log_message(f"Đã sao chép ảnh gốc: {original_processed_local_file_path}")
        except Exception as e:
            log_message(f"Lỗi khi sao chép ảnh gốc: {e}")
            continue

        # Upload ảnh gốc lên HDFS
        upload_success = upload_image_to_hdfs(
            spark,
            original_processed_local_file_path,
            os.path.join(hdfs_output_category_dir, f"{name}_original{ext}").replace("\\", "/")
        )
        if not upload_success:
            log_message(f"Không thể tải lên ảnh gốc: {hdfs_output_category_dir}/{name}_original{ext}")
            # Không dừng quá trình, tiếp tục với các ảnh khác

        # Xử lý ảnh
        segmented_image, selected_cluster, ground_truth_mask = process_image(local_file_path)
        if segmented_image is not None and selected_cluster is not None and ground_truth_mask is not None:
            # Lưu ảnh Segmented Image (K-Means)
            try:
                cv2.imwrite(segmented_local_file_path, segmented_image)
                log_message(f"Đã xử lý và lưu ảnh Segmented K-Means: {segmented_local_file_path}")
            except Exception as e:
                log_message(f"Lỗi khi lưu ảnh Segmented K-Means: {e}")

            # Lưu ảnh Selected Cluster
            try:
                cv2.imwrite(selected_cluster_local_file_path, selected_cluster)
                log_message(f"Đã xử lý và lưu ảnh Selected Cluster: {selected_cluster_local_file_path}")
            except Exception as e:
                log_message(f"Lỗi khi lưu ảnh Selected Cluster: {e}")

            # Lưu ảnh Ground Truth Mask
            try:
                cv2.imwrite(ground_truth_mask_local_file_path, ground_truth_mask)
                log_message(f"Đã xử lý và lưu ảnh Ground Truth Mask: {ground_truth_mask_local_file_path}")
            except Exception as e:
                log_message(f"Lỗi khi lưu ảnh Ground Truth Mask: {e}")

            # Upload ảnh Segmented K-Means lên HDFS
            upload_success = upload_image_to_hdfs(
                spark,
                segmented_local_file_path,
                os.path.join(hdfs_output_category_dir, f"{name}_segmented_kmeans{ext}").replace("\\", "/")
            )
            if not upload_success:
                log_message(f"Không thể tải lên ảnh Segmented K-Means: {hdfs_output_category_dir}/{name}_segmented_kmeans{ext}")

            # Upload ảnh Selected Cluster lên HDFS
            upload_success = upload_image_to_hdfs(
                spark,
                selected_cluster_local_file_path,
                os.path.join(hdfs_output_category_dir, f"{name}_selected_cluster{ext}").replace("\\", "/")
            )
            if not upload_success:
                log_message(f"Không thể tải lên ảnh Selected Cluster: {hdfs_output_category_dir}/{name}_selected_cluster{ext}")

            # Upload ảnh Ground Truth Mask lên HDFS
            upload_success = upload_image_to_hdfs(
                spark,
                ground_truth_mask_local_file_path,
                os.path.join(hdfs_output_category_dir, f"{name}_ground_truth_mask{ext}").replace("\\", "/")
            )
            if not upload_success:
                log_message(f"Không thể tải lên ảnh Ground Truth Mask: {hdfs_output_category_dir}/{name}_ground_truth_mask{ext}")
        else:
            log_message(f"Không thể xử lý tệp: {local_file_path}")

        # Xóa tệp ảnh tạm cục bộ sau khi xử lý
        try:
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
                log_message(f"Đã xóa tệp tạm: {local_file_path}")
            if os.path.exists(original_processed_local_file_path):
                os.remove(original_processed_local_file_path)
                log_message(f"Đã xóa tệp tạm đã xử lý: {original_processed_local_file_path}")
            if os.path.exists(segmented_local_file_path):
                os.remove(segmented_local_file_path)
                log_message(f"Đã xóa tệp tạm đã xử lý: {segmented_local_file_path}")
            if os.path.exists(selected_cluster_local_file_path):
                os.remove(selected_cluster_local_file_path)
                log_message(f"Đã xóa tệp tạm đã xử lý: {selected_cluster_local_file_path}")
            if os.path.exists(ground_truth_mask_local_file_path):
                os.remove(ground_truth_mask_local_file_path)
                log_message(f"Đã xóa tệp tạm đã xử lý: {ground_truth_mask_local_file_path}")
        except Exception as e:
            log_message(f"Lỗi khi xóa tệp tạm: {e}")

    # Xóa thư mục tạm nếu cần (tùy chọn)
    for temp_dir in [temp_download_dir, temp_upload_dir]:
        try:
            shutil.rmtree(temp_dir)
            log_message(f"Đã xóa thư mục tạm: {temp_dir}")
        except Exception as e:
            log_message(f"Lỗi khi xóa thư mục tạm {temp_dir}: {e}")

    # Dừng SparkSession
    spark.stop()
    log_message("SparkSession đã dừng.")


if __name__ == "__main__":
    main()