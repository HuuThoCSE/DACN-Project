
# COMMAND Ở LOCAL
```
spark-submit \
--master local[*] \
--deploy-mode client \
--conf spark.yarn.archive=hdfs://node1:9000/user/ubuntu/spark/spark-archive.zip \
--conf spark.hadoop.fs.defaultFS=hdfs://node1:9000 \
--conf spark.port.maxRetries=100 \
--conf spark.driver.extraJavaOptions="-Djava.net.preferIPv4Stack=true" \
--conf spark.executor.extraJavaOptions="-Djava.net.preferIPv4Stack=true" \
--conf spark.driver.host=172.20.201.154 \
--conf spark.driver.bindAddress=0.0.0.0 \
--conf spark.driver.port=7077 \
--conf spark.blockManager.port=10025 \
--executor-memory 2G \
--executor-cores 2 \
--num-executors 2 \
--conf spark.network.timeout=800s \
--conf spark.broadcast.timeout=600 \
--py-files /home/ubuntu/huutho/dependencies.zip \
~/huutho/DACN-Project/latest/run_1_test3.py \
/user/ubuntu/dacn/dataset/ \
/user/ubuntu/dacn/processed_dataset1/
```

# Cơ bản - Ok (Đã test lần đầu - Ok) - 1h 27' 35'
```
spark-submit \
--master yarn \
--deploy-mode cluster \
--name "Spark-Test-1" \
--conf spark.yarn.archive=hdfs://node1:9000/user/ubuntu/spark/spark-archive.zip \
--conf spark.executorEnv.PYSPARK_PYTHON=/home/ubuntu/environment/bin/python \
--conf spark.driverEnv.PYSPARK_PYTHON=/home/ubuntu/environment/bin/python \
--conf spark.hadoop.fs.defaultFS=hdfs://node1:9000 \
--executor-memory 2G \
--executor-cores 2 \
--num-executors 2 \
~/huutho/DACN-Project/latest/run_1_test3.py \
/user/ubuntu/dacn/dataset \
/user/ubuntu/dacn/processed_dataset4
```

## Nâng cao - Cải thiện hiệu xuất và độ ổn định
```
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --conf spark.yarn.archive=hdfs://node1:9000/user/ubuntu/spark/spark-archive.zip \
  --conf spark.hadoop.fs.defaultFS=hdfs://node1:9000 \
  --conf spark.port.maxRetries=100 \
  --conf spark.driver.extraJavaOptions="-Djava.net.preferIPv4Stack=true" \
  --conf spark.executor.extraJavaOptions="-Djava.net.preferIPv4Stack=true" \
  --conf spark.network.timeout=800s \
  --conf spark.broadcast.timeout=600 \
  --executor-memory 2G \
  --executor-cores 2 \
  --num-executors 2 \
  --conf spark.executorEnv.PYSPARK_PYTHON=/home/ubuntu/environment/bin/python \
  --conf spark.driverEnv.PYSPARK_PYTHON=/home/ubuntu/environment/bin/python \
  --py-files /home/ubuntu/huutho/dependencies.zip \
  ~/huutho/DACN-Project/latest/run_1_test3.py \
  /user/ubuntu/dacn/dataset/ \
  /user/ubuntu/dacn/processed_dataset1/
```

# Dùng spark dataclone
```
spark-submit \
    --master spark://node1:7077 \
    --deploy-mode cluster \
    --conf spark.executorEnv.PYSPARK_PYTHON=/home/ubuntu/environment/bin/python \
    --conf spark.driverEnv.PYSPARK_PYTHON=/home/ubuntu/environment/bin/python \
    --conf spark.hadoop.fs.defaultFS=hdfs://node1:9000 \
    --executor-memory 2G \
    --executor-cores 2 \
    --total-executor-cores 4 \
    ~/huutho/DACN-Project/latest/run_1_test3.py \
    /user/ubuntu/dacn/dataset \
    /user/ubuntu/dacn/processed_dataset3
```

# Ok ổn - 1.5h
```
spark-submit \
    --master spark://master:7077 \
    --deploy-mode client \
    --name "Spark-Standalone-1204-Test-1" \
    --conf spark.driver.host=master \
    --conf spark.driver.bindAddress=0.0.0.0 \
    --conf spark.executorEnv.PYSPARK_PYTHON=/home/ubuntu/anaconda3/bin/python \
    --conf spark.driverEnv.PYSPARK_PYTHON=/home/ubuntu/anaconda3/bin/python \
    --conf spark.hadoop.fs.defaultFS=hdfs://node1:9000 \
    --executor-memory 2G \
    --executor-cores 2 \
    --total-executor-cores 4 \
    --files /home/ubuntu/huutho/DACN-Project/latest/run_1_test5.py \
    --py-files /home/ubuntu/huutho/dependencies.zip \
    ~/huutho/DACN-Project/latest/run_1_test5.py \
    /user/ubuntu/dacn/dataset \
    /user/ubuntu/dacn/processed_dataset6
```

# Dynamic Resource Allocation - Chạy được (Tăng dung lượng) -
```
spark-submit \
    --master spark://master:7077 \
    --deploy-mode client \
    --name "Spark-Standalone-1204-Test-1" \
    --conf spark.driver.host=master \
    --conf spark.driver.bindAddress=0.0.0.0 \
    --conf spark.executorEnv.PYSPARK_PYTHON=/home/ubuntu/anaconda3/bin/python \
    --conf spark.driverEnv.PYSPARK_PYTHON=/home/ubuntu/anaconda3/bin/python \
    --conf spark.hadoop.fs.defaultFS=hdfs://node1:9000 \
    --conf spark.dynamicAllocation.enabled=true \
    --conf spark.dynamicAllocation.minExecutors=2 \
    --conf spark.dynamicAllocation.maxExecutors=10 \
    --conf spark.dynamicAllocation.executorIdleTimeout=60s \
    --conf spark.shuffle.service.enabled=true \
    --executor-memory 4G \
    --executor-cores 4 \
    --files /home/ubuntu/huutho/DACN-Project/latest/run_1_test5.py \
    --py-files /home/ubuntu/huutho/dependencies.zip \
    ~/huutho/DACN-Project/latest/run_1_test5.py \
    /user/ubuntu/dacn/dataset \
    /user/ubuntu/dacn/processed_dataset7
```

## Đợi test
```
spark-submit \
    --master spark://master:7077 \
    --deploy-mode client \
    --name "Spark-Standalone-1204-Test-1" \
    --conf spark.driver.host=master \
    --conf spark.driver.bindAddress=0.0.0.0 \
    --conf spark.executorEnv.PYSPARK_PYTHON=/home/ubuntu/anaconda3/bin/python \
    --conf spark.driverEnv.PYSPARK_PYTHON=/home/ubuntu/anaconda3/bin/python \
    --conf spark.hadoop.fs.defaultFS=hdfs://node1:9000 \
    --executor-memory 5.4G \
    --executor-cores 3 \
    --total-executor-cores 9 \
    --files /home/ubuntu/huutho/DACN-Project/latest/run_1_test5.py \
    --py-files /home/ubuntu/huutho/dependencies.zip \
    ~/huutho/DACN-Project/latest/run_1_test5.py \
    /user/ubuntu/dacn/dataset \
    /user/ubuntu/dacn/processed_dataset8
```

# Thử tăng bộ nhớ - thêm bộ nhớ overhead để quản lý executor
````
spark-submit \
    --master spark://master:7077 \
    --deploy-mode client \
    --name "Spark-Standalone-1204-Test-2" \
    --conf spark.driver.host=master \
    --conf spark.driver.bindAddress=0.0.0.0 \
    --conf spark.executorEnv.PYSPARK_PYTHON=/home/ubuntu/anaconda3/bin/python \
    --conf spark.driverEnv.PYSPARK_PYTHON=/home/ubuntu/anaconda3/bin/python \
    --conf spark.hadoop.fs.defaultFS=hdfs://node1:9000 \
    --executor-memory 8G \
    --executor-cores 3 \
    --total-executor-cores 9 \
    --conf spark.executor.memoryOverhead=1536 \
    --files /home/ubuntu/huutho/DACN-Project/latest/run_1_test5.py \
    --py-files /home/ubuntu/huutho/dependencies.zip \
    ~/huutho/DACN-Project/latest/run_1_test5.py \
    /user/ubuntu/dacn/dataset \
    /user/ubuntu/dacn/processed_dataset9
```
