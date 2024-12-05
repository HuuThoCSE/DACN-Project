# COMMAND
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