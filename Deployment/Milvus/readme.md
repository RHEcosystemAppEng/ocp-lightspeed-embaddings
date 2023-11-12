
https://milvus.io/docs/install_standalone-gpu-helm.md




$ helm install lightspeed-milvus milvus/milvus --set cluster.enabled=false --set etcd.replicaCount=1 --set minio.mode=standalone --set pulsar.enabled=false -f custom-values.yaml

