mkdir -p logs

# 下载数据集
echo "开始下载Multi30k数据集..."
docker-compose run --rm download | tee logs/download.log

echo "数据集下载完成！"