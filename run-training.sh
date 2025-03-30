mkdir -p logs

# 训练模型
echo "开始训练模型..."
docker-compose run --rm train | tee logs/training.log

echo "训练完成！"
