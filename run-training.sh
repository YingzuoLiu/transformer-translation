mkdir -p logs

# 评估模型
echo "开始评估模型..."
docker-compose run --rm evaluate | tee logs/evaluation.log

echo "评估完成！"
