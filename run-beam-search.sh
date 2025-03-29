mkdir -p logs

# 运行束搜索演示
echo "开始束搜索演示..."
docker-compose run --rm transformer python src/beam_search.py --beam_size 5 --save_attention | tee logs/beam_search.log

echo "演示完成！"
