# Docker化的Transformer翻译模型

本项目提供了一个完整的Docker化Transformer翻译模型实现，包含束搜索(Beam Search)解码算法。项目基于PyTorch和torchtext，使用Multi30k数据集训练德语到英语的翻译模型。

## 项目特点

- **完整的Transformer架构**：包含多头注意力机制、位置编码等
- **束搜索实现**：支持可配置的束宽，提升翻译质量
- **Docker化部署**：一键配置环境，无需手动安装依赖
- **注意力可视化**：理解模型的关注点
- **全面的评估工具**：包括BLEU分数计算和不同束宽的性能对比

## 快速开始

### 环境要求

- Docker
- Docker Compose
- NVIDIA GPU (推荐，但不是必需)
- NVIDIA Container Toolkit (使用GPU时需要)

### 1. 克隆项目

```bash
git clone https://github.com/YingzuoLiu/transformer-translation.git
cd transformer-translation
```

### 2. 构建Docker镜像

```bash
# 构建所有服务的Docker镜像
docker-compose build
```

### 3. 训练模型

```bash
# 使用默认参数训练模型
chmod +x run-training.sh
./run-training.sh

# 或者手动指定参数
docker-compose run --rm train python src/train.py --epochs 15 --batch_size 64
```

### 4. 评估和可视化

```bash
# 评估不同束宽的效果
chmod +x run-evaluation.sh
./run-evaluation.sh

# 运行束搜索演示
chmod +x run-beam-search.sh
./run-beam-search.sh
```

### 5. 启动Jupyter Notebook (交互式探索)

```bash
chmod +x jupyter.sh
./jupyter.sh
```

然后在浏览器中访问 `http://localhost:8888`，密码为 `transformer`。

## 项目结构

```
.
├── Dockerfile              # Docker镜像定义
├── docker-compose.yml      # Docker Compose配置
├── requirements.txt        # Python依赖
├── src/                    # 源代码
│   ├── model.py            # Transformer模型定义
│   ├── beam_search.py      # 束搜索实现
│   ├── train.py            # 训练脚本
│   └── evaluate.py         # 评估脚本
├── models/                 # 保存训练好的模型
├── data/                   # 数据目录
└── results/                # 结果和可视化
    ├── attention_maps/     # 注意力可视化
    └── bleu_scores/        # BLEU分数评估结果
```

## 束搜索参数调整

束搜索是一种在解码阶段保留多个候选路径的算法，可以显著提升翻译质量。在本项目中，可以通过以下方式调整束搜索参数：

```bash
# 在评估时比较不同束宽(1,3,5,10)的效果
docker-compose run --rm evaluate python src/evaluate.py --beam_sizes "1 3 5 10" 

# 使用特定束宽进行翻译演示
docker-compose run --rm transformer python src/beam_search.py --beam_size 5
```

## 自定义模型参数

您可以通过修改`train.py`的命令行参数来自定义Transformer模型：

```bash
docker-compose run --rm train python src/train.py \
    --epochs 20 \
    --batch_size 64 \
    --hid_dim 512 \
    --enc_layers 6 \
    --dec_layers 6 \
    --enc_heads 8 \
    --dec_heads 8
```

## 结果示例

训练完成后，您可以查看以下结果：

1. **BLEU分数**：在`results/bleu_scores/`目录
2. **注意力可视化**：在`results/attention_maps/`目录
3. **训练日志**：在`logs/`目录

## 扩展方向

1. **添加更多语言对**：修改数据加载代码，支持其他语言对的翻译
2. **长句子处理**：改进模型以更好地处理长句子
3. **集成更先进的解码策略**：如长度惩罚、覆盖惩罚等
4. **整合预训练模型**：如mBART、T5等

## 疑难解答

1. **CUDA问题**：
   - 确保已安装NVIDIA Container Toolkit
   - 检查`docker-compose.yml`中的GPU设置

2. **内存不足**：
   - 减小批处理大小：修改`--batch_size`参数
   - 减小模型尺寸：减少层数或隐藏层维度

3. **找不到预训练模型**：
   - 确保已完成训练步骤
   - 检查`models/`目录中是否存在`transformer-model.pt`文件

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原始论文
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Transformer可视化解释
- [PyTorch教程：语言翻译与Transformer](https://pytorch.org/tutorials/beginner/translation_transformer.html)