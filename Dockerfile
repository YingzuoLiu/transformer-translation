# 使用官方PyTorch镜像作为基础镜像
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 安装基本工具和依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 复制项目需求文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 下载spaCy语言模型
RUN python -m spacy download de_core_news_sm && \
    python -m spacy download en_core_web_sm

# 复制项目文件
COPY . .

# 创建数据、模型和结果目录
RUN mkdir -p data models results/attention_maps results/bleu_scores

# 设置环境变量
ENV PYTHONPATH="${PYTHONPATH}:/app"

# 默认命令
CMD ["bash"]