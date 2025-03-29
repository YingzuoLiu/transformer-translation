import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import numpy as np
import argparse
import math
import time
import os
import random
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from tqdm import tqdm

from model import Encoder, Decoder, Seq2Seq

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def initialize_weights(m):
    """初始化模型权重"""
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion, clip, device):
    """
    训练函数
    
    参数:
        model: 模型
        iterator: 数据迭代器
        optimizer: 优化器
        criterion: 损失函数
        clip: 梯度裁剪值
        device: 计算设备
    
    返回:
        epoch_loss: 本轮训练的平均损失
    """
    model.train()
    
    epoch_loss = 0
    
    pbar = tqdm(iterator, desc="Training")
    
    for batch in pbar:
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
        
        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]
        
        output_dim = output.shape[-1]
        
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        
        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # 更新进度条信息
        pbar.set_postfix(loss=loss.item())
    
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    """
    评估函数
    
    参数:
        model: 模型
        iterator: 数据迭代器
        criterion: 损失函数
        device: 计算设备
    
    返回:
        epoch_loss: 本轮评估的平均损失
    """
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            src = batch.src.to(device)
            trg = batch.trg.to(device)
            
            output, _ = model(src, trg[:,:-1])
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    """计算epoch所用时间"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def save_model(model, path):
    """保存模型"""
    torch.save(model.state_dict(), path)
    print(f"模型已保存到: {path}")

def main():
    """主函数"""
    # 命令行参数
    parser = argparse.ArgumentParser(description='训练Transformer翻译模型')
    parser.add_argument('--seed', type=int, default=1234, help='随机种子')
    parser.add_argument('--batch_size', type=int, default=128, help='批量大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--hid_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--enc_layers', type=int, default=3, help='编码器层数')
    parser.add_argument('--dec_layers', type=int, default=3, help='解码器层数')
    parser.add_argument('--enc_heads', type=int, default=8, help='编码器注意力头数')
    parser.add_argument('--dec_heads', type=int, default=8, help='解码器注意力头数')
    parser.add_argument('--enc_pf_dim', type=int, default=512, help='编码器前馈网络维度')
    parser.add_argument('--dec_pf_dim', type=int, default=512, help='解码器前馈网络维度')
    parser.add_argument('--enc_dropout', type=float, default=0.1, help='编码器dropout率')
    parser.add_argument('--dec_dropout', type=float, default=0.1, help='解码器dropout率')
    parser.add_argument('--clip', type=float, default=1.0, help='梯度裁剪值')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    parser.add_argument('--save_dir', type=str, default='models', help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载spaCy模型
    try:
        spacy_de = spacy.load('de_core_news_sm')
        spacy_en = spacy.load('en_core_web_sm')
    except OSError:
        print("正在下载spaCy模型...")
        os.system('python -m spacy download de_core_news_sm')
        os.system('python -m spacy download en_core_web_sm')
        spacy_de = spacy.load('de_core_news_sm')
        spacy_en = spacy.load('en_core_web_sm')
    
    # 定义分词器
    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]
    
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
    
    # 定义Field
    SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
    TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)
    
    # 加载数据
    print("正在加载Multi30k数据集...")
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
    
    print(f"训练集大小: {len(train_data.examples)}")
    print(f"验证集大小: {len(valid_data.examples)}")
    print(f"测试集大小: {len(test_data.examples)}")
    
    # 构建词汇表
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    
    print(f"源语言词汇表大小: {len(SRC.vocab)}")
    print(f"目标语言词汇表大小: {len(TRG.vocab)}")
    
    # 创建数据迭代器
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size=args.batch_size,
        device=device)
    
    # 构建模型
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    
    enc = Encoder(
        INPUT_DIM, 
        args.hid_dim, 
        args.enc_layers, 
        args.enc_heads, 
        args.enc_pf_dim, 
        args.enc_dropout, 
        device)
    
    dec = Decoder(
        OUTPUT_DIM, 
        args.hid_dim, 
        args.dec_layers, 
        args.dec_heads, 
        args.dec_pf_dim, 
        args.dec_dropout, 
        device)
    
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    
    # 初始化参数
    model.apply(initialize_weights)
    
    print(f"模型共有 {count_parameters(model):,} 个参数")
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    
    # 训练循环
    best_valid_loss = float('inf')
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer, criterion, args.clip, device)
        valid_loss = evaluate(model, valid_iterator, criterion, device)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # 保存最好的模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_model(model, os.path.join(args.save_dir, 'transformer-model.pt'))
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
    # 加载最好的模型并在测试集上评估
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'transformer-model.pt')))
    
    test_loss = evaluate(model, test_iterator, criterion, device)
    
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

if __name__ == "__main__":
    main()