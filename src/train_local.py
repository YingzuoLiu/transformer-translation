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
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from model import Encoder, Decoder, Seq2Seq

# 定义特殊标记
SRC_LANGUAGE = 'de'
TRG_LANGUAGE = 'en'
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']

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

def tensor_transform(token_ids):
    """添加BOS/EOS并创建tensor"""
    return torch.cat((torch.tensor([BOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

class Vocab:
    """简单的词汇表类"""
    def __init__(self, specials=None):
        self.stoi = {}  # string to index
        self.itos = []  # index to string
        self.freqs = {}  # token frequencies
        
        # 添加特殊标记
        if specials:
            for i, token in enumerate(specials):
                self.stoi[token] = i
                self.itos.append(token)
    
    def add_token(self, token):
        """添加标记到词汇表"""
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
            self.freqs[token] = 1
        else:
            self.freqs[token] += 1
    
    def __getitem__(self, token):
        """获取标记的索引"""
        return self.stoi.get(token, self.stoi.get('<unk>', 0))
    
    def __len__(self):
        """词汇表大小"""
        return len(self.itos)
    
    def get_itos(self):
        """获取索引到字符串的映射"""
        return self.itos

class TranslationDataset(Dataset):
    """翻译数据集"""
    def __init__(self, src_file, trg_file):
        self.src_sentences = self.read_file(src_file)
        self.trg_sentences = self.read_file(trg_file)
        
        assert len(self.src_sentences) == len(self.trg_sentences), \
            "源语言和目标语言文件行数不匹配"
    
    def read_file(self, file_path):
        """读取文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        trg = self.trg_sentences[idx]
        
        # 对于已经标记化的文件，直接分割
        src_tokens = src.split()
        trg_tokens = trg.split()
        
        return src_tokens, trg_tokens

def build_vocab_from_dataset(dataset, index=0, min_freq=2):
    """从数据集构建词汇表
    
    参数:
        dataset: 数据集对象
        index: 0表示源语言，1表示目标语言
        min_freq: 最小频率阈值
    """
    vocab = Vocab(SPECIAL_SYMBOLS)
    
    # 构建词频字典
    token_freq = {}
    
    for i in range(len(dataset)):
        tokens = dataset[i][index]  # 获取对应语言的tokens
        for token in tokens:
            if token not in token_freq:
                token_freq[token] = 1
            else:
                token_freq[token] += 1
    
    # 添加频率大于min_freq的标记
    for token, freq in token_freq.items():
        if freq >= min_freq:
            vocab.add_token(token)
    
    return vocab

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
    
    for src, trg in pbar:
        src = src.to(device)
        trg = trg.to(device)
        
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
        for src, trg in tqdm(iterator, desc="Evaluating"):
            src = src.to(device)
            trg = trg.to(device)
            
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
    parser.add_argument('--data_dir', type=str, default='data/multi30k', help='数据目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 文件路径
    train_src_file = os.path.join(args.data_dir, 'train.de-en.de')
    train_trg_file = os.path.join(args.data_dir, 'train.de-en.en')
    val_src_file = os.path.join(args.data_dir, 'val.de-en.de')
    val_trg_file = os.path.join(args.data_dir, 'val.de-en.en')
    test_src_file = os.path.join(args.data_dir, 'test.de-en.de')
    test_trg_file = os.path.join(args.data_dir, 'test.de-en.en')
    
    # 创建数据集
    print("加载数据集...")
    train_dataset = TranslationDataset(train_src_file, train_trg_file)
    val_dataset = TranslationDataset(val_src_file, val_trg_file)
    test_dataset = TranslationDataset(test_src_file, test_trg_file)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 构建词汇表
    print("构建词汇表...")
    vocab_src = build_vocab_from_dataset(train_dataset, index=0, min_freq=2)
    vocab_trg = build_vocab_from_dataset(train_dataset, index=1, min_freq=2)
    
    print(f"源语言词汇表大小: {len(vocab_src)}")
    print(f"目标语言词汇表大小: {len(vocab_trg)}")
    
    # 数据处理
    def collate_fn(batch):
        src_batch, trg_batch = [], []
        for src_tokens, trg_tokens in batch:
            # 数字化
            src_ids = [vocab_src[token] for token in src_tokens]
            trg_ids = [vocab_trg[token] for token in trg_tokens]
            
            # 添加BOS/EOS
            src_tensor = tensor_transform(src_ids)
            trg_tensor = tensor_transform(trg_ids)
            
            src_batch.append(src_tensor)
            trg_batch.append(trg_tensor)
            
        # 填充序列
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX)
        
        # 转置，使批大小在第一维
        return src_batch.transpose(0, 1), trg_batch.transpose(0, 1)
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                 shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_fn)
    
    # 构建模型
    enc = Encoder(
        len(vocab_src), 
        args.hid_dim, 
        args.enc_layers, 
        args.enc_heads, 
        args.enc_pf_dim, 
        args.enc_dropout, 
        device)
    
    dec = Decoder(
        len(vocab_trg), 
        args.hid_dim, 
        args.dec_layers, 
        args.dec_heads, 
        args.dec_pf_dim, 
        args.dec_dropout, 
        device)
    
    model = Seq2Seq(enc, dec, PAD_IDX, PAD_IDX, device).to(device)
    
    # 初始化参数
    model.apply(initialize_weights)
    
    print(f"模型共有 {count_parameters(model):,} 个参数")
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # 训练循环
    best_valid_loss = float('inf')
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss = train(model, train_dataloader, optimizer, criterion, args.clip, device)
        valid_loss = evaluate(model, val_dataloader, criterion, device)
        
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
    
    test_loss = evaluate(model, test_dataloader, criterion, device)
    
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

if __name__ == "__main__":
    main()