import torch
import torch.nn as nn
import spacy
import argparse
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sacrebleu.metrics import BLEU
from tqdm import tqdm

from model import Encoder, Decoder, Seq2Seq

# 定义特殊标记
SRC_LANGUAGE = 'de'
TRG_LANGUAGE = 'en'
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']

def sequential_transforms(*transforms):
    """将多个变换函数组合在一起"""
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids):
    """添加BOS/EOS并创建tensor"""
    return torch.cat((torch.tensor([BOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

def load_tokenizers():
    """加载分词器"""
    try:
        spacy_de = spacy.load("de_core_news_sm")
        spacy_en = spacy.load("en_core_web_sm")
    except OSError:
        # 如果spaCy模型不存在，尝试下载
        os.system('python -m spacy download de_core_news_sm')
        os.system('python -m spacy download en_core_web_sm')
        spacy_de = spacy.load("de_core_news_sm")
        spacy_en = spacy.load("en_core_web_sm")
    
    # 定义德语和英语分词器
    tokenizer_de = get_tokenizer(lambda text: [tok.text for tok in spacy_de.tokenizer(text)], language='de')
    tokenizer_en = get_tokenizer(lambda text: [tok.text for tok in spacy_en.tokenizer(text)], language='en')
    
    return tokenizer_de, tokenizer_en

def yield_tokens(data_iter, tokenizer, index):
    """生成器函数，从数据迭代器中产生标记"""
    for data_sample in data_iter:
        yield tokenizer(data_sample[index].lower())

def build_vocab(tokenizer, dataset, language):
    """构建词汇表"""
    # 根据语言选择索引
    index = 0 if language == SRC_LANGUAGE else 1
    
    # 构建词汇表
    vocab = build_vocab_from_iterator(
        yield_tokens(dataset, tokenizer, index),
        min_freq=2,
        specials=SPECIAL_SYMBOLS,
        special_first=True
    )
    
    # 设置默认索引
    vocab.set_default_index(UNK_IDX)
    
    return vocab

def create_datasets(tokenizer_src, tokenizer_trg, vocab_src, vocab_trg):
    """创建数据集处理管道"""
    # 源语言变换
    text_transform_src = sequential_transforms(
        tokenizer_src,  # 分词
        lambda x: [vocab_src[token] for token in x],  # 数字化
        tensor_transform  # 添加BOS/EOS并转换为tensor
    )
    
    # 目标语言变换
    text_transform_trg = sequential_transforms(
        tokenizer_trg,  # 分词
        lambda x: [vocab_trg[token] for token in x],  # 数字化
        tensor_transform  # 添加BOS/EOS并转换为tensor
    )
    
    # 使用Multi30k数据集
    train_iter, valid_iter, test_iter = Multi30k(split=('train', 'valid', 'test'),
                                               language_pair=(SRC_LANGUAGE, TRG_LANGUAGE))
    
    # 数据处理函数
    def collate_fn(batch):
        src_batch, trg_batch = [], []
        for src_sample, trg_sample in batch:
            src_batch.append(text_transform_src(src_sample))
            trg_batch.append(text_transform_trg(trg_sample))
            
        # 填充序列
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX)
        
        return src_batch, trg_batch
    
    return train_iter, valid_iter, test_iter, collate_fn

def translate_sentence_beam_search(sentence, tokenizer_src, vocab_src, vocab_trg, model, device, max_len=50, beam_size=5):
    """
    使用束搜索翻译一个句子
    
    参数:
        sentence: 源语言句子（字符串或token列表）
        tokenizer_src: 源语言分词器
        vocab_src: 源语言词汇表
        vocab_trg: 目标语言词汇表
        model: 训练好的模型
        device: 计算设备
        max_len: 生成的最大长度
        beam_size: 束宽
        
    返回:
        翻译后的句子和注意力权重
    """
    model.eval()
    
    # 处理输入句子
    if isinstance(sentence, str):
        tokens = tokenizer_src(sentence.lower())
    else:
        tokens = [token.lower() for token in sentence]
    
    # 数字化
    token_ids = [vocab_src[token] for token in tokens]
    
    # 添加BOS/EOS
    token_ids = [BOS_IDX] + token_ids + [EOS_IDX]
    
    # 转换为张量
    src_tensor = torch.LongTensor(token_ids).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    
    # 获取编码器输出
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    
    # 候选序列记录，每个序列包含 [token_ids, score, finished_flag]
    beams = [(torch.LongTensor([BOS_IDX]).to(device), 0.0, False)]
    complete_beams = []
    
    # 在最大长度内继续生成
    for _ in range(max_len):
        # 如果所有候选都已完成，则提前结束
        if all(beam[2] for beam in beams):
            break
            
        # 扩展候选
        all_candidates = []
        
        # 扩展每个候选序列
        for seq, score, is_finished in beams:
            # 如果序列已经完成则不再扩展
            if is_finished:
                all_candidates.append((seq, score, True))
                continue
                
            # 准备解码器输入
            trg_tensor = seq.unsqueeze(0)  # [1, seq_len]
            trg_mask = model.make_trg_mask(trg_tensor)  # [1, 1, seq_len, seq_len]
            
            # 通过解码器获取预测
            with torch.no_grad():
                output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                # output: [1, seq_len, output_dim]
                
            # 获取最后一个时间步的预测
            pred = output[:, -1, :]  # [1, output_dim]
            
            # 获取前beam_size个高概率的词
            topk_probs, topk_idxs = torch.topk(torch.log_softmax(pred, dim=-1), beam_size)
            
            # 为每个高概率词创建新的候选
            for prob, idx in zip(topk_probs[0], topk_idxs[0]):
                # 计算新候选的得分
                new_score = score + prob.item()
                # 创建新序列
                new_seq = torch.cat([seq, torch.LongTensor([idx]).to(device)], dim=0)
                # 检查是否生成结束符
                is_complete = (idx == EOS_IDX)
                
                all_candidates.append((new_seq, new_score, is_complete))
        
        # 保留得分最高的beam_size个候选
        incomplete_candidates = [c for c in all_candidates if not c[2]]
        complete_candidates = [c for c in all_candidates if c[2]]
        
        # 将完成的候选添加到最终列表
        complete_beams.extend(complete_candidates)
        
        # 如果所有候选都已完成，提前结束
        if len(incomplete_candidates) == 0:
            break
            
        # 根据得分对未完成的候选排序，选择前beam_size个
        beams = sorted(incomplete_candidates, key=lambda x: x[1] / len(x[0]), reverse=True)[:beam_size]
    
    # 如果有完成的候选，从中选择得分最高的
    if complete_beams:
        # 标准化得分（按长度归一化，避免偏向短序列）
        complete_beams = [(seq, score / len(seq), flag) for seq, score, flag in complete_beams]
        best_seq, best_score, _ = max(complete_beams, key=lambda x: x[1])
    else:
        # 如果没有完成的候选，从未完成的中选择最好的
        best_seq, best_score, _ = max(beams, key=lambda x: x[1] / len(x[0]))
    
    # 再次前向传播以获取注意力权重
    trg_tensor = best_seq.unsqueeze(0)
    trg_mask = model.make_trg_mask(trg_tensor)
    
    with torch.no_grad():
        _, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
    
    # 转换回词汇
    translation = [vocab_trg.get_itos()[idx] for idx in best_seq]
    
    # 移除特殊标记并在遇到结束标记时停止
    if translation[0] == '<bos>':
        translation = translation[1:]
    
    if '<eos>' in translation:
        end_idx = translation.index('<eos>')
        translation = translation[:end_idx]
    
    return translation, attention

def display_attention(sentence, translation, attention, tokenizer_src, n_heads=8, n_rows=4, n_cols=2, save_path=None):
    """
    可视化注意力权重
    
    参数:
        sentence: 源语言句子
        translation: 翻译后的句子
        attention: 注意力权重
        n_heads: 要显示的注意力头数量
        n_rows, n_cols: 图表的行数和列数
        save_path: 保存图像的路径
    """
    assert n_rows * n_cols >= n_heads
    
    # 准备源语言和目标语言的标记
    if isinstance(sentence, str):
        src_tokens = tokenizer_src(sentence.lower())
    else:
        src_tokens = [token.lower() for token in sentence]
        
    src_tokens = ['<bos>'] + src_tokens + ['<eos>']
    trg_tokens = ['<bos>'] + translation + ['<eos>']
    
    # 选择要显示的注意力头
    attention = attention.squeeze(0).cpu().detach().numpy()
    
    # 创建图表
    fig = plt.figure(figsize=(16, 16))
    
    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        # 获取当前注意力头的权重
        _attention = attention[i]
        
        # Plot heatmap
        cax = ax.matshow(_attention, cmap='viridis')
        
        # Set labels
        ax.set_xticklabels([''] + src_tokens, rotation=90)
        ax.set_yticklabels([''] + trg_tokens)
        
        # 设置坐标轴刻度
        ax.xaxis.set_major_locator(plt.ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.ticker.MultipleLocator(1))
        
        ax.set_title(f'Head {i+1}')
        
    plt.tight_layout()
    plt.colorbar(cax, ax=ax)
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力图已保存到: {save_path}")
    
    plt.show()

def evaluate_beam_sizes(test_data, tokenizer_src, vocab_src, vocab_trg, model, device, beam_sizes=[1, 3, 5, 10], max_len=50, num_examples=100):
    """
    评估不同束宽的BLEU分数
    
    参数:
        test_data: 测试数据迭代器
        tokenizer_src: 源语言分词器
        vocab_src: 源语言词汇表
        vocab_trg: 目标语言词汇表
        model: 训练好的模型
        device: 计算设备
        beam_sizes: 要评估的束宽列表
        max_len: 最大生成长度
        num_examples: 评估的样例数量
        
    返回:
        results: 各束宽对应的BLEU分数和计算时间
    """
    results = {}
    
    # 初始化BLEU计算器
    bleu = BLEU(effective_order=True)
    
    # 准备测试数据
    test_samples = list(test_data)[:num_examples]
    
    for beam_size in beam_sizes:
        start_time = time.time()
        
        hypotheses = []
        references = []
        
        print(f"使用束宽 {beam_size} 计算BLEU分数...")
        
        for src_sample, trg_sample in tqdm(test_samples):
            # 翻译源句子
            translation, _ = translate_sentence_beam_search(
                src_sample, tokenizer_src, vocab_src, vocab_trg, model, device, max_len, beam_size
            )
            
            # 添加到列表
            hypotheses.append(' '.join(translation))
            references.append([' '.join(trg_sample.split())])
        
        # 计算BLEU分数
        bleu_score = bleu.corpus_score(hypotheses, references).score
        
        end_time = time.time()
        evaluation_time = end_time - start_time
        avg_time_per_example = evaluation_time / len(test_samples)
        
        print(f"束宽 {beam_size} 的BLEU分数: {bleu_score:.4f}")
        print(f"评估 {len(test_samples)} 个样例用时: {evaluation_time:.2f}秒 (平均每个样例 {avg_time_per_example:.2f}秒)")
        print("-" * 50)
        
        results[beam_size] = {
            'bleu': bleu_score,
            'total_time': evaluation_time,
            'avg_time': avg_time_per_example
        }
    
    return results

def plot_bleu_vs_beam_size(results, save_path='results/bleu_scores/bleu_vs_beam_size.png'):
    """绘制束宽与BLEU分数的关系图"""
    beam_sizes = list(results.keys())
    bleu_scores = [results[k]['bleu'] for k in beam_sizes]
    avg_times = [results[k]['avg_time'] for k in beam_sizes]
    
    fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()
    
    # BLEU分数曲线
    color = 'tab:blue'
    ax1.set_xlabel('束宽', fontsize=14)
    ax1.set_ylabel('BLEU分数', color=color, fontsize=14)
    ax1.plot(beam_sizes, bleu_scores, marker='o', linestyle='-', linewidth=2, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 添加数据标签
    for i, (beam_size, bleu) in enumerate(zip(beam_sizes, bleu_scores)):
        ax1.annotate(f'{bleu:.4f}', (beam_size, bleu), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=10)
    
    # 时间曲线
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('平均时间 (秒/样例)', color=color, fontsize=14)
    ax2.plot(beam_sizes, avg_times, marker='s', linestyle='-', linewidth=2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 添加数据标签
    for i, (beam_size, time) in enumerate(zip(beam_sizes, avg_times)):
        ax2.annotate(f'{time:.2f}s', (beam_size, time), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=10)
    
    plt.title('束宽对翻译质量和速度的影响', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 创建目录并保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")
    
    plt.show()

def load_model(model_path, vocab_src_size, vocab_trg_size, device):
    """加载预训练模型"""
    
    # 模型参数
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    
    # 创建编码器和解码器
    enc = Encoder(vocab_src_size, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
    dec = Decoder(vocab_trg_size, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)
    
    # 创建Seq2Seq模型
    model = Seq2Seq(enc, dec, PAD_IDX, PAD_IDX, device).to(device)
    
    # 加载预训练权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def main():
    """主函数"""
    # 命令行参数
    parser = argparse.ArgumentParser(description='评估翻译模型')
    parser.add_argument('--model_path', type=str, default='models/transformer-model.pt', help='模型路径')
    parser.add_argument('--beam_sizes', type=str, default='1 3 5 10', help='要评估的束宽（空格分隔）')
    parser.add_argument('--max_len', type=int, default=50, help='最大生成长度')
    parser.add_argument('--num_examples', type=int, default=100, help='评估的样例数量')
    parser.add_argument('--visualize', action='store_true', help='可视化翻译和注意力')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    
    args = parser.parse_args()
    
    # 将束宽字符串转换为列表
    beam_sizes = [int(x) for x in args.beam_sizes.split()]
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载分词器和构建词汇表
    print("加载分词器...")
    tokenizer_src, tokenizer_trg = load_tokenizers()
    
    # 加载数据集
    print("加载数据集...")
    train_iter, valid_iter, test_iter, _ = Multi30k(split=('train', 'valid', 'test'),
                                                  language_pair=(SRC_LANGUAGE, TRG_LANGUAGE))
    
    # 构建词汇表
    print("构建词汇表...")
    vocab_src = build_vocab(tokenizer_src, train_iter, SRC_LANGUAGE)
    vocab_trg = build_vocab(tokenizer_trg, train_iter, TRG_LANGUAGE)
    
    print(f"源语言词汇表大小: {len(vocab_src)}")
    print(f"目标语言词汇表大小: {len(vocab_trg)}")
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = load_model(args.model_path, len(vocab_src), len(vocab_trg), device)
    
    # 评估不同束宽的BLEU分数
    print("开始评估...")
    results = evaluate_beam_sizes(
        test_iter, tokenizer_src, vocab_src, vocab_trg, model, device, 
        beam_sizes=beam_sizes, 
        max_len=args.max_len, 
        num_examples=args.num_examples
    )
    
    # 绘制束宽与BLEU分数的关系图
    plot_bleu_vs_beam_size(results)
    
    # 可视化样例翻译
    if args.visualize:
        print("可视化样例翻译...")
        test_samples = list(test_iter)[:3]  # 取前3个样例
        
        for i, (src_sample, trg_sample) in enumerate(test_samples):
            print(f"\n样例 {i+1}:")
            print(f"源句: {src_sample}")
            print(f"参考译文: {trg_sample}")
            
            # 翻译
            translation, attention = translate_sentence_beam_search(
                src_sample, tokenizer_src, vocab_src, vocab_trg, model, device, 
                max_len=args.max_len, beam_size=5
            )
            
            print(f"束搜索翻译 (束宽=5): {' '.join(translation)}")
            
            # 可视化注意力
            save_path = f'results/attention_maps/attention_sample{i+1}_beam5.png'
            os.makedirs('results/attention_maps', exist_ok=True)
            display_attention(src_sample, translation, attention, tokenizer_src, 
                            n_heads=4, n_rows=2, n_cols=2, save_path=save_path)

if __name__ == "__main__":
    main()