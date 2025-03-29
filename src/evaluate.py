import torch
import torch.nn as nn
import spacy
import argparse
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
from tqdm import tqdm

from model import Encoder, Decoder, Seq2Seq
from beam_search import translate_sentence_beam_search, display_attention

def calculate_bleu(data, src_field, trg_field, model, device, beam_size=1, max_len=50, num_examples=None):
    """
    计算BLEU分数
    
    参数:
        data: 测试数据
        src_field: 源语言Field对象
        trg_field: 目标语言Field对象
        model: 训练好的模型
        device: 计算设备
        beam_size: 束宽
        max_len: 最大生成长度
        num_examples: 评估的样例数量，默认为全部
        
    返回:
        bleu: BLEU分数
    """
    model.eval()
    
    if num_examples is None:
        examples = data.examples
    else:
        examples = data.examples[:num_examples]
    
    hypotheses = []
    references = []
    
    print(f"使用束宽 {beam_size} 计算BLEU分数...")
    
    for example in tqdm(examples):
        src = vars(example)['src']
        trg = vars(example)['trg']
        
        # 使用束搜索翻译
        translation, _ = translate_sentence_beam_search(
            src, src_field, trg_field, model, device, max_len, beam_size
        )
        
        # 添加到列表
        hypotheses.append(translation)
        references.append([trg])
    
    # 计算BLEU分数
    bleu = bleu_score(hypotheses, references)
    
    return bleu

def evaluate_beam_sizes(test_data, src_field, trg_field, model, device, beam_sizes=[1, 3, 5, 10], max_len=50, num_examples=100):
    """
    评估不同束宽的BLEU分数
    
    参数:
        test_data: 测试数据
        src_field: 源语言Field对象
        trg_field: 目标语言Field对象
        model: 训练好的模型
        device: 计算设备
        beam_sizes: 要评估的束宽列表
        max_len: 最大生成长度
        num_examples: 评估的样例数量
        
    返回:
        results: 各束宽对应的BLEU分数和计算时间
    """
    results = {}
    
    for beam_size in beam_sizes:
        start_time = time.time()
        bleu = calculate_bleu(test_data, src_field, trg_field, model, device, beam_size, max_len, num_examples)
        end_time = time.time()
        
        evaluation_time = end_time - start_time
        avg_time_per_example = evaluation_time / num_examples
        
        print(f"束宽 {beam_size} 的BLEU分数: {bleu:.4f}")
        print(f"评估 {num_examples} 个样例用时: {evaluation_time:.2f}秒 (平均每个样例 {avg_time_per_example:.2f}秒)")
        print("-" * 50)
        
        results[beam_size] = {
            'bleu': bleu,
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

def visualize_sample_translations(model, test_data, src_field, trg_field, device, num_examples=3, beam_size=5, max_len=50, save_dir='results/attention_maps'):
    """可视化样例翻译及其注意力"""
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_examples):
        example_idx = i * 10  # 选择不同的样例
        src = vars(test_data.examples[example_idx])['src']
        trg = vars(test_data.examples[example_idx])['trg']
        
        print(f"\n样例 {i+1}:")
        print(f"源句: {' '.join(src)}")
        print(f"参考译文: {' '.join(trg)}")
        
        # 使用束搜索翻译
        translation, attention = translate_sentence_beam_search(
            src, src_field, trg_field, model, device, max_len, beam_size
        )
        
        print(f"束搜索翻译 (束宽={beam_size}): {' '.join(translation)}")
        
        # 可视化注意力
        save_path = os.path.join(save_dir, f'attention_sample{i+1}_beam{beam_size}.png')
        display_attention(src, translation, attention, src_field, trg_field, 
                        n_heads=4, n_rows=2, n_cols=2, save_path=save_path)

def load_model_and_data(model_path, device):
    """加载预训练模型和数据"""
    
    # 定义分词器
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
    
    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]
    
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
    
    # 定义Field
    SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
    TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)
    
    # 加载数据
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
    
    # 构建词汇表
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    
    # 加载模型
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    
    enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
    dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)
    
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    
    # 加载预训练权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, SRC, TRG, test_data

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
    parser.add_argument('--test_data', type=str, default='Multi30k', help='测试数据集')
    
    args = parser.parse_args()
    
    # 将束宽字符串转换为列表
    beam_sizes = [int(x) for x in args.beam_sizes.split()]
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载模型和数据
    print(f"正在加载模型: {args.model_path}")
    model, SRC, TRG, test_data = load_model_and_data(args.model_path, device)
    
    # 评估不同束宽的BLEU分数
    results = evaluate_beam_sizes(
        test_data, SRC, TRG, model, device, 
        beam_sizes=beam_sizes, 
        max_len=args.max_len, 
        num_examples=args.num_examples
    )
    
    # 绘制束宽与BLEU分数的关系图
    plot_bleu_vs_beam_size(results)
    
    # 可视化样例翻译
    if args.visualize:
        visualize_sample_translations(model, test_data, SRC, TRG, device)

if __name__ == "__main__":
    main()