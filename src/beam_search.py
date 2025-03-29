import torch
import torch.nn.functional as F
import spacy
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time
import os
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from model import Encoder, Decoder, Seq2Seq

def beam_search(model, src, src_mask, max_len, beam_size, device, sos_idx, eos_idx):
    """
    对输入序列src执行束搜索翻译
    
    参数:
        model: 训练好的序列到序列模型
        src: 源语言句子张量 [1, src_len]
        src_mask: 源语言掩码 [1, 1, 1, src_len]
        max_len: 生成的最大长度
        beam_size: 束宽，即保留的候选路径数量
        device: 计算设备
        sos_idx: 起始符的索引
        eos_idx: 结束符的索引
        
    返回:
        最佳翻译序列和注意力权重
    """
    # 获取编码器输出
    with torch.no_grad():
        enc_src = model.encoder(src, src_mask)  # [1, src_len, hid_dim]
    
    # 候选序列记录，每个序列包含 [token_ids, score, finished_flag]
    beams = [(torch.LongTensor([sos_idx]).to(device), 0.0, False)]
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
            topk_probs, topk_idxs = torch.topk(F.log_softmax(pred, dim=-1), beam_size)
            
            # 为每个高概率词创建新的候选
            for prob, idx in zip(topk_probs[0], topk_idxs[0]):
                # 计算新候选的得分
                new_score = score + prob.item()
                # 创建新序列
                new_seq = torch.cat([seq, torch.LongTensor([idx]).to(device)], dim=0)
                # 检查是否生成结束符
                is_complete = (idx == eos_idx)
                
                all_candidates.append((new_seq, new_score, is_complete))
        
        # 保留得分最高的beam_size个候选
        # 对于已完成的序列，将其与未完成的序列分开处理
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
    
    return best_seq, attention

def translate_sentence_beam_search(sentence, src_field, trg_field, model, device, max_len=50, beam_size=5):
    """
    使用束搜索翻译一个句子
    
    参数:
        sentence: 源语言句子（字符串或token列表）
        src_field: 源语言Field对象
        trg_field: 目标语言Field对象
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
        nlp = spacy.load('de_core_news_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    
    # 添加起始和结束标记
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    
    # 转换为索引
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    
    # 转换为张量
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    
    # 获取起始和结束标记的索引
    sos_idx = trg_field.vocab.stoi[trg_field.init_token]
    eos_idx = trg_field.vocab.stoi[trg_field.eos_token]
    
    # 执行束搜索
    translation_tensor, attention = beam_search(
        model, src_tensor, src_mask, max_len, beam_size, device, sos_idx, eos_idx
    )
    
    # 转换回词汇
    translation = [trg_field.vocab.itos[idx] for idx in translation_tensor]
    
    # 移除起始标记（如果存在）并在遇到结束标记时停止
    if translation[0] == trg_field.init_token:
        translation = translation[1:]
    
    end_idx = translation.index(trg_field.eos_token) if trg_field.eos_token in translation else len(translation)
    translation = translation[:end_idx]
    
    return translation, attention

def display_attention(sentence, translation, attention, src_field, trg_field, n_heads=8, n_rows=4, n_cols=2, save_path=None):
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
        nlp = spacy.load('de_core_news_sm')
        src_tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        src_tokens = [token.lower() for token in sentence]
        
    src_tokens = [src_field.init_token] + src_tokens + [src_field.eos_token]
    trg_tokens = [trg_field.init_token] + translation + [trg_field.eos_token]
    
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
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        
        ax.set_title(f'Head {i+1}')
        
    plt.tight_layout()
    plt.colorbar(cax, ax=ax)
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力图已保存到: {save_path}")
    
    plt.show()

def compare_beam_sizes(sentence, src_field, trg_field, model, device, max_len=50, beam_sizes=[1, 3, 5, 10]):
    """
    比较不同束宽对翻译结果的影响
    
    参数:
        sentence: 源语言句子
        src_field: 源语言Field对象
        trg_field: 目标语言Field对象
        model: 训练好的模型
        device: 计算设备
        max_len: 生成的最大长度
        beam_sizes: 要比较的束宽列表
    """
    print(f"原句: {' '.join([token.lower() for token in sentence])}")
    
    results = {}
    
    for beam_size in beam_sizes:
        start_time = time.time()
        translation, _ = translate_sentence_beam_search(
            sentence, src_field, trg_field, model, device, max_len, beam_size
        )
        end_time = time.time()
        translation_time = end_time - start_time
        
        print(f"束宽 = {beam_size}, 翻译时间: {translation_time:.2f}秒")
        print(f"翻译结果: {' '.join(translation)}")
        print("-" * 50)
        
        results[beam_size] = {
            'translation': ' '.join(translation),
            'time': translation_time
        }
    
    return results

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

def translate_demo(model, SRC, TRG, test_data, device, beam_size=5, max_len=50, save_attention=True):
    """演示翻译样例"""
    
    # 选择一个测试样例
    example_idx = 10
    src = vars(test_data.examples[example_idx])['src']
    trg = vars(test_data.examples[example_idx])['trg']
    
    print(f"源句: {' '.join(src)}")
    print(f"参考译文: {' '.join(trg)}")
    
    # 使用束搜索翻译
    translation, attention = translate_sentence_beam_search(
        src, SRC, TRG, model, device, max_len, beam_size
    )
    
    print(f"束搜索翻译 (束宽={beam_size}): {' '.join(translation)}")
    
    # 可视化注意力
    if save_attention:
        os.makedirs('results/attention_maps', exist_ok=True)
        save_path = f'results/attention_maps/attention_beam{beam_size}.png'
        display_attention(src, translation, attention, SRC, TRG, n_heads=4, n_rows=2, n_cols=2, save_path=save_path)
    else:
        display_attention(src, translation, attention, SRC, TRG, n_heads=4, n_rows=2, n_cols=2)
    
    # 比较不同束宽
    print("\n比较不同束宽的结果:")
    compare_beam_sizes(src, SRC, TRG, model, device, beam_sizes=[1, 3, 5, 7, 10])

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='使用束搜索进行翻译')
    parser.add_argument('--model_path', type=str, default='models/transformer-model.pt', help='模型路径')
    parser.add_argument('--beam_size', type=int, default=5, help='束宽')
    parser.add_argument('--max_len', type=int, default=50, help='最大生成长度')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--compare', action='store_true', help='比较不同束宽')
    parser.add_argument('--save_attention', action='store_true', help='保存注意力图')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    
    # 加载模型和数据
    print(f"正在加载模型: {args.model_path}")
    model, SRC, TRG, test_data = load_model_and_data(args.model_path, device)
    
    # 演示翻译
    translate_demo(model, SRC, TRG, test_data, device, args.beam_size, args.max_len, args.save_attention)

if __name__ == "__main__":
    main()