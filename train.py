import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.optim as optim  # 导入 PyTorch 的优化器模块
from torch.utils.data import Dataset, DataLoader  # 导入 PyTorch 的数据集和数据加载器模块
from torch.nn.utils.rnn import pad_sequence  # 导入 PyTorch 的序列填充工具
from collections import Counter  # 导入 collections 库中的 Counter 类
import jieba  # 导入 jieba 库用于中文分词
import spacy  # 导入 spacy 库用于英文分词
import os  # 导入 os 库用于文件和目录操作
import pandas as pd  # 导入 pandas 库用于数据处理
import transformer

# 1. 数据读取和合并
folder_path = "Dataset"  # 替换为你的数据文件夹路径
parquet_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet')]  # 获取所有 parquet 文件的路径

# 将所有 parquet 文件加载并合并为一个 DataFrame
df_list = [pd.read_parquet(file) for file in parquet_files]  # 读取每个 parquet 文件并存储在列表中
df = pd.concat(df_list, ignore_index=True)  # 将所有 DataFrame 合并为一个
df = df.head(50) # 仅使用前 50 行数据进行训练

# 检查合并后的数据
print("合并后的数据集大小:", df.shape)  # 打印合并后的数据集大小
print(df.head())  # 打印合并后的数据集的前几行

# 将 `translation` 列的内容拆分成两列：`en` 和 `zh`
df[['en', 'zh']] = df['translation'].apply(pd.Series)  # 将 translation 列拆分为 en 和 zh 列
df = df.drop(columns=['translation'])  # 删除不再需要的 translation 列
print(df[['en', 'zh']].head())  # 打印 en 和 zh 列的前几行

# 加载英文分词器
spacy_en = spacy.load("en_core_web_sm", disable=["ner"])  # 加载 spacy 的英文分词器

# 数据处理部分
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]  # 使用 spacy 分词器对英文文本进行分词

def tokenize_zh(text):
    return list(jieba.cut(text))  # 使用 jieba 对中文文本进行分词

en_tokenized = [tokenize_en(text) for text in df['en']]  # 对英文文本进行分词
zh_tokenized = [tokenize_zh(text) for text in df['zh']]  # 对中文文本进行分词

# 构建词汇表函数
def build_vocab(tokenized_texts, max_vocab_size=10000):
    word_freq = Counter()  # 创建一个 Counter 对象用于统计词频
    for tokens in tokenized_texts:
        word_freq.update(tokens)  # 更新词频统计
    vocab = {word: idx + 4 for idx, (word, _) in enumerate(word_freq.most_common(max_vocab_size))}  # 构建词汇表
    vocab['<PAD>'] = 0  # 添加填充标记
    vocab['<UNK>'] = 1  # 添加未知标记
    vocab['<SOS>'] = 2  # 添加句子开始标记
    vocab['<EOS>'] = 3  # 添加句子结束标记
    return vocab

en_vocab = build_vocab(en_tokenized)  # 构建英文词汇表
zh_vocab = build_vocab(zh_tokenized)  # 构建中文词汇表

# 将句子编码为ID序列
def encode_sentence(sentence, vocab, add_sos_eos=True):
    if add_sos_eos:
        sentence = ["<SOS>"] + sentence + ["<EOS>"]  # 添加句子开始和结束标记
    return [vocab.get(word, vocab['<UNK>']) for word in sentence]  # 将句子中的词转换为ID

en_encoded = [encode_sentence(sentence, en_vocab) for sentence in en_tokenized]  # 将英文句子编码为ID序列
zh_encoded = [encode_sentence(sentence, zh_vocab) for sentence in zh_tokenized]  # 将中文句子编码为ID序列

# 自定义数据集类
class TranslationDataset(Dataset):
    def __init__(self, en_data, zh_data):
        self.en_data = en_data  # 英文数据
        self.zh_data = zh_data  # 中文数据
    
    def __len__(self):
        return len(self.en_data)  # 返回数据集的大小
    
    def __getitem__(self, idx):
        return torch.tensor(self.en_data[idx]), torch.tensor(self.zh_data[idx])  # 返回指定索引的数据

# collate_fn 用于动态填充
def collate_fn(batch):
    en_batch, zh_batch = zip(*batch)  # 解压批次数据
    
    # 修改英文序列的填充方式
    en_batch = pad_sequence([
        seq.clone().detach() if isinstance(seq, torch.Tensor)
        else torch.tensor(seq, dtype=torch.long)
        for seq in en_batch
    ], batch_first=True, padding_value=en_vocab['<PAD>'])
    
    # 修改中文序列的填充方式
    zh_batch = pad_sequence([
        seq.clone().detach() if isinstance(seq, torch.Tensor)
        else torch.tensor(seq, dtype=torch.long)
        for seq in zh_batch
    ], batch_first=True, padding_value=zh_vocab['<PAD>'])
    
    return en_batch, zh_batch  # 返回填充后的批次数据

dataset = TranslationDataset(en_encoded, zh_encoded)  # 创建自定义数据集
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)  # 创建数据加载器

# 检查数据加载器
for en_batch, zh_batch in train_dataloader:
    print("英文批次大小:", en_batch.size())  # 打印英文批次的大小
    print("中文批次大小:", zh_batch.size())  # 打印中文批次的大小
    break  # 仅检查第一个批次


# 模型实例化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = transformer.Transformer(
    src_vocab_size=len(en_vocab),
    trg_vocab_size=len(zh_vocab),
    src_pad_idx=en_vocab['<PAD>'],
    trg_pad_idx=zh_vocab['<PAD>'],
    embed_size=512,
    num_layers=6,
    heads=8,
    forward_expansion=4,
    dropout=0.1,
    max_length=100
).to(device)

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss(ignore_index=zh_vocab['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# 训练循环
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for en_batch, zh_batch in train_dataloader:
        en_batch = en_batch.to(device)
        zh_batch = zh_batch.to(device)
        
        optimizer.zero_grad()
        output = model(en_batch, zh_batch[:, :-1])
        output = output.reshape(-1, output.shape[2])
        loss = loss_fn(output, zh_batch[:, 1:].reshape(-1))
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_dataloader):.4f}")

# 保存模型
torch.save(model.state_dict(), "transformer_model.pth")

# # 推理函数
# def translate_sentence(sentence, model, en_vocab, zh_vocab, max_length=50):
#     model.eval()
#     tokens = [token.lower() for token in tokenize_en(sentence)]
#     encoded = [en_vocab.get(token, en_vocab['<UNK>']) for token in tokens]
#     src_tensor = torch.tensor([encoded], dtype=torch.long, device=device)
    
#     outputs = [zh_vocab['<SOS>']]
#     for _ in range(max_length):
#         trg_tensor = torch.tensor([outputs], dtype=torch.long, device=device)
#         with torch.no_grad():
#             output = model(src_tensor, trg_tensor)
#             next_token = output.argmax(2)[:, -1].item()
#             outputs.append(next_token)
        
#         if next_token == zh_vocab['<EOS>']:
#             break
    
#     translated_sentence = [list(zh_vocab.keys())[list(zh_vocab.values()).index(idx)] for idx in outputs[1:]]
#     return ''.join(translated_sentence)

# # 测试推理
# sentence = "I love coding"
# translation = translate_sentence(sentence, model, en_vocab, zh_vocab)
# print("翻译结果:", translation)
