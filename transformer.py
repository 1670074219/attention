import torch
import torch.nn as nn
import math
from torch.amp import autocast, GradScaler

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        """
        初始化位置编码层
        参数:
            embed_size: 词嵌入维度
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        # 创建一个零矩阵来存储位置编码
        pe = torch.zeros(max_len, embed_size)
        # 生成位置索引矩阵，形状为 (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1)
        # 计算分母项
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        # 计算正弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算余弦位置编码
        pe[:, 1::2] = torch.cos(position * div_term)
        # 添加批次维度
        pe = pe.unsqueeze(0)
        # 将位置编码注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量，形状为 (batch_size, seq_len, embed_size)
        返回:
            添加位置编码后的张量
        """
        # 检查输入序列长度是否超过最大长度
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"输入序列长度 ({x.size(1)}) 超过位置编码的最大长度 ({self.pe.size(1)})")
        # 将位置编码加到输入上
        x = x + self.pe[:, :x.size(1), :]
        return x
    

# # 测试PositionalEncoding代码
# embed_size = 512
# max_len = 5000
# seq_length = 20
# batch_size = 2
# # 实例化位置编码类
# pos_encoding = PositionalEncoding(embed_size, max_len)
# # 生成一个假设的输入张量 x，形状为 (batch_size, seq_length, embed_size)
# x = torch.randn(batch_size, seq_length, embed_size)
# # 前向传播
# output = pos_encoding(x)
# print("位置编码后的输出形状：", output.shape)

    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        """
        初始化多头注意力层
        参数:
            embed_size: 词嵌入维度
            heads: 注意力头数
        """
        super(MultiHeadAttention, self).__init__()  # 初始化父类
        self.embed_size = embed_size  # 存储嵌入维度大小
        self.heads = heads  # 存储注意力头的数量
        self.head_dim = embed_size // heads  # 计算每个注意力头的维度

        # 验证嵌入维度是否可以被头数整除
        assert(self.head_dim * heads == embed_size), "嵌入维度需要能够被头数整除"

        # 创建三个线性变换层，用于转换查询、键和值
        # 输入维度是embed_size，输出维度是head_dim * heads
        self.values = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        self.keys = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        self.queries = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        
        # 创建输出的线性变换层
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, query, keys, values, mask):
        """
        前向传播
        参数:
            query: 查询张量
            keys: 键张量
            values: 值张量
            mask: 掩码张量
        """
        N = query.shape[0]  # 批次大小
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 对输入进行线性变换
        values = self.values(values)  # [N, value_len, embed_size]
        keys = self.keys(keys)        # [N, key_len, embed_size]
        queries = self.queries(query)  # [N, query_len, embed_size]

        # 重塑张量维度，将每个头的数据分开
        # [N, seq_len, heads, head_dim]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # 使用爱因斯坦求和计算注意力分数
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # 如果提供了掩码，则应用掩码
        if mask is not None:
            mask = mask.to(energy.device)
            energy = energy.masked_fill(mask == 0, -1e4)

        # 计算注意力权重（使用缩放点积注意力机制）
        attention = torch.softmax(energy / (self.head_dim ** (1/2)), dim=3)

        # 确保注意力权重在正确的设备上
        attention = attention.to(values.device)

        # 使用注意力权重和值计算输出
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # out shape: (N, query_len, heads, head_dim)
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values])
        
        # 重塑输出张量并通过最终的线性层
        out = out.reshape(N, query_len, self.heads * self.head_dim)  # [N, query_len, embed_size]
        out = self.fc_out(out)  # 通过输出线性层
        return out

class CustomLayerNorm(nn.Module):
    def __init__(self, embed_size, epsilon=1e-6):
        super(CustomLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(embed_size))
        self.beta = nn.Parameter(torch.zeros(embed_size))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.epsilon) 
        out = self.gamma * x_normalized + self.beta
        return out

class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super(CustomDropout, self).__init__()
        if not 0 <= p <= 1:
            raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, x):
        if self.training:
            # 直接在与输入相同的设备上生成掩码
            mask = (torch.rand_like(x, device=x.device) > self.p).float()
            return x * mask / (1 - self.p)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        """
        初始化编码器块
        参数:
            embed_size: 词嵌入维度
            heads: 注意力头数
            forward_expansion: 前馈网络扩展因子
            dropout: dropout比率
        """
        super(EncoderBlock, self).__init__()
        # 多头自注意力层
        self.attention = MultiHeadAttention(embed_size, heads)
        # 两个层归一化层
        self.norm1 = CustomLayerNorm(embed_size)
        self.norm2 = CustomLayerNorm(embed_size)

        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        # Dropout层
        self.dropout = CustomDropout(dropout)

    def forward(self, x, mask):
        """
        前向传播
        参数:
            x: 输入张量
            mask: 掩码张量
        """
        # 1.多头自注意力层
        attention = self.attention(x, x, x, mask)
        # 2.第一个残差连接和层归一化
        x = self.dropout(self.norm1(attention + x))
        # 3.前馈神经网络
        forward = self.feed_forward(x)
        # 4.第二个残差连接和层归一化
        out = self.dropout(self.norm2(forward + x))
        return out
    

# # 测试EncoderBlock代码
# embed_size = 512
# heads = 8
# forward_expansion = 4
# dropout = 0.1
# seq_length = 10
# batch_size = 2

# # 假设我们一个输入张量 x
# x = torch.rand(batch_size, seq_length, embed_size)
# mask = None  # 可以暂时不使用 mask

# # 实例化编码器块并进行前向传播
# encoder_block = EncoderBlock(embed_size, heads, forward_expansion, dropout)
# out = encoder_block(x, mask)

# print("编码器块的输出形状：", out.shape)

class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length):
        super(TransformerEncoder, self).__init__()

        # 1. 输入嵌入层
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_encoding = PositionalEncoding(embed_size, max_length)

        # 2. 多个编码器块的堆叠
        self.layers = nn.ModuleList(
            [EncoderBlock(embed_size, heads, forward_expansion, dropout) for _ in range(num_layers)]
        )

        # 3.自定义Dropout层
        self.dropout = CustomDropout(dropout)


    def forward(self, x, mask):
        # 1. 添加词嵌入和位置编码:
        out = self.word_embedding(x)
        out = self.position_encoding(out)
        out = self.dropout(out) # 使用自定义Dropout
        
        # 2. 通过多个编码器块
        for layer in self.layers:
            out = layer(out, mask)
        
        return out


# # 测试TransformerEncoder代码
# src_vocab_size = 10000  # 假设词汇表大小
# embed_size = 512
# num_layers = 6
# heads = 8
# forward_expansion = 4
# dropout = 0.1
# max_length = 100
# seq_length = 20
# batch_size = 2
# # 假设输入的词序列
# x = torch.randint(0, src_vocab_size, (batch_size, seq_length))  # (batch_size, seq_length)
# mask = None  # 在训练时可以不使用 mask
# # 实例化 Transformer 编码器并进行前向传播
# encoder = TransformerEncoder(src_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length)
# out = encoder(x, mask)
# print("编码器的输出形状：", out.shape)


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()

        # 自注意力层，使用掩码避免息泄露
        self.self_attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = CustomLayerNorm(embed_size)

        # 编码器-解码器注意力层
        self.encoder_decoder_attention = MultiHeadAttention(embed_size, heads)
        self.norm2 = CustomLayerNorm(embed_size)

        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.norm3 = CustomLayerNorm(embed_size)

        # Dropout
        self.dropout = CustomDropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        # 1. 自注意力层
        self_attn = self.self_attention(x, x, x, trg_mask)
        x = self.dropout(self.norm1(self_attn + x))

        # 2. 编码器-解码器注意力层
        enc_dec_attn = self.encoder_decoder_attention(x, enc_out, enc_out, src_mask)
        x = self.dropout(self.norm2(enc_dec_attn + x))

        # 3. 前馈神经网络
        forward = self.feed_forward(x)
        out = self.dropout(self.norm3(forward + x))

        return out
    
class TransformerDecoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length):
        super(TransformerDecoder, self).__init__()

        # 词嵌入层
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_encoding = PositionalEncoding(embed_size, max_length)

        # 解码器层堆叠
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout) for _ in range(num_layers)]
        )

        # 输出层，用于预测词的概率
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

        # Dropout
        self.dropout = CustomDropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        # 1. 添加词嵌入和位置编码
        out = self.word_embedding(x)
        out = self.position_encoding(out)
        out = self.dropout(out)

        # 2. 逐层通过解码器块
        for layer in self.layers:
            out = layer(out, enc_out, src_mask, trg_mask)

        # 3. 输出层
        out = self.fc_out(out)
        return out
    
# # 测试TransformerDecoder代码
# trg_vocab_size = 10000  # 假设目标词汇表大小
# embed_size = 512
# num_layers = 6
# heads = 8
# forward_expansion = 4
# dropout = 0.1
# max_length = 100
# seq_length = 20
# batch_size = 2

# # 假设输入的目标词序列
# x = torch.randint(0, trg_vocab_size, (batch_size, seq_length))  # (batch_size, seq_length)
# enc_out = torch.rand(batch_size, seq_length, embed_size)  # 编码器的输出
# src_mask = None
# trg_mask = None

# # 实例化 Transformer 解码器并进行前向传播
# decoder = TransformerDecoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length)
# out = decoder(x, enc_out, src_mask, trg_mask)

# print("解码器的输出形状：", out.shape)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,
                 embed_size=512, num_layers=6, forward_expansion=4, heads=8,
                 dropout=0.1, max_length=200, device=device):
        super(Transformer, self).__init__()
        
        self.device = device  # 存储设备信息
        
        # 将编码器移至指定设备
        self.encoder = TransformerEncoder(
            src_vocab_size, embed_size, num_layers, heads, 
            forward_expansion, dropout, max_length
        ).to(device)
        
        # 将解码器移至指定设备
        self.decoder = TransformerDecoder(
            trg_vocab_size, embed_size, num_layers, heads,
            forward_expansion, dropout, max_length
        ).to(device)
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        
        # 将整个模型移至指定设备
        self.to(device)

    def make_src_mask(self, src):
        # 直接在正确的设备上创建掩码
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        # 直接在正确的设备上创建掩码
        trg_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_mask.expand(N, 1, trg_len, trg_len)
        padding_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        return (trg_mask & padding_mask).to(self.device)

    def forward(self, src, trg):
        # 确保输入数据在正确的设备上
        src = src.to(self.device)
        trg = trg.to(self.device)
        
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
    

# # 假设源和标词汇表大小
# src_vocab_size = 10000
# trg_vocab_size = 10000

# # 定义填充索引
# src_pad_idx = 1
# trg_pad_idx = 1

# # 其他参数
# embed_size = 512
# num_layers = 6
# heads = 8
# forward_expansion = 4
# dropout = 0.1
# max_length = 100

# # 模拟输数据
# src = torch.randint(0, src_vocab_size, (2, 20))  # (batch_size=2, src_seq_len=20)
# trg = torch.randint(0, trg_vocab_size, (2, 20))  # (batch_size=2, trg_seq_len=20)

# # 实例化 Transformer 模型并进行前向传播
# model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
#                     embed_size, num_layers, forward_expansion, heads, dropout, max_length)

# out = model(src, trg)
# print("Transformer 模型的输出形状：", out.shape)




# def train_model():
#     # 模型参数设置
#     src_vocab_size = 10000
#     trg_vocab_size = 10000
#     src_pad_idx = 1
#     trg_pad_idx = 1
#     embed_size = 512
#     num_layers = 6
#     heads = 8
#     forward_expansion = 4
#     dropout = 0.1
#     max_length = 100
#     learning_rate = 0.0001
#     batch_size = 32
    
#     print(f"使用设备: {device}")
    
#     # 实例化模型并移至GPU
#     model = Transformer(
#         src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,
#         embed_size, num_layers, forward_expansion, heads,
#         dropout, max_length, device
#     )
    
#     # 设置优化器和损失函数
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
#     scaler = GradScaler()  # 更新 GradScaler
    
#     # 添加学习率调度器
#     from torch.optim.lr_scheduler import LambdaLR
    
#     def lr_lambda(step):
#         # 实现预热策略
#         warmup_steps = 4000
#         step = float(step + 1)
#         return min(step ** (-0.5), step * warmup_steps ** (-1.5))
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
#     scheduler = LambdaLR(optimizer, lr_lambda)
    
#     def train_step(src, trg):
#         model.train()
#         optimizer.zero_grad()
        
#         with autocast(device_type='cuda', dtype=torch.float16):
#             output = model(src, trg[:, :-1])
#             output = output.reshape(-1, output.shape[-1])
#             target = trg[:, 1:].reshape(-1)
#             loss = criterion(output, target)
        
#         scaler.scale(loss).backward()
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         scaler.step(optimizer)
#         scaler.update()
#         scheduler.step()
        
#         return loss.item()
    
#     try:
#         # 示例训练数据
#         src = torch.randint(0, src_vocab_size, (batch_size, 20)).to(device)
#         trg = torch.randint(0, trg_vocab_size, (batch_size, 20)).to(device)
        
#         # 训练一个批次
#         loss = train_step(src, trg)
#         print(f"批次损失: {loss:.4f}")
        
#     except RuntimeError as e:
#         print(f"训练出错: {str(e)}")
#         # 如果是 CUDA 内存错误，清理内存
#         if "out of memory" in str(e):
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#         raise e

# if __name__ == "__main__":
#     try:
#         train_model()
#     except Exception as e:
#         print(f"程序出错: {str(e)}")