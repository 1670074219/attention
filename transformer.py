import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
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
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, query, keys, values, mask):
        N = query.shape[0] # batch_size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # 将每个头的数据分开计算内积：对最后一个维度 (head_dim) 进行矩阵乘法
        energy = torch.zeros(N, self.heads, query_len, key_len)  # 初始化输出张量
        for b in range(N):  # 遍历每个 batch
            for h in range(self.heads):    # 遍历每个注意力头
                # queries[b, :, h, :] 的形状是 (query_len, head_dim)
                # keys[b, :, h, :] 的形状是 (key_len, head_dim)
                # 转置 keys 的最后两个维度，以便进行矩阵乘法
                energy[b, h] = torch.matmul(queries[b, :, h, :], keys[b, :, h, :].transpose(0, 1))

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)

        # out = torch.einsum("bhqk,bkhd->bqhd", [attention, values])  # (batch_size, query_len, heads, head_dim) 
        # 初始化输出张量，形状为 (batch_size, query_len, heads, head_dim)
        out = torch.zeros(N, query_len, self.heads, self.head_dim)
        # 遍历每个 batch 和每个头进行计算
        for b in range(N):  # 遍历每个 batch
            for h in range(self.heads):   # 遍历每个注意力头
                for q in range(query_len):  # 遍历每个 query
                    # 对 values 的 key_len 维度进行加权求和
                    # 注意力权重 (1, key_len) @ values (key_len, head_dim) -> (1, head_dim)
                    out[b, q, h, :] = torch.matmul(attention[b, h, q, :], values[b, :, h, :])
        
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
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
        self.p = p # 丢弃概率

    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) > self.p).float()
            return x * mask / (1 - self.p)
        else:
            return x * (1 - self.p)

class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = CustomLayerNorm(embed_size)
        self.norm2 = CustomLayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = CustomDropout(dropout)

    def forward(self, x, mask):
        # 1.多头自注意力层
        attention = self.attention(x, x, x, mask)
        # 2.残差连接 + Layer Normalization
        x = self.dropout(self.norm1(attention + x))
        # 3.前馈神经网络层
        forward = self.feed_forward(x)
        # 4.残差连接 + Layer Normalization
        out = self.dropout(self.norm2(forward + x))
        return out
    

# # 测试EncoderBlock代码
# embed_size = 512
# heads = 8
# forward_expansion = 4
# dropout = 0.1
# seq_length = 10
# batch_size = 2

# # 假设我们有一个输入张量 x
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

        # 自注意力层，使用掩码避免信息泄露
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
                 dropout=0.1, max_length=100):
        super(Transformer, self).__init__()

        #初始化编码器和解码器
        self.encoder = TransformerEncoder(src_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length)
        self.decoder = TransformerDecoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length)
    
        #输出层，将解码器的输出映射到词汇表的大小
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.embed_size = embed_size

    def make_src_mask(self, src):
        # 生成 src_mask，用于编码器-解码器注意力层，屏蔽源序列中的填充词
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask 形状：(batch_size, 1, 1, src_len)
        return src_mask
    
    def make_trg_mask(self, trg):
        # 生成 trg_mask，用于解码器自注意力层，防止解码器看到未来的信息
        trg_len = trg.shape[1]
        trg_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool() # (trg_len, trg_len)
        trg_padding_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, trg_len)
        trg_mask = trg_mask & trg_padding_mask # (batch_size, 1, trg_len, trg_len)
        return trg_mask

    def forward(self, src, trg):
        # 生成源序列和目标序列的掩码
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # 编码器处理源序列
        enc_src = self.encoder(src, src_mask) # (batch_size, src_len, embed_size)

        # 解码器处理目标序列和编码器输出
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
    

# # 假设源和目标词汇表大小
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

# # 模拟输入数据
# src = torch.randint(0, src_vocab_size, (2, 20))  # (batch_size=2, src_seq_len=20)
# trg = torch.randint(0, trg_vocab_size, (2, 20))  # (batch_size=2, trg_seq_len=20)

# # 实例化 Transformer 模型并进行前向传播
# model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
#                     embed_size, num_layers, forward_expansion, heads, dropout, max_length)

# out = model(src, trg)
# print("Transformer 模型的输出形状：", out.shape)