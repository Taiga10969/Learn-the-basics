import torch
import torch.nn as nn
import math

class Embedder(nn.Module):
    'IDで表されている単語をベクトル表現に変換する'
    'text_embeddign_vectors : (ボキャブラリの総単語数*分散表現の次元数)  参考 : p.356, pytorch.org/text/stable/vocab.html'
    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(embeddings=text_embedding_vectors, freeze=True)
        #print(self.embeddings.weight.size())
        #self.embeddings.weight = torch.cat((self.embeddings.weight, torch.rand(3,300)), dim=0)

        #freeze=Trueにより，Back Propagationの際に，更新されない設定となる．

    def forward(self, x):
        x_vec = self.embeddings(x)
        return x_vec


class PositionalEncoder(nn.Module):
    def __init__(self, d_model=300, max_seq_len=256):
        super().__init__()

        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2*i) / d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2*i) / d_model)))

        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False

    def forward(self, x):
        ret = math.sqrt(self.d_model)*x + self.pe
        return ret
    

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, head_num, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        
        self.linearQ = nn.Linear(dim, dim, bias=False)
        self.linearK = nn.Linear(dim, dim, bias=False)
        self.linearV = nn.Linear(dim, dim, bias=False)

        self.linear = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim = 3)
        self.dropout = nn.Dropout(dropout)
    
    def split_head(self, x):
        x = torch.tensor_split(x, self.head_num, dim=2)
        x = torch.stack(x, dim=1)
        return x
    
    def concat_head(self, x):
        x = torch.tensor_split(x, x.size()[1], dim=1)
        x = torch.concat(x, dim=3).squeeze(dim=1)
        return x
    
    def forward(self, Q, K, V, mask=None):
        Q = self.linearQ(Q)
        K = self.linearK(K)
        V = self.linearV(V)

        Q = self.split_head(Q)
        K = self.split_head(K)
        V = self.split_head(V)

        #print("K.shape : ", K.shape)
        K_T = torch.transpose(K, 3, 2)
        #print("K_T.shape : ", K_T.shape)
        QK = torch.matmul(Q, K_T)
        #print("QK.shape : ", QK.shape)

        #QK = torch.matmul(Q, torch.transpose(K, 3, 2))
        QK = QK/((self.dim//self.head_num)**0.5)

        if mask is not None:
            QK = QK + mask
        
        softmax_QK = self.softmax(QK)
        softmax_QK = self.dropout(softmax_QK)

        QKV = torch.matmul(softmax_QK, V)
        QKV = self.concat_head(QKV)
        QKV = self.linear(QKV)
        return QKV


class FeedForward(nn.Module):

  def __init__(self, dim, hidden_dim = 2048, dropout = 0.1):
    super().__init__() 
    self.dropout = nn.Dropout(dropout)
    self.linear_1 = nn.Linear(dim, hidden_dim)
    self.relu = nn.ReLU()
    self.linear_2 = nn.Linear(hidden_dim, dim)

  def forward(self, x):
    x = self.linear_1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.linear_2(x)
    return x


class EncoderBlock(nn.Module):
    def __init__(self, dim, head_num, dropout=0.1):
        super().__init__()

        self.MHA = MultiHeadAttention(dim, head_num)
        self.layer_norm1 = nn.LayerNorm([dim])
        self.layer_norm2 = nn.LayerNorm([dim])
        self.FF = FeedForward(dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        Q = K = V = _x = x

        x = self.MHA(Q, K, V)

        x = self.dropout_1(x)
        x = x + _x # Add
        x = self.layer_norm1(x)

        _x = x

        x = self.FF(x)

        x = self.dropout_2(x)
        x = x + _x #Add
        x = self.layer_norm2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, vectors, dim, head_num, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.embed = Embedder(vectors)
        self.PE = PositionalEncoder(dim)
        self.dropout = nn.Dropout(dropout)
        self.EncoderBlocks = nn.ModuleList([EncoderBlock(dim, head_num) for _ in range(6)])

    def forward(self, x):
        x = self.embed(x)
        x = x*(self.dim**0.5)
        x = self.PE(x)
        x = self.dropout(x)

        for i in range(6):
            x = self.EncoderBlocks[i](x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, head_num, dropout=0.1):
        super().__init__()

        self.MMHA = MultiHeadAttention(dim, head_num) # Masked Multi-Head Attention
        self.MHA = MultiHeadAttention(dim, head_num)

        self.layer_norm_1 = nn.LayerNorm([dim])
        self.layer_norm_2 = nn.LayerNorm([dim])
        self.layer_norm_3 = nn.LayerNorm([dim])

        self.FF = FeedForward(dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, tgt, encoder_output, mask):
        # Decoder の一つ目のMHAは，self AttentionのMasked Multi-Head Attentionとなっており，入力QKVは，全て同じデータ
        Q = K = V = _x = tgt

        x = self.MMHA(Q, K, V, mask)
        x = self.dropout_1(x)
        x = x + _x
        x = self.layer_norm_1(x)
        
        # ここまでアーキテクチャ図のDecoderの下半分
        # ここからアーキテクチャ図のDecoderの上半分

        Q = x # queryには下半分からの出力を
        _x = x
        K = V = encoder_output # key,valueにはencoderからの出力を

        x = self.MHA(Q, K, V)
        x = self.dropout_2(x)
        x = x + _x
        x = self.layer_norm_2(x)
        
        _x = x

        x = self.FF(x)
        x = self.dropout_3(x)
        x = x + _x
        x = self.layer_norm_3(x)

        return x


class Decoder(nn.Module):
    def __init__(self, vectors, dim, head_num, dropout=0.1):
        super().__init__()
        self.dim = dim
        dec_vocab_size = len(vectors)
        
        self.embed = Embedder(vectors)
        self.PE = PositionalEncoder(dim)

        self.DecoderBlocks = nn.ModuleList([DecoderBlock(dim, head_num) for _ in range(6)])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim, dec_vocab_size)

    def forward(self, tgt, encoder_output, mask):
        x = self.embed(tgt)
        x = x*(self.dim**0.5)
        x = self.PE(x)
        x = self.dropout(x)

        for i in range(6):
            x = self.DecoderBlocks[i](x, encoder_output, mask)

        x = self.linear(x)
        return x


class Transformer(nn.Module):
    def __init__(self, vectors, dim, head_num):
        super().__init__()
        self.encoder = Encoder(vectors=torch.cat((vectors.vectors, torch.rand(3,300)), dim=0), dim=dim, head_num=head_num)
        self.decoder = Decoder(vectors=torch.cat((vectors.vectors, torch.rand(3,300)), dim=0), dim=dim, head_num=head_num)
    
    def forward(self, src, tgt, mask):
        encoder_output = self.encoder(src)
        output = self.decoder(tgt, encoder_output, mask)
        return output

if __name__=='__main__':

    import torchtext.vocab as vocab
    vectors = vocab.FastText(language='en')

    src = torch.randint(low=1, high=20000, size = (5,256))
    tgt = torch.randint(low=1, high=20000, size = (5,256))
    print("input.shape : ", src.shape)
    mask = torch.zeros(256,256)
    model = Transformer(vectors=vectors, dim=300, head_num=6)
    output = model(src=src, tgt=tgt, mask=mask)
    print("output.shape : ", output.shape)
    print("output : \n", output)
    