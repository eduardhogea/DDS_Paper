import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
# from parameter import parse_option
# args = parse_option()


class Feed_Forward(nn.Module):
    """
    Feedforward network in encoder
    """
    def __init__(self, hidden_size, mlp_ratio, attn_drop_rate):
        super(Feed_Forward, self).__init__()
        mlp_dim = int(mlp_ratio * hidden_size)
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.activation = F.gelu
        self.dropout = nn.Dropout(attn_drop_rate)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.dropout(out)
        # out = out + x  # 残差连接
        # out = self.layer_norm(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    """
    Scaled Dot-Product Attention
    """
    def __init__(self, attn_dropout=0.1):
        super(Scaled_Dot_Product_Attention, self).__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, Q, K, V):
        attention = torch.matmul(Q, K.transpose(-1, -2))
        scale = K.size(-1) ** -0.5
        attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        weights = attention
        output = torch.matmul(attention, V)
        return output, weights


class DenseAttention(nn.Module):

    def __init__(self, max_seq_len, d_k, d_hid=32, attn_dropout=0.1):
        super(DenseAttention, self).__init__()
        self.w_1 = nn.Linear(d_k, d_hid)
        self.w_2 = nn.Linear(d_hid, max_seq_len)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, Q, K, V, len_q):
        dense_attn = self.w_2(self.relu(self.w_1(Q)))[:, :, :, :len_q]
        dense_attn = F.softmax(dense_attn, dim=-1)
        weights = dense_attn
        output = torch.matmul(dense_attn, V)
        return output, weights


class RandomAttention(nn.Module):

    def __init__(self, batch_size, n_head, max_seq_len, attn_dropout=0.1):
        super(RandomAttention, self).__init__()
        self.random_attn = torch.randn(batch_size, n_head, max_seq_len, max_seq_len, requires_grad=True)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, Q, K, V, len_q):
        random_attn = self.random_attn[:Q.size()[0], :, :len_q, :len_q]
       #  random_attn = random_attn.to(torch.device("cuda"))
        random_attn = F.softmax(random_attn, dim=-1)
        weights = random_attn
        output = torch.matmul(random_attn, V)
        return output, weights


class Multi_Head_Attention(nn.Module):

    def __init__(self, hidden_size, num_heads, drop_rate, attention_choice):
        super(Multi_Head_Attention, self).__init__()
        self.num_attention_heads = num_heads
        self.head_dim = int(hidden_size / num_heads)
        self.all_head_size = self.head_dim * self.num_attention_heads
        self.fc_query = nn.Linear(hidden_size, self.all_head_size)
        self.fc_key = nn.Linear(hidden_size, self.all_head_size)
        self.fc_value = nn.Linear(hidden_size, self.all_head_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(drop_rate)
        self.attention_choice = attention_choice
        if self.attention_choice == "dot":
            self.attention = Scaled_Dot_Product_Attention()  # 点积Attention
        elif self.attention_choice == "dense":
            self.attention = DenseAttention(max_seq_len=40, d_k=self.head_dim, d_hid=32)
        elif self.attention_choice == "random":
            self.attention = RandomAttention(batch_size=1, n_head=num_heads, max_seq_len=40)
        else:
            self.attention = None
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.fc_query(hidden_states)
        mixed_key_layer = self.fc_key(hidden_states)
        mixed_value_layer = self.fc_value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        if self.attention_choice == "dot":
            attention_scores, weights = self.attention(query_layer, key_layer, value_layer)
        elif self.attention_choice == "dense":
            attention_scores, weights = self.attention(query_layer, key_layer, value_layer, len_q=query_layer.size()[-2])
        elif self.attention_choice == "random":
            attention_scores, weights = self.attention(query_layer, key_layer, value_layer, len_q=query_layer.size()[-2])
        else:
            attention_scores, weights = None, None
        context_layer = attention_scores.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        out = self.fc(context_layer)
        out = self.dropout(out)
        out = out + hidden_states  # residual connection like in ResNet
        out = self.layer_norm(out)
        return out, weights


class Position_Embedding(nn.Module):
    """
    Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, input_channel, signal_size, patch_size, hidden_size, drop_rate):
        super(Position_Embedding, self).__init__()
        seq_size = (1, signal_size)
        n_patches = seq_size[1] // patch_size
        self.patch_embeddings = nn.Conv1d(in_channels=input_channel, out_channels=hidden_size,
                                          kernel_size=patch_size, stride=patch_size // 2)
        self.cls_token = nn.Parameter(torch.rand(1, hidden_size))  # 分类器
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x0 = x.shape[0]
        cls_tokens = self.cls_token.expand(x0, -1, -1)
        x = self.patch_embeddings(x)
        x = x.transpose(-2, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = self.dropout(x)
        return embeddings


class Encoder_Block(nn.Module):
    """
    Block in encoder, including multihead attention and feedforward neural network
    Encoder层里的单元，包括多头注意力层和前馈神经网络
    """
    def __init__(self, hidden_size, mlp_ratio, attn_drop_rate, num_heads, drop_rate, drop_path_rate, attention_choice):
        super(Encoder_Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)  # Normalization layer before multihead attentiuon
        self.attention = Multi_Head_Attention( hidden_size, num_heads, drop_rate, attention_choice)  # multi head attention
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)  # nomalization layer before feed network
        self.ffn = Feed_Forward(hidden_size, mlp_ratio, attn_drop_rate)  # feed network

    def forward(self, x):
        x = self.attention_norm(x)
        x, weights = self.attention(x)
        x = self.ffn_norm(x)
        x = self.ffn(x)
        return x, weights


class Encoder(nn.Module):
    """
    Transformer Encoder 层
    """
    def __init__(self, hidden_size, depth, mlp_ratio, attn_drop_rate, num_heads, drop_rate,
                 drop_path_rate, attention_choice):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        for _ in range(depth):
            layer = Encoder_Block(hidden_size, mlp_ratio, attn_drop_rate, num_heads, drop_rate, drop_path_rate,
                                  attention_choice)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            attn_weights.append(weights)
        # encoded = self.encoder_norm(hidden_states)
        return hidden_states, attn_weights


class Transformer(nn.Module):
    """
    Transformer layer
    """
    def __init__(self, input_channel, signal_size, patch_size, hidden_size, drop_rate, depth, mlp_ratio, attn_drop_rate, num_heads,
                 drop_path_rate, attention_choice):
        super(Transformer, self).__init__()
        self.embeddings = Position_Embedding(input_channel, signal_size, patch_size, hidden_size, drop_rate)  # 将序列编码
        self.encoder = Encoder(hidden_size, depth, mlp_ratio, attn_drop_rate, num_heads, drop_rate, drop_path_rate,
                               attention_choice)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    """
    Transformer for classification
    """
    def __init__(self, input_channel=10, signal_size=1000, patch_size=50, num_class=12, hidden_size=256, depth=3,
                 attention_choice="dot",
                 num_heads=8, mlp_ratio=4., drop_rate=0.2, attn_drop_rate=0.2, drop_path_rate=0.2, classifier="gap"):
        super(VisionTransformer, self).__init__()
        self.num_class = num_class
        self.transformer = Transformer(input_channel, signal_size, patch_size, hidden_size, drop_rate,
                                       depth, mlp_ratio, attn_drop_rate, num_heads, drop_path_rate, attention_choice)
        self.linear_dim = int(hidden_size / 2)
        self.classifier = classifier
        self.avg_pool = nn.AdaptiveAvgPool1d(4)
        self.max_pool = nn.AdaptiveMaxPool1d(4)
        self.fc = nn.Sequential(
            nn.Linear(256 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_class)
        )

    def forward(self, x):
        hidden_state, attn_weights = self.transformer(x)
        if self.classifier == "token":
            cls_head = hidden_state[:, 0]
            x_logits = self.head(cls_head)
        elif self.classifier == "gap":
            features = hidden_state[:, 1:]
            features = features.transpose(-2, -1)
            avg_features = self.avg_pool(features)
            max_features = self.max_pool(features)
            features = torch.cat([avg_features, max_features], dim=-1)
            features = features.view(features.size()[0], -1)
            x_logits = self.fc(features)
        else:
            raise ValueError("config.classifier is non-existent")
        return x_logits, attn_weights


if __name__ == '__main__':
    input = torch.randn(64, 33, 1024)
    model = VisionTransformer(input_channel=33, signal_size=1024, num_class=20, attention_choice="dot")
    y, weights = model(input)
    print(y.shape)
    print(len(weights[0][0][0][0]))

