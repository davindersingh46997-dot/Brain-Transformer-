import torch
import torch.nn as nn

class PositionalEncoding(nn.Module) :
    def __init__(self,d_model, max_len=5000):
        super(PositionalEncoding,self).__init__()
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-torch.log(torch.tensor(10000.0)))/d_model)
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + self.pe[:x.size(1), :].transpose(0,1)
        return x
       

class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_model,dropout=0.1):
        super(ScaledDotProductAttention,self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self,query,key,value,mask=None):
        scores = torch.matmul(query,key.transpose(-2,-1))/torch.sqrt(torch.tensor(self.d_model).float()) 
        if mask is not None:
            scores = scores.masked_fill(mask==0,-1e9)
        attention = torch.softmax(scores,dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention,value)
        return output,attention

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout=0.1):
        super(MultiHeadAttention,self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.attention = ScaledDotProductAttention(self.d_k,dropout)
        self.w_o = nn.Linear(d_model,d_model)

    def forward(self,query,key,value,mask=None):
        batch_size = query.size(0)
        query = self.w_q(query).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        key = self.w_k(key).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        value = self.w_v(value).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)

        out,attention = self.attention(query,key,value,mask)
        out = out.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        out = self.w_o(out)

        return out,attention
    
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(FeedForward,self).__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out    

class AddNorm(nn.Module):
    def __init__(self,d_model,dropout=0.1):
        super(AddNorm,self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self,x,sub_layer_out):
        out = self.dropout(sub_layer_out)
        out = self.norm(x+out)
        return out 

class EncoderLayer(nn.Module):
    def __init__(self,d_model,d_ff,num_heads,dropout=0.1):
        super(EncoderLayer,self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model,num_heads,dropout)
        self.feed_forward = FeedForward(d_model,d_ff,dropout)
        self.add_norm1 = AddNorm(d_model,dropout)
        self.add_norm2 = AddNorm(d_model,dropout)

    def forward(self,x,mask=None):
        att_out,attention = self.multi_head_attention(x,x,x,mask)
        norm1_out = self.add_norm1(x,att_out)

        ffn_out = self.feed_forward(norm1_out)
        out = self.add_norm2(norm1_out,ffn_out)
        
        return out,attention

class Encoder(nn.Module):
    def __init__(self,num_layers,d_model,d_ff,num_heads,dropout=0.1):
        super(Encoder,self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model,d_ff,num_heads,dropout) for _ in range(num_layers)])

    def forward(self,x,mask=None):
        attention_weights = []
        for layer in self.layers:
            x,attention = layer(x,mask)
            attention_weights.append(attention)
        return x,attention_weights 

class DecoderLayer(nn.Module):
    def __init__(self,d_model,d_ff,num_heads,dropout=0.1):
        super(DecoderLayer,self).__init__()
        self.masked_self_attention = MultiHeadAttention(d_model,num_heads,dropout)
        self.cross_attention = MultiHeadAttention(d_model,num_heads,dropout)
        self.feed_forward = FeedForward(d_model,d_ff,dropout)
        self.add_norm1 = AddNorm(d_model,dropout)
        self.add_norm2 = AddNorm(d_model,dropout)
        self.add_norm3 = AddNorm(d_model,dropout)
        

    def forward(self,x,mask,encoder_output):
        self_att_out,self_attention = self.masked_self_attention(x,x,x,mask)
        norm1_out = self.add_norm1(x,self_att_out)
        cross_att_out,cross_attention = self.cross_attention(norm1_out,encoder_output,encoder_output,mask)
        norm2_out = self.add_norm2(norm1_out,cross_att_out)
        ffn_out = self.feed_forward(norm2_out)
        norm3_out = self.add_norm3(norm2_out,ffn_out)
        
        return norm3_out,self_attention,cross_attention


class Decoder(nn.Module):
    def __init__(self,num_layers,d_model,d_ff,num_heads,vocab_size,dropout=0.1):
        super(Decoder,self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model,d_ff,num_heads,dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model,vocab_size)

    def forward(self,x,mask,encoder_output):
        self_attention_weights = []

        cross_attention_weights = []

        for layer in self.layers:
            x,self_attention,cross_attention = layer(x,mask,encoder_output)
            self_attention_weights.append(self_attention)
            cross_attention_weights.append(cross_attention)
            logits = self.linear(x)
            
        return logits,self_attention_weights,cross_attention_weights


        

        

        
        




