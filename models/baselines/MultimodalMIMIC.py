"""
1. BertForRepresentation Model
- Input: input_ids_sequence and attention_mask_sequence (a list of input text tensors).  
- Output: A tensor containing text embeddings aggregated using the torch.stack() function, with a shape of (sequence_count, batch_size, hidden_size).  

2. TextModel
- Input: (batch_size, seq_len, n_features).  
- Output: Loss(Scalar).

3. MULTCrossModel
- Input: (batch_size, seq_len, n_features).  
- Output: Loss(Scalar).

4. TSMixed Model
- Input: (batch_size, seq_len, n_features).
- Output: Loss(Scalar).
"""

from typing import Any, Optional
import torch.onnx.operators
from fairseq import utils
from torch import nn, Tensor
import sys
import math
import copy
import numpy as np

import torch
from torch import nn
from torch.nn import Parameter, BCELoss, CrossEntropyLoss
import torch.nn.functional as F

# Code adapted from the fairseq repo.
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AdamW,
    BertTokenizer,
    BertModel,
    get_scheduler,
    set_seed,
    BertPreTrainedModel
) 

class BertForRepresentation(nn.Module):

    def __init__(self, args,BioBert):
        super().__init__()
        self.bert = BioBert

        self.dropout = torch.nn.Dropout(BioBert.config.hidden_dropout_prob)
        self.model_name=args.model_name     def forward(self, input_ids_sequence, attention_mask_sequence, sent_idx_list=None , doc_idx_list=None):
        txt_arr = []

        for input_ids,attention_mask  in zip(input_ids_sequence,attention_mask_sequence):

            if 'Longformer' in self.model_name:

                attention_mask-=1

                text_embeddings=self.bert(input_ids, global_attention_mask=attention_mask)
            else:
                text_embeddings=self.bert(input_ids, attention_mask=attention_mask)
            text_embeddings= text_embeddings[0][:,0,:]
            text_embeddings = self.dropout(text_embeddings)
            txt_arr.append(text_embeddings)

        txt_arr=torch.stack(txt_arr)
        return txt_arr 
class TextModel(nn.Module):
    def __init__(self,args,device,orig_d_txt=768,Biobert=None):
        """
        Construct a TextModel.
        """
        super(TextModel, self).__init__()

        self.device=device
        self.task=args.task
        self.agg_type=args.agg_type 
        self.out_dropout = args.dropout
        self.orig_d_txt=orig_d_txt
        self.d_txt= args.embed_dim
        self.bertrep=BertForRepresentation(args,Biobert) 
        self.proj_txt =nn.Linear(self.orig_d_txt, self.d_txt)

        output_dim = args.num_labels 

        self.proj1 = nn.Linear(self.d_txt, self.d_txt)
        self.proj2 = nn.Linear(self.d_txt, self.d_txt)
        self.out_layer = nn.Linear(self.d_txt, output_dim)

        if self.task=='ihm':
            self.loss_fct1=CrossEntropyLoss()
        elif self.task=='pheno':
            self.loss_fct1=nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")     def forward(self,  input_ids_sequences,
                attn_mask_sequences,labels=None):
        """
        dimension [batch_size, seq_len, n_features]

        """

        x_txt=self.bertrep(input_ids_sequences,attn_mask_sequences)

        x_txt=torch.mean(x_txt,dim=1)

        proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(x_txt)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(proj_x_txt)), p=self.out_dropout, training=self.training))
        last_hs_proj += proj_x_txt
        output = self.out_layer(last_hs_proj) 
        if self.task == 'ihm':
            if labels!=None:
                return self.loss_fct1(output, labels)
            return torch.nn.functional.softmax(output,dim=-1)[:,1]

        elif self.task == 'pheno':
            if labels!=None:
                labels=labels.float()
                return self.loss_fct1(output, labels)
            return torch.nn.functional.sigmoid(output) 

class MULTCrossModel(nn.Module):
    def __init__(self,args,device,modeltype=None,orig_d_ts=None,orig_reg_d_ts=None,orig_d_txt=None,ts_seq_num=None,text_seq_num=None, Biobert=None):
        """
        Construct a MulT Cross model.
        """
        super(MULTCrossModel, self).__init__()
        if modeltype!=None:
            self.modeltype=modeltype
        else:
            self.modeltype=args.modeltype
        self.num_heads = args.num_heads         self.layers = args.layers
        self.device=device
        self.kernel_size=args.kernel_size
        self.dropout=args.dropout
        self.attn_mask = False
        self.irregular_learn_emb_ts=args.irregular_learn_emb_ts
        self.irregular_learn_emb_text=args.irregular_learn_emb_text
        self.reg_ts=args.reg_ts
        self.TS_mixup=args.TS_mixup
        self.mixup_level=args.mixup_level
        self.task=args.task
        self.tt_max=args.tt_max
        self.cross_method=args.cross_method
        if self.irregular_learn_emb_ts or self.irregular_learn_emb_text :
            self.time_query=torch.linspace(0, 1., self.tt_max)
            self.periodic = nn.Linear(1, args.embed_time-1)
            self.linear = nn.Linear(1, 1)

        if "TS" in self.modeltype:
            self.orig_d_ts=orig_d_ts
            self.d_ts=args.embed_dim
            self.ts_seq_num=ts_seq_num

            if self.irregular_learn_emb_ts:
                self.time_attn_ts=multiTimeAttention(self.orig_d_ts*2, self.d_ts, args.embed_time, 8)
 
            if self.reg_ts:
                self.orig_reg_d_ts=orig_reg_d_ts
                self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

            if self.TS_mixup:
                if self.mixup_level=='batch':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=args.dropout)
                elif self.mixup_level=='batch_seq':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=args.dropout)
                elif self.mixup_level=='batch_seq_feature':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=self.d_ts,dropout=args.dropout)
                else:
                    raise ValueError("Unknown mixedup type")

        if "Text" in self.modeltype:
            self.orig_d_txt=orig_d_txt
            self.d_txt= args.embed_dim
            self.text_seq_num=text_seq_num
            self.bertrep=BertForRepresentation(args,Biobert)

            if self.irregular_learn_emb_text:
                self.time_attn=multiTimeAttention(768, self.d_txt, args.embed_time, 8)
            else:
                self.proj_txt = nn.Conv1d(self.orig_d_txt, self.d_txt, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        output_dim = args.num_labels
        if self.modeltype=="TS_Text":
            if self.cross_method=="self_cross":
                self.trans_self_cross_ts_txt=self.get_cross_network(layers=args.cross_layers)
                self.proj1 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                self.proj2 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                self.out_layer = nn.Linear(self.d_ts+self.d_txt, output_dim)
            else:
                self.trans_ts_mem = self.get_network(self_type='ts_mem', layers=args.layers)
                self.trans_txt_mem = self.get_network(self_type='txt_mem', layers=args.layers)

                if self.cross_method=="MulT":
                    self.trans_txt_with_ts=self.get_network(self_type='txt_with_ts',layers=args.cross_layers)
                    self.trans_ts_with_txt=self.get_network(self_type='ts_with_txt',layers=args.cross_layers)
                    self.proj1 = nn.Linear((self.d_ts+self.d_txt), (self.d_ts+self.d_txt))
                    self.proj2 = nn.Linear((self.d_ts+self.d_txt), (self.d_ts+self.d_txt))
                    self.out_layer = nn.Linear((self.d_ts+self.d_txt), output_dim)
                elif self.cross_method=="MAGGate":
                    self.gate_fusion=MAGGate(inp1_size=self.d_txt, inp2_size=self.d_ts, dropout=self.embed_dropout)
                    self.proj1 = nn.Linear(self.d_txt, self.d_txt)
                    self.proj2 = nn.Linear(self.d_txt, self.d_txt)
                    self.out_layer = nn.Linear(self.d_txt, output_dim)
                elif  self.cross_method=="Outer":
                    self.outer_fusion=Outer(inp1_size=self.d_txt, inp2_size=self.d_ts)
                    self.proj1 = nn.Linear(self.d_txt, self.d_txt)
                    self.proj2 = nn.Linear(self.d_txt, self.d_txt)
                    self.out_layer = nn.Linear(self.d_txt, output_dim)
                else:
                    self.proj1 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                    self.proj2 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                    self.out_layer = nn.Linear(self.d_ts+self.d_txt, output_dim)         if self.task=='ihm':
            self.loss_fct1=nn.CrossEntropyLoss()
        elif self.task=='pheno':
            self.loss_fct1=nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")     def get_network(self, self_type='ts_mem', layers=-1):
        if self_type == 'ts_mem':
            if self.irregular_learn_emb_ts:
                embed_dim, q_seq_len,kv_seq_len= self.d_ts,self.tt_max, None
            else:
                embed_dim, q_seq_len,kv_seq_len= self.d_ts,  self.ts_seq_num,None
        elif self_type == 'txt_mem':
            if self.irregular_learn_emb_text:
                embed_dim,q_seq_len,kv_seq_len= self.d_txt,self.tt_max, None
            else:
                embed_dim,q_seq_len,kv_seq_len= self.d_txt, self.text_seq_num, None

        elif self_type =='txt_with_ts':
            if self.irregular_learn_emb_ts:
                embed_dim,  q_seq_len,kv_seq_len= self.d_ts,self.tt_max, self.tt_max
            else:

                embed_dim,q_seq_len,kv_seq_len= self.d_ts, self.text_seq_num, self.ts_seq_num
        elif self_type =='ts_with_txt':
            if self.irregular_learn_emb_text:
                embed_dim, q_seq_len,kv_seq_len= self.d_txt, self.tt_max, self.tt_max
            else:
                embed_dim, q_seq_len,kv_seq_len= self.d_txt, self.ts_seq_num, self.text_seq_num
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  device=self.device,
                                  attn_dropout=self.dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask,
                                q_seq_len=q_seq_len,
                                 kv_seq_len=kv_seq_len)

    def get_cross_network(self, layers=-1):
        embed_dim,  q_seq_len= self.d_ts, self.tt_max
        return TransformerCrossEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  device=self.device,
                                  attn_dropout=self.dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask,
                                        q_seq_len_1=q_seq_len) 
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)     def forward(self, x_ts, x_ts_mask, ts_tt_list, input_ids_sequences,
                attn_mask_sequences, note_time_list, note_time_mask_list,labels=None,reg_ts=None):
        """
        dimension [batch_size, seq_len, n_features]

        """
        if "TS" in self.modeltype:

            if self.irregular_learn_emb_ts:
                time_key_ts = self.learn_time_embedding(ts_tt_list).to(self.device)
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                x_ts_irg = torch.cat((x_ts,x_ts_mask), 2)
                x_ts_mask = torch.cat((x_ts_mask,x_ts_mask), 2)

                proj_x_ts_irg=self.time_attn_ts(time_query, time_key_ts, x_ts_irg, x_ts_mask)
                proj_x_ts_irg=proj_x_ts_irg.transpose(0, 1)

            if self.reg_ts and reg_ts!=None:
                x_ts_reg = reg_ts.transpose(1, 2)
                proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts== self.d_ts else self.proj_ts(x_ts_reg)
                proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

            if self.TS_mixup:
                if self.mixup_level=='batch':
                    g_irg=torch.max(proj_x_ts_irg,dim=0).values
                    g_reg =torch.max(proj_x_ts_reg,dim=0).values
                    moe_gate=torch.cat([g_irg,g_reg],dim=-1)
                elif self.mixup_level=='batch_seq' or  self.mixup_level=='batch_seq_feature':
                    moe_gate=torch.cat([proj_x_ts_irg,proj_x_ts_reg],dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")
                mixup_rate=self.moe(moe_gate)
                proj_x_ts=mixup_rate*proj_x_ts_irg+(1-mixup_rate)*proj_x_ts_reg

            else:
                if self.irregular_learn_emb_ts:
                    proj_x_ts=proj_x_ts_irg
                elif self.reg_ts:
                    proj_x_ts=proj_x_ts_reg
                else:
                    raise ValueError("Unknown time series type")
        if "Text" in self.modeltype:
            x_txt=self.bertrep(input_ids_sequences,attn_mask_sequences)
            if self.irregular_learn_emb_text:
                time_key = self.learn_time_embedding(note_time_list).to(self.device)
                if not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                proj_x_txt=self.time_attn(time_query, time_key, x_txt, note_time_mask_list)
                proj_x_txt=proj_x_txt.transpose(0, 1)

            else:
                x_txt = x_txt.transpose(1, 2)
                proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(x_txt)
                proj_x_txt = proj_x_txt.permute(2, 0, 1)
        if self.cross_method=="self_cross":
            hiddens = self.trans_self_cross_ts_txt([proj_x_txt, proj_x_ts])
            h_txt_with_ts, h_ts_with_txt=hiddens
            last_hs = torch.cat([h_txt_with_ts[-1], h_ts_with_txt[-1]], dim=1)

        else:
            if self.cross_method=="MulT":
                # ts --> txt
                h_txt_with_ts = self.trans_txt_with_ts(proj_x_txt, proj_x_ts, proj_x_ts)
                # txt --> ts
                h_ts_with_txt = self.trans_ts_with_txt(proj_x_ts, proj_x_txt, proj_x_txt)
                proj_x_ts = self.trans_ts_mem(h_txt_with_ts)
                proj_x_txt = self.trans_txt_mem(h_ts_with_txt)

                last_h_ts=proj_x_ts[-1]
                last_h_txt=proj_x_txt[-1]
                last_hs = torch.cat([last_h_ts,last_h_txt], dim=1)

            else:
                proj_x_ts = self.trans_ts_mem(proj_x_ts)
                proj_x_txt = self.trans_txt_mem(proj_x_txt)
                if self.cross_method=="MAGGate":
                    last_hs=self.gate_fusion(proj_x_txt[-1],proj_x_ts[-1])
                elif self.cross_method=="Outer":
                    last_hs=self.outer_fusion(proj_x_txt[-1],proj_x_ts[-1])
                else:
                    last_hs = torch.cat([proj_x_txt[-1],proj_x_ts[-1]], dim=1)         last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj) 
        if self.task == 'ihm':
            if labels!=None:
                return self.loss_fct1(output, labels)
            return torch.nn.functional.softmax(output,dim=-1)[:,1]

        elif self.task == 'pheno':
            if labels!=None:
                labels=labels.float()
                return self.loss_fct1(output, labels)
            return torch.nn.functional.sigmoid(output)

class TSMixed(nn.Module):
    def __init__(self,args,device,modeltype=None,orig_d_ts=None,orig_reg_d_ts=None,ts_seq_num=None):

        super(TSMixed, self).__init__()
        if modeltype!=None:
            self.modeltype=modeltype
        else:
            self.modeltype=args.modeltype
        self.num_heads = args.num_heads

        self.attn_mask = False
        self.layers = args.layers
        self.device=device
        self.kernel_size=args.kernel_size
        self.dropout=args.dropout
        self.irregular_learn_emb_ts=args.irregular_learn_emb_ts
        self.irregular_learn_emb_text=args.irregular_learn_emb_text
        self.Interp=args.Interp
        self.reg_ts=args.reg_ts
        self.TS_mixup=args.TS_mixup
        self.mixup_level=args.mixup_level
        self.task=args.task
        self.TS_model=args.TS_model
        self.tt_max=args.tt_max         self.time_query=torch.linspace(0, 1., self.tt_max)
        self.periodic = nn.Linear(1, args.embed_time-1)
        self.linear = nn.Linear(1, 1)

        output_dim = args.num_labels

        self.orig_d_ts=orig_d_ts
        self.d_ts=args.embed_dim
        self.ts_seq_num=ts_seq_num

        if self.Interp:
            self.s_intp=S_Interp(args,self.device,self.orig_d_ts)
            self.c_intp=Cross_Interp(args,self.device,self.orig_d_ts)
            self.proj_ts_intp = nn.Conv1d(self.orig_d_ts*3, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if self.irregular_learn_emb_ts:
            self.time_attn_ts=multiTimeAttention(self.orig_d_ts*2,  self.d_ts, args.embed_time, 8)

        if self.reg_ts:
            self.orig_reg_d_ts=orig_reg_d_ts
            self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if self.TS_mixup:
            if self.mixup_level=='batch':
                self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=self.dropout)
            elif self.mixup_level=='batch_seq':
                self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=self.dropout)
            elif self.mixup_level=='batch_seq_feature':
                self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=self.d_ts,dropout=self.dropout)
            else:
                raise ValueError("Unknown mixedup type")

                # self.moe = nn.Linear(self.d_ts*self.tt_max*2, 1)
        if self.TS_model=='LSTM':
            self.trans_ts_mem=nn.LSTM(input_size=self.d_ts, hidden_size=self.d_ts, num_layers=args.layers,dropout=self.dropout,bidirectional=True)

        elif self.TS_model=='CNN':
            self.trans_ts_mem=TimeSeriesCnnModel(input_size=self.d_ts,n_filters=self.d_ts,filter_size=self.kernel_size,\
            dropout=self.dropout,length=self.tt_max,n_neurons=self.d_ts,layers=args.layers)
        elif self.TS_model=='Atten':
            self.trans_ts_mem = self.get_network(self_type='ts_mem', layers=args.layers)         
        self.proj1 = nn.Linear(self.d_ts, self.d_ts)
        self.proj2 = nn.Linear(self.d_ts, self.d_ts)
        self.out_layer= nn.Linear(self.d_ts, output_dim)

        if self.task=='ihm':
            self.loss_fct1=nn.CrossEntropyLoss()
        elif self.task=='pheno':
            self.loss_fct1=nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def get_network(self, self_type='ts_mem', layers=-1):
        embed_dim=self.d_ts
        if self_type == 'ts_mem':
            if self.irregular_learn_emb_ts :
                q_seq_len= self.tt_max
            else:
                q_seq_len= self.ts_seq_num

        return TransformerEncoder(embed_dim=embed_dim,
                                    num_heads=self.num_heads,
                                    layers=layers,
                                    device=self.device,
                                    attn_dropout=self.dropout,
                                    relu_dropout=self.dropout,
                                    res_dropout=self.dropout,
                                    embed_dropout=self.dropout,
                                    attn_mask=self.attn_mask,
                                q_seq_len=q_seq_len,
                                    kv_seq_len=None)
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x_ts, x_ts_mask, ts_tt_list,labels=None,reg_ts=None):
        """
        dimension [batch_size, seq_len, n_features]

        """

        if "TS" in self.modeltype :

            if self.Interp:
                x_ts_mask_interp=copy.deepcopy(x_ts_mask)
                x_ts_interp=copy.deepcopy(x_ts)
                recon_m=hold_out(x_ts_mask_interp)
                recon_m=torch.Tensor(recon_m).to(self.device)
                proj_x_ts_interp=self.proj_ts_intp(self.c_intp(self.s_intp(x_ts_interp, x_ts_mask_interp, ts_tt_list,recon_m))) #dimension [batch_size,  n_features,seq_len]
                proj_x_ts_interp = proj_x_ts_interp.permute(2, 0, 1)
                recon_interp=self.c_intp(self.s_intp(x_ts_interp, x_ts_mask_interp, ts_tt_list,recon_m, reconstruction=True),reconstruction=True)

            if self.irregular_learn_emb_ts:
                time_key_ts = self.learn_time_embedding(ts_tt_list).to(self.device)
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                x_ts_irg = torch.cat((x_ts,x_ts_mask), 2)
                x_ts_mask = torch.cat((x_ts_mask,x_ts_mask), 2)

                proj_x_ts_irg=self.time_attn_ts(time_query, time_key_ts, x_ts_irg, x_ts_mask)
                proj_x_ts_irg=proj_x_ts_irg.transpose(0, 1)

            if self.reg_ts and reg_ts!=None:
                x_ts_reg = reg_ts.transpose(1, 2)
                proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts== self.d_ts else self.proj_ts(x_ts_reg)
                proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

            if self.TS_mixup:
                if self.Interp and not self.irregular_learn_emb_ts and self.reg_ts:
                    proj_x_ts_irg=proj_x_ts_interp
                if self.Interp and self.irregular_learn_emb_ts and not self.reg_ts :
                    proj_x_ts_reg=proj_x_ts_interp
                # import pdb; pdb.set_trace()
                if self.mixup_level=='batch':
                    g_irg=torch.max(proj_x_ts_irg,dim=0).values
                    g_reg =torch.max(proj_x_ts_reg,dim=0).values
                    moe_gate=torch.cat([g_irg,g_reg],dim=-1)
                elif self.mixup_level=='batch_seq' or  self.mixup_level=='batch_seq_feature':
                    moe_gate=torch.cat([proj_x_ts_irg,proj_x_ts_reg],dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")

                # for name, parameter in self.moe.named_parameters():
                mixup_rate=self.moe(moe_gate)
                proj_x_ts=mixup_rate*proj_x_ts_irg+(1-mixup_rate)*proj_x_ts_reg

            else:
                if self.irregular_learn_emb_ts:
                    proj_x_ts=proj_x_ts_irg
                elif self.reg_ts:
                    proj_x_ts=proj_x_ts_reg
                else:
                    raise ValueError("Unknown time series type")             if self.TS_model=='CNN':
                proj_x_ts = proj_x_ts.permute(1, 2, 0)
                proj_x_ts = self.trans_ts_mem(proj_x_ts)

            elif self.TS_model=='LSTM':
                    _, (proj_x_ts, _) = self.trans_ts_mem(proj_x_ts)
            else:
                proj_x_ts = self.trans_ts_mem(proj_x_ts)
            if  self.TS_model!='CNN':
                last_h_ts=proj_x_ts[-1]

            else:
                last_h_ts=proj_x_ts

 
            if self.modeltype=="TS" :
                last_hs=last_h_ts
            else:
                raise ValueError("Unknown model type")
                       
    #
            last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_h_ts)), p=self.dropout, training=self.training))
            last_hs_proj += last_hs
            output = self.out_layer(last_hs_proj)         if self.Interp:
            reconloss_interp=recon_loss(x_ts_interp,x_ts_mask_interp,recon_m,recon_interp,self.d_ts)

        if self.task == 'ihm':
            if labels!=None:
                if self.Interp:
                    return self.loss_fct1(output, labels)+reconloss_interp
                else:
                    return self.loss_fct1(output, labels)
            return torch.nn.functional.softmax(output,dim=-1)[:,1]

        elif self.task == 'pheno':
            if labels!=None:
                labels=labels.float()
                if self.Interp:
                    return self.loss_fct1(output, labels)+reconloss_interp
                else:
                    return self.loss_fct1(output, labels)
            return torch.nn.functional.sigmoid(output) 
class  Outer(nn.Module):
    def __init__(self,
                 inp1_size: int = 128,
                 inp2_size: int = 128,
                 n_neurons: int = 128):
        super(Outer, self).__init__()
        self.inp1_size = inp1_size
        self.inp2_size = inp2_size
        self.feedforward = nn.Sequential(
            nn.Linear((inp1_size + 1) * (inp2_size + 1), n_neurons),
            nn . ReLU (),
            nn.Linear(n_neurons, n_neurons),
            nn . ReLU (),
        )

    def forward(self, inp1, inp2):
        # import pdb; pdb.set_trace()
        batch_size = inp1.size(0)
        append = torch.ones((batch_size, 1)).type_as(inp1)
        inp1 = torch.cat([inp1, append], dim=-1)
        inp2 = torch.cat([inp2, append], dim=-1)
        fusion = torch.zeros((batch_size, self.inp1_size + 1, self.inp2_size + 1)).type_as(inp1)
        for  i  in  range ( batch_size ):
            fusion[i] = torch.outer(inp1[i], inp2[i])
        fusion = fusion.flatten(1)

        return self.feedforward(fusion) 
class MAGGate(nn.Module):
    def __init__(self, inp1_size, inp2_size, dropout):
        super(MAGGate, self).__init__()

        self.fc1 = nn.Linear(inp1_size + inp2_size, 1)
        self.fc3 = nn.Linear(inp2_size, inp1_size)
        self.beta = nn.Parameter(torch.randn((1,)))
        self.norm = nn.LayerNorm(inp1_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp1, inp2):
        w2 = torch.sigmoid(self.fc1(torch.cat([inp1, inp2], -1)))
        adjust = self.fc3(w2 * inp2)
        one = torch.tensor(1).type_as(adjust)
        alpha = torch.min(torch.norm(inp1) / torch.norm(adjust) * self.beta, one)
        output = inp1 + alpha * adjust
        output = self.dropout(self.norm(output))
        return output class gateMLP(nn.Module):
    def __init__(self,input_dim,hidden_size,output_dim,dropout=0.1):
        super().__init__()

        self.gate = nn.Sequential(
             nn.Dropout(dropout),
             nn.Linear(input_dim, hidden_size),
             nn.ReLU(),
             nn.Linear(hidden_size,output_dim),
             nn.Sigmoid()
        )         self._initialize()

    def _initialize(self):
        for model in [self.gate]:
            for layer in model:
                if type(layer) in [nn.Linear]:
                    torch.nn.init.xavier_normal_(layer.weight)     def forward(self,hidden_states ):
        gate_logits = self.gate(hidden_states)
        return gate_logits

class TimeSeriesCnnModel(nn.Module):
    def __init__(self,input_size,n_filters,filter_size,dropout,length,n_neurons,layers):
        super().__init__()

        padding = int(np.floor(filter_size / 2))
        self.layers=layers
        if layers>=1:
            self.conv1 = nn.Conv1d(input_size, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)

        if layers>=2:
            self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool2 = nn.MaxPool1d(2, 2)

        if layers>=3:
            self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool3 = nn.MaxPool1d(2, 2)

        self.fc1 = nn.Linear(int(length * n_filters / (2**layers)), n_neurons)
        self.fc1_drop = nn.Dropout(dropout)     def forward(self, x):
        if self.layers>=1:
            x = self.pool1(F.relu(self.conv1(x)))
        if self.layers>=2:
            x = self.pool2(F.relu(self.conv2(x)))
        if self.layers>=3:
            x = self.pool3(F.relu(self.conv3(x)))
        # import pdb; pdb.set_trace()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_drop(self.fc1(x)))

        return x

# F.gumbel_softmax(logits, tau=1, hard=True)

class multiTimeAttention(nn.Module):

    def __init__(self, input_dim, nhidden=16,
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"

        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            if len(mask.shape)==3:
                mask=mask.unsqueeze(-1)

            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -10000)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn=F.dropout(p_attn, p=dropout, training=self.training)
#             p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn     def forward(self, query, key, value, mask=None, dropout=0.1):
        "Compute 'Scaled Dot Product Attention'"
        # import pdb; pdb.set_trace()
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        return self.linears[-1](x)  

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        # import pdb;
        # pdb.set_trace()
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias) 

class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, device,attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False,learn_embed=True, q_seq_len=None, kv_seq_len=None,):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.device=device
        self.q_seq_len=q_seq_len
        self.kv_seq_len=kv_seq_len
        if learn_embed:
            if self.q_seq_len!=None:
                self.embed_positions_q=nn.Embedding(self.q_seq_len,embed_dim,padding_idx=0)
                nn.init.normal_(self.embed_positions_q.weight, std=0.02)

            if self.kv_seq_len!=None:
                self.embed_positions_kv=nn.Embedding(self.kv_seq_len,embed_dim)
                nn.init.normal_(self.embed_positions_kv.weight, std=0.02)

        else:
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)

        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k = None, x_in_v = None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """         x=x_in
        length_x = x.size(0) # (length,Batch_size,input_dim)
        x = self.embed_scale * x_in
        if self.q_seq_len is not None:
            position_x = torch.tensor(torch.arange(length_x),dtype=torch.long).to(self.device)
            x += (self.embed_positions_q(position_x).unsqueeze(0)).transpose(0,1)  # Add positional embedding
        x =F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions

            length_kv = x_in_k.size(0) # (Batch_size,length,input_dim)
            position_kv = torch.tensor(torch.arange(length_kv),dtype=torch.long).to(self.device)

            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.kv_seq_len is not None:
                x_k += (self.embed_positions_kv(position_kv).unsqueeze(0)).transpose(0,1)   # Add positional embedding
                x_v += (self.embed_positions_kv(position_kv).unsqueeze(0)).transpose(0,1)   # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)         # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

class TransformerCrossEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerCrossEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, device,attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False,q_seq_len_1=None,q_seq_len_2=None):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.device=device

        self.q_seq_len_1=q_seq_len_1
        self.q_seq_len_2=q_seq_len_2
        # self.intermediate=intermediate
        self.embed_positions_q_1=nn.Embedding(self.q_seq_len_1,embed_dim,padding_idx=0)
        nn.init.normal_(self.embed_positions_q_1.weight, std=0.02)

        if self.q_seq_len_2!= None:
            self.embed_positions_q_2=nn.Embedding(self.q_seq_len_2,embed_dim,padding_idx=0)
            nn.init.normal_(self.embed_positions_q_2.weight, std=0.02)

            self.embed_positions_q=nn.ModuleList([self.embed_positions_q_1,self.embed_positions_q_2])
        else:
            self.embed_positions_q=nn.ModuleList([self.embed_positions_q_1,self.embed_positions_q_1,])         self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerCrossEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.normalize = True
        if self.normalize:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(2)])

    def forward(self, x_in_list):
        """
        Args:
            x_in_list (list of FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the list of last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`

        """

        # import pdb;
        # pdb.set_trace()
        x_list=x_in_list
        length_x1 = x_list[0].size(0) # (length,Batch_size,input_dim)
        length_x2 = x_list[1].size(0)
        x_list = [ self.embed_scale * x_in for x_in in x_in_list]
        if self.q_seq_len_1 is not None:
            position_x1 = torch.tensor(torch.arange(length_x1),dtype=torch.long).to(self.device)
            position_x2 = torch.tensor(torch.arange(length_x2),dtype=torch.long).to(self.device)
            positions=[position_x1 ,position_x2]
            x_list=[ l(position_x).unsqueeze(0).transpose(0,1) +x for l, x,position_x in zip(self.embed_positions_q, x_list,positions)]
              # Add positional embedding
        x_list[0]=F.dropout(x_list[0], p=self.dropout, training=self.training)
        x_list[1]=F.dropout(x_list[1], p=self.dropout, training=self.training)

        # encoder layers

        # x_low_level=None         for layer in self.layers:
            x_list= layer(x_list) #proj_x_txt, proj_x_ts         if self.normalize:
            x_list=[ l(x)  for l, x in zip(self.layer_norm, x_list)]
        return x_list 

class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                     attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pre_self_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

        self.self_attns = nn.ModuleList([MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        ) for _ in range(2)])

        self.post_self_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])         self.pre_encoder_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

        self.cross_attn_1 = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )

        self.cross_attn_2 = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )

        self.post_encoder_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.pre_ffn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])
        self.fc1 =  nn.ModuleList([nn.Linear(self.embed_dim, 4*self.embed_dim) for _ in range(2)])  # The "Add & Norm" part in the paper
        self.fc2 = nn.ModuleList([nn.Linear(4*self.embed_dim, self.embed_dim) for _ in range(2)])
        self.pre_ffn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])     def forward(self, x_list):
        """
        Args:
            x (List of Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
        Returns:
            list of encoded output of shape `(batch, src_len, embed_dim)`
        """
        ###self attn
        residual = x_list

        x_list = [l(x) for l, x in zip(self.pre_self_attn_layer_norm, x_list)]

        output= [l(query=x, key=x, value=x) for l, x in zip(self.self_attns, x_list)]

        x_list=[ x for x, _ in output]

        x_list[0]=F.dropout(x_list[0], p=self.res_dropout , training=self.training)
        x_list[1]=F.dropout(x_list[1], p=self.res_dropout , training=self.training)

        x_list = [r + x  for r, x in zip(residual, x_list) ]
#         x_list = [l(x) for l, x in zip(self.post_self_attn_layer_norm, x_list)]

        #### cross attn

        residual=x_list
        x_list = [l(x) for l, x in zip(self.pre_encoder_attn_layer_norm, x_list)]
        x_txt,x_ts=  x_list #proj_x_txt, proj_x_ts

        # cross: ts -> txt
        x_ts_to_txt,_=self.cross_attn_1(query=x_txt, key=x_ts, value=x_ts)
        # cross:  txt->ts
        x_txt_to_ts,_=self.cross_attn_2(query=x_ts, key=x_txt, value=x_txt)

        # else:
        #     x_low_level = [l(x) for l, x in zip(self.pre_encoder_attn_layer_norm, x_low_level)]
        #     x_txt_low,x_ts_low=  x_low_level
        #     # cross: ts -> txt
        #     x_ts_to_txt,_=self.cross_attn_1(query=x_txt, key=x_ts_low, value=x_ts_low)
        #     # cross:  txt->ts
        #     x_txt_to_ts,_=self.cross_attn_2(query=x_ts, key=x_txt_low, value=x_txt_low)         x_ts_to_txt  = F.dropout(x_ts_to_txt, p=self.res_dropout, training=self.training)
        x_txt_to_ts  = F.dropout(x_txt_to_ts, p=self.res_dropout, training=self.training)

        x_list = [r+ x for r, x in zip(residual, (x_ts_to_txt, x_txt_to_ts))]

#         x_list = [l(x) for l, x in zip(self.post_encoder_attn_layer_norm, x_list)]

        # FNN
        residual = x_list
        x_list = [l(x) for l, x in zip(self.pre_ffn_layer_norm, x_list)]
        x_list = [F.relu(l(x)) for l, x in zip(self.fc1, x_list)]

        x_list[0]=F.dropout(x_list[0], p=self.relu_dropout , training=self.training)
        x_list[1]=F.dropout(x_list[1], p=self.relu_dropout , training=self.training)

        x_list = [l(x) for l, x in zip(self.fc2, x_list)]

        x_list[0]=F.dropout(x_list[0], p=self.res_dropout, training=self.training)
        x_list[1]=F.dropout(x_list[1], p=self.res_dropout, training=self.training)

        x_list = [r + x  for r, x in zip(residual, x_list) ]

#         x_list = [l(x) for l, x in zip(self.post_ffn_layer_norm, x_list)]         return x_list  
class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:bpbpp
            encoded output of shape `(batch, src_len, embed_dim)`
        """

        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
#     nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)

    return m def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t) def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]

   # Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree. 

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024, auto_expand=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.register_buffer(
            "weights",
            SinusoidalPositionalEmbedding.get_embedding(
                init_size, embedding_dim, padding_idx
            ),
            persistent=False,
        )
        self.max_positions = int(1e5)
        self.auto_expand = auto_expand
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Ignore some deprecated keys that were used in older versions
        deprecated_keys = ["weights", "_float_tensor"]
        for key in deprecated_keys:
            if prefix + key in state_dict:
                del state_dict[prefix + key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        weights = self.weights

        if max_pos > self.weights.size(0):
            # If the input is longer than the number of pre-computed embeddings,
            # compute the extra embeddings on the fly.
            # Only store the expanded embeddings if auto_expand=True.
            # In multithreading environments, mutating the weights of a module
            # may cause trouble. Set auto_expand=False if this happens.
            weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            ).to(self.weights)
            if self.auto_expand:
                self.weights = weights

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = utils.make_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace
        )
        if self.onnx_trace:
            flat_embeddings = weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()
        )  def hold_out(mask, perc=0.2):
    """To implement the autoencoder component of the loss, we introduce a set
    of masking variables mr (and mr1) for each data point. If drop_mask = 0,
    then we removecthe data point as an input to the interpolation network,
    and includecthe predicted value at this time point when assessing
    the autoencoder loss. In practice, we randomly select 20% of the
    observed data points to hold out from
    every input time series."""

    mask=mask.cpu().detach().numpy()
    drop_mask = np.ones_like(mask)
    drop_mask *= mask
    for  i  in  range ( mask .shape[ 0 ]):
        for j in range(mask.shape[1]):
            count = np.sum(mask[i, j], dtype='int')
            if int(0.20*count) > 1:
                index = 0
                r = np.ones((count, 1))
                b = np.random.choice(count, int(0.20*count), replace=False)
                r[b] = 0
                for k in range(mask.shape[2]):
                    if mask[i, j, k] > 0:
                        drop_mask[i, j, k] = r[index]
                        index += 1
    return drop_mask def recon_loss(x_ts, m1,m2,ypred,num_features):
    """ Autoencoder loss
    """
    # standard deviation of each feature mentioned in paper for MIMIC_III data
    # wc = np.array([3.33, 23.27, 5.69, 22.45, 14.75, 2.32,
    #                3.75, 1.0, 98.1, 23.41, 59.32, 1.41])
    # wc.shape = (1, num_features)
    y=x_ts.transpose(1,2)
    m1=m1.transpose(1,2)
    m2=m2.transpose(1,2)
    m2 = 1 - m2
    m = m1*m2
    ypred = ypred[:, :num_features, :]
    x = (y - ypred)*(y - ypred)
    x = x*m
    count = torch.sum(m, dim=2)
    count = torch.where(count > 0, count,torch.ones_like(count))
    x = torch.sum(x, dim=2)/count
    # x = x/(wc**2)  # dividing by standard deviation
    x = torch.sum(x, dim=1)/num_features
    return torch.mean(x) class S_Interp(nn.Module):
    def __init__( self,args,device,orig_d_ts):
        super(S_Interp, self).__init__()

        self.tt_max=args.tt_max
        self.device=device
        self.ref_t=torch.linspace(0, 1., self.tt_max).to(self.device)
        self.d_dim=orig_d_ts
        self.output = nn.Linear(args.embed_dim, args.embed_dim)
        self.kernel=  Parameter(torch.zeros(self.d_dim))

    def forward(self, x_ts, x_ts_mask, ts_tt_list,rec_mask,reconstruction=False):
        x_ts=x_ts.transpose(1,2)
        x_ts_mask=x_ts_mask.transpose(1,2)
        tt_len=ts_tt_list.shape[-1]
        d=ts_tt_list.unsqueeze(1)
        d=d.repeat(1,self.d_dim,1)
        if reconstruction:
            output_dim = tt_len
            m = rec_mask.transpose(1,2)
            ref_t=d.unsqueeze(-2).repeat(1,1,output_dim, 1)
            # ts_tt_list
            # ref_t = K.tile(d[:, :, None, :], (1, 1, output_dim, 1))

        else:
            m = x_ts_mask
            ref_t = self.ref_t.unsqueeze(0)
            output_dim = self.tt_max
        # import pdb; pdb.set_trace()
        d = d.unsqueeze(-1).repeat(1,1, 1,output_dim)
        mask =m.unsqueeze(-1).repeat(1,1, 1,output_dim)
        x_ts=x_ts.unsqueeze(-1).repeat(1,1,1,output_dim)         norm = (d - ref_t)*(d - ref_t)
        a= torch.ones([self.d_dim,tt_len,output_dim]).to(self.device)

        pos_kernel = torch.log(1 + torch.exp(self.kernel))
        alpha =a*pos_kernel.unsqueeze(-1).unsqueeze(-1)
        w = torch.logsumexp(-alpha*norm + torch.log(mask+1e-12), dim=2)
        w1=w.unsqueeze(2).repeat(1,1,tt_len,1)
        w1 = torch.exp(-alpha*norm + torch.log(mask+1e-12)- w1)
        y = torch.sum(w1*x_ts, dim=2)
#
        w_t = torch.logsumexp(-10.0*alpha*norm + torch.log(mask+1e-12),
                          dim=2)  # kappa = 10
        w_t=w.unsqueeze(2).repeat(1,1,tt_len,1)

        w_t = torch.exp(-10.0*alpha*norm + torch.log(mask+1e-12) - w_t)
        y_trans = torch.sum(w_t*x_ts, dim=2)
        rep1 = torch.cat([y, w, y_trans], dim= 1)

        return rep1
class Cross_Interp(nn.Module):
    def __init__( self,args,device,orig_d_ts):
        super(Cross_Interp, self).__init__()
        self.device=device
        self.d_dim=orig_d_ts
        self.activation = nn.Sigmoid()
        self.cross_channel_interp =torch.empty(self.d_dim, self.d_dim).to(self.device)
        nn.init.eye_(self.cross_channel_interp)     def forward(self, x,reconstruction=False):
        self.output_dim = x.shape[-1]
        cross_channel_interp = self.cross_channel_interp
        y = x[:, :self.d_dim, :]
        w = x[:, self.d_dim:2*self.d_dim, :] #x
        intensity = torch.exp(w)
        y = y.permute(0, 2, 1)
        w = w.permute(0, 2, 1)
        w2  =  w
        w=w.unsqueeze(-1).repeat(1,1,1,self.d_dim)
        den = torch.logsumexp(w, dim=2)
        w = torch.exp(w2 - den)
        mean = torch.mean(y, dim=1)
        mean = mean.unsqueeze(1).repeat(1,self.output_dim,1)
        w2 = torch.matmul(w*(y - mean), cross_channel_interp) + mean
        rep1 = w2.permute(0, 2, 1)
        if reconstruction is False:
            y_trans = x[:, 2*self.d_dim:3*self.d_dim, :]
            y_trans = y_trans - rep1  # subtracting smooth from transient part
            rep1 = torch.cat([rep1, intensity, y_trans], 1)
        return rep1