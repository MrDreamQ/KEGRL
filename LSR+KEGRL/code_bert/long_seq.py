import torch
import torch.nn.functional as F
import numpy as np

    

def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    sequence_output = bert_forward(model, input_ids, attention_mask, start_tokens, end_tokens)
    
    return sequence_output
    
def bert_forward(model, input_ids, attention_mask, start_tokens, end_tokens):
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    
    # embedding_output = model.embeddings.word_embeddings(input_ids)
    
    new_input_ids, new_attention_mask, num_seg = [], [], []
    seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
    for i, l_i in enumerate(seq_len):   # 遍历文档
        # 去除起始token与终止token，并将长度超过512的输入分隔并记录起止位置
        bert_indexs = [(bert_s, min(l_i - len_end, bert_s + 512-(len_start+len_end))) for bert_s in
                        range(len_start, l_i - len_end, 512-(len_start+len_end))]
        num_seg.append(len(bert_indexs))
        for j, (bert_s, bert_e) in enumerate(bert_indexs):  # 遍历文档的分文档
            if j == len(bert_indexs) - 1:   # 是否当前文档的最后一个分文档
                if l_i <= 512:
                    new_input_ids.append(torch.cat([start_tokens,
                                                    input_ids[i, len_start: min(512-len_end, c-len_end)],
                                                    end_tokens],
                                                    dim=0))
                    new_attention_mask.append(attention_mask[i, :512])
                else:
                    new_input_ids.append(torch.cat([start_tokens,
                                                    input_ids[i, bert_e - 512 + len_start + len_end: bert_e],
                                                    end_tokens],
                                                    dim=0))
                    new_attention_mask.append(attention_mask[i, bert_e - 512 + len_end:bert_e + len_end])
            else:
                new_input_ids.append(torch.cat([start_tokens,
                                                input_ids[i, bert_s: bert_e],
                                                end_tokens],
                                                dim=0))
                new_attention_mask.append(attention_mask[i, bert_s - len_start:bert_e + len_end])
    input_ids = torch.stack(new_input_ids, dim=0)
    attention_mask = torch.stack(new_attention_mask, dim=0)
    output = model(input_ids, attention_mask=attention_mask)
    sequence_output = output[0]
    
    sequence_output = _re_cz(num_seg, seq_len, c, sequence_output, attention_mask, len_start, len_end)
    
    return sequence_output
    
def _re_cz(num_seg, seq_len, c, context_output, attention_mask, len_start, len_end):
        i = 0
        re_context_output = []
        for n_seg, l_i in zip(num_seg, seq_len):
            if l_i <= 512:
                assert n_seg == 1
                if c <= 512:
                    re_context_output.append(context_output[i])
                else:
                    context_output1 = F.pad(context_output[i, :512, :], (0, 0, 0, c-512))
                    re_context_output.append(context_output1)
            else:
                context_output1 = []
                mask1 = []
                for j in range(i, i + n_seg - 1):
                    if j == i:
                        context_output1.append(context_output[j][:512 - len_end, :])
                        mask1.append(attention_mask[j][:512 - len_end])
                    else:
                        context_output1.append(context_output[j][len_start:512 - len_end, :])
                        mask1.append(attention_mask[j][len_start:512 - len_end])

                context_output1 = F.pad(torch.cat(context_output1, dim=0),
                                            (0, 0, 0, c - (n_seg - 1) * (512 - len_end) + (n_seg - 2) * len_start))

                context_output2 = context_output[i + n_seg - 1][len_start:]
                context_output2 = F.pad(context_output2, (0, 0, l_i - 512 + len_start, c - l_i))

                mask1 = F.pad(torch.cat(mask1, dim=0), (0, c - (n_seg - 1) * (512 - len_end) + (n_seg - 2) * len_start))
                mask2 = attention_mask[i + n_seg - 1][len_start:]
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                mask = mask1 + mask2 + 1e-10
                context_output1 = (context_output1 + context_output2) / mask.unsqueeze(-1)
                re_context_output.append(context_output1)

            i += n_seg
        context_output = torch.stack(re_context_output, dim=0)
        return context_output