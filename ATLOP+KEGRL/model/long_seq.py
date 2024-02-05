import torch
import torch.nn.functional as F
import numpy as np

from config import cfg


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    if cfg.DATA.DATASET in ['DocRED', 'Re-DocRED']:
        sequence_output, attention = docred_process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens)
    elif cfg.DATA.DATASET == 'DWIE':
        sequence_output, attention = dwie_process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens)
    else:
        raise NotImplementedError

    return sequence_output, attention

def docred_process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= 512:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat(
                    [input_ids[i, :512 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat(
                    [start_tokens, input_ids[i, (l_i - 512 + len_start):l_i]],
                    dim=-1)
                attention_mask1 = attention_mask[i, :512]
                attention_mask2 = attention_mask[i, (l_i - 512):l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
        i = 0
        new_output, new_attention = [], []

        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:
                output1 = sequence_output[i][:512 - len_end]
                mask1 = attention_mask[i][:512 - len_end]
                att1 = attention[i][:, :512 - len_end, :512 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                att1 = F.pad(att1,
                             (0, c - 512 + len_end, 0, c - 512 + len_end))

                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                output2 = F.pad(output2,
                                (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                att2 = F.pad(att2, [
                    l_i - 512 + len_start, c - l_i, l_i - 512 + len_start,
                    c - l_i
                ])

                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
                    
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)

    return sequence_output, attention


def dwie_process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    
    embedding_output = model.embeddings.word_embeddings(input_ids)
    
    new_input_ids, new_attention_mask, num_seg = [], [], []
    seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
    for i, l_i in enumerate(seq_len):
        bert_indexs = [(bert_s, min(l_i - len_end, bert_s + 512-(len_start+len_end))) for bert_s in
                        range(len_start, l_i - len_end, 512-(len_start+len_end))]
        num_seg.append(len(bert_indexs))
        for j, (bert_s, bert_e) in enumerate(bert_indexs):
            if j == len(bert_indexs) - 1:
                if l_i <= 512:
                    new_input_ids.append(torch.cat([model.embeddings.word_embeddings(start_tokens),
                                                    embedding_output[i, len_start: min(512-len_end, c-len_end)],
                                                    model.embeddings.word_embeddings(end_tokens)],
                                                    dim=0))
                    new_attention_mask.append(attention_mask[i, :512])
                else:
                    new_input_ids.append(torch.cat([model.embeddings.word_embeddings(start_tokens),
                                                    embedding_output[i, bert_e - 512 + len_start + len_end: bert_e],
                                                    model.embeddings.word_embeddings(end_tokens)],
                                                    dim=0))
                    new_attention_mask.append(attention_mask[i, bert_e - 512 + len_end:bert_e + len_end])
            else:
                new_input_ids.append(torch.cat([model.embeddings.word_embeddings(start_tokens),
                                                embedding_output[i, bert_s: bert_e],
                                                model.embeddings.word_embeddings(end_tokens)],
                                                dim=0))
                new_attention_mask.append(attention_mask[i, bert_s - len_start:bert_e + len_end])
    embedding_output = torch.stack(new_input_ids, dim=0)
    attention_mask = torch.stack(new_attention_mask, dim=0)
    output = model(attention_mask=attention_mask, inputs_embeds=embedding_output)
    sequence_output = output[0]
    attention = output[-1][-1]
    
    sequence_output, attention = _re_cz(num_seg, seq_len, c, sequence_output, attention, attention_mask, len_start, len_end)
    
    return sequence_output, attention
    
    
def _re_cz(num_seg, seq_len, c, context_output, attention, attention_mask, len_start, len_end):
        i = 0
        re_context_output = []
        re_attention = []
        for n_seg, l_i in zip(num_seg, seq_len):
            if l_i <= 512:
                assert n_seg == 1
                if c <= 512:
                    re_context_output.append(context_output[i])
                    re_attention.append(attention[i])
                else:
                    context_output1 = F.pad(context_output[i, :512, :], (0, 0, 0, c-512))
                    re_context_output.append(context_output1)
                    attention1 = F.pad(attention[i][:, :512, :512], (0, c-512, 0, c-512))
                    re_attention.append(attention1)
            else:
                context_output1 = []
                attention1 = None
                mask1 = []
                for j in range(i, i + n_seg - 1):
                    if j == i:
                        context_output1.append(context_output[j][:512 - len_end, :])
                        attention1 = F.pad(attention[j][:, :512-len_end, :512-len_end], (0, c-(512-len_end), 0, c-(512-len_end)))
                        mask1.append(attention_mask[j][:512 - len_end])
                    else:
                        context_output1.append(context_output[j][len_start:512 - len_end, :])
                        attention2 = F.pad(attention[j][:, len_start:512-len_end, len_start:512-len_end],
                                                        (512-len_end+(j-i-1)*(512-len_end-len_start), c-(512-len_end+(j-i)*(512-len_end-len_start)),
                                                         512-len_end+(j-i-1)*(512-len_end-len_start), c-(512-len_end+(j-i)*(512-len_end-len_start))))
                        if attention1 is None:
                            attention1 = attention2
                        else:
                            attention1 = attention1 + attention2
                        mask1.append(attention_mask[j][len_start:512 - len_end])

                context_output1 = F.pad(torch.cat(context_output1, dim=0),
                                            (0, 0, 0, c - (n_seg - 1) * (512 - len_end) + (n_seg - 2) * len_start))
                att = attention1 + F.pad(attention[i + n_seg - 1][:, len_start:, len_start:],
                                         (l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i))

                context_output2 = context_output[i + n_seg - 1][len_start:]
                context_output2 = F.pad(context_output2, (0, 0, l_i - 512 + len_start, c - l_i))

                mask1 = F.pad(torch.cat(mask1, dim=0), (0, c - (n_seg - 1) * (512 - len_end) + (n_seg - 2) * len_start))
                mask2 = attention_mask[i + n_seg - 1][len_start:]
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                mask = mask1 + mask2 + 1e-10
                context_output1 = (context_output1 + context_output2) / mask.unsqueeze(-1)
                re_context_output.append(context_output1)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                re_attention.append(att)

            i += n_seg
        attention = torch.stack(re_attention, dim=0)
        context_output = torch.stack(re_context_output, dim=0)
        return context_output, attention