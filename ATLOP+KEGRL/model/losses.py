import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import cfg


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params:
        num: int, the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += (0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)).squeeze()
        return loss_sum


class ReBCELoss(nn.Module):
    def __init__(self, threshold=0.4):
        super().__init__()
        self.threshold = threshold
        self.loss_fnt = nn.BCELoss()

    def forward(self, logits, labels):
        if cfg.USE_RE_BCE:
            labels = labels[:, 1:]
        loss = self.loss_fnt(torch.sigmoid(logits), labels)
        return loss

    def get_label(self, logits):
        logits = torch.sigmoid(logits)
        logits = (logits > self.threshold).int()
        logits = torch.cat([torch.zeros(logits.shape[0], 1).to(logits), logits], dim=-1)
        logits[:, 0] = (logits.sum(-1) == 0)
        return logits


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, is_dwie, entity_mask):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        if is_dwie:
            cnt = 0
            for h in range(len(entity_mask)):
                for t in range(len(entity_mask)):
                    if not entity_mask[h] or not entity_mask[t]:
                        loss[h * len(entity_mask) + t] = 0
                    else:
                        cnt += 1
            loss = loss.sum() / cnt
        else:
            loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output


class AFLoss(nn.Module):
    def __init__(self, gamma_pos, gamma_neg):
        super().__init__()
        threshod = nn.Threshold(0, 0)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg


    def forward(self, logits, labels):
        # Adapted from Focal loss https://arxiv.org/abs/1708.02002, multi-label focal loss https://arxiv.org/abs/2009.14119
        # TH label 
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0
        label_idx = labels.sum(dim=1)

        two_idx = torch.where(label_idx==2)[0]
        pos_idx = torch.where(label_idx>0)[0]

        neg_idx = torch.where(label_idx==0)[0]
     
        p_mask = labels + th_label
        n_mask = 1 - labels
        neg_target = 1- p_mask
        
        num_ex, num_class = labels.size()
        num_ent = int(np.sqrt(num_ex))
        # Rank each positive class to TH
        logit1 = logits - neg_target * 1e30
        logit0 = logits - (1 - labels) * 1e30

        # Rank each class to threshold class TH
        th_mask = torch.cat( num_class * [logits[:,:1]], dim=1)
        logit_th = torch.cat([logits.unsqueeze(1), 1.0 * th_mask.unsqueeze(1)], dim=1) 
        log_probs = F.log_softmax(logit_th, dim=1)
        probs = torch.exp(F.log_softmax(logit_th, dim=1))

        # Probability of relation class to be positive (1)
        prob_1 = probs[:, 0 ,:]
        # Probability of relation class to be negative (0)
        prob_0 = probs[:, 1 ,:]
        prob_1_gamma = torch.pow(prob_1, self.gamma_neg)
        prob_0_gamma = torch.pow(prob_0, self.gamma_pos)
        log_prob_1 = log_probs[:, 0 ,:]
        log_prob_0 = log_probs[:, 1 ,:]
        
        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        rank2 = F.log_softmax(logit2, dim=-1)

        loss1 = - (log_prob_1 * (1 + prob_0_gamma ) * labels) 
        
        loss2 = -(rank2 * th_label).sum(1) 


        

        loss =  1.0 * loss1.sum(1).mean() + 1.0 * loss2.mean()
        

        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1) * 1.0
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

    
