import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import json


def show_confuse_mat(confuse_mat, save_path, id2rel):
    norm_mat = np.zeros_like(confuse_mat)
    norm_mat[0, 0] = 1
    norm_mat[0, 1:] = confuse_mat[0, 1:] / confuse_mat[0, 1:].sum()
    norm_mat[1:, :] = confuse_mat[1:, :] / (np.expand_dims(confuse_mat[1:, :].sum(axis=-1), axis=-1) + 1e-10)

    xy_tick = [id2rel[i] for i in range(len(id2rel))] 
    confuse_metric = {
        'n2n': round(confuse_mat[0, 0]),
        'n2p': round(confuse_mat[0, 1:].sum()),
        'p2n': round(confuse_mat[1:, 0].sum()),
        'p2pp': round((confuse_mat[1:, 1:] * np.eye(confuse_mat.shape[0] - 1)).sum()),
        'p2np': round((confuse_mat[1:, 1:] * (1 - np.eye(confuse_mat.shape[0] - 1))).sum()),
    }
    sns.heatmap(data=norm_mat, cmap='rainbow', vmax=1, vmin=0, center=0, annot=True, 
                annot_kws={'size': 6}, square=True, linewidths=0,
                cbar_kws={"shrink":.6}, xticklabels=xy_tick, yticklabels=xy_tick, fmt='.2f')
    title_str = json.dumps({'confuse_metric': confuse_metric})
                
    plt.title(title_str)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(save_path, dpi=300)
    print('save as %s' % save_path)
    plt.clf()