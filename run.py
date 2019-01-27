from sklearn import manifold, datasets
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from matplotlib.patches import Ellipse
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from wrapper import Wrapper
# from tsne import TSNE
import os
from vtsne import VTSNE
import pdb
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--save_dir', default='scatter',metavar='DIR',
                    help='path to dataset')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batchsize', default=100, type=int,
                    metavar='N',
                    help='batch_size for calculating tsne every time')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight_decay', default=0.05, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-s', '--save_freq', default=100, type=int,
                    metavar='N', help='save model frequency and lr decay frequency')
parser.add_argument('--perplexity', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
args = parser.parse_args()






def preprocess(perplexity=args.perplexity, metric='euclidean'):
    """ Compute pairiwse probabilities for MNIST pixels.
    """
    digits = datasets.load_digits(n_class=6)
    pos = digits.data
    y = digits.target
    # pos = np.load("/mnt/storage/team03/finetune/tsnefeatures_clean_train_97_model_best.npy")
    # y = np.load("/mnt/storage/team03/finetune/tsnelabels_clean_train_97_model_best.npy")

    n_points = pos.shape[0]
    distances2 = pairwise_distances(pos, metric=metric, squared=True)
    # This return a n x (n-1) prob array
    pij = manifold.t_sne._joint_probabilities(distances2, perplexity, False)
    # Convert to n x n prob array
    pij = squareform(pij)
    return n_points, pij, y

if not os.path.exists(args.save_dir):
	os.mkdir(args.save_dir)
draw_ellipse = True
n_points, pij2d, y = preprocess()
i, j = np.indices(pij2d.shape)
i = i.ravel()
j = j.ravel()
pij = pij2d.ravel().astype('float32')
# Remove self-indices
idx = i != j
i, j, pij = i[idx], j[idx], pij[idx]

n_topics = 2
n_dim = 2
print(n_points, n_dim, n_topics)

model = VTSNE(n_points, n_topics, n_dim)
wrap = Wrapper(model, batchsize = args.batchsize, epochs = args.epochs, lr = args.lr)
for itr in range(1000):
    # wrap.optimizer = wrap.optimizer.param_groups[0]['lr']*0.05
    print('Iteration: ', itr )
    wrap.fit(pij, i, j)

    # Visualize the results
    embed = model.logits.weight.cpu().data.numpy()
    f = plt.figure()
    if not draw_ellipse:
        plt.scatter(embed[:, 0], embed[:, 1], c=y * 1.0 / y.max())
        plt.axis('off')
        plt.savefig(args.save_dir+'/scatter_{:03d}.png'.format(itr), bbox_inches='tight')
        plt.close(f)
    else:
        # Visualize with ellipses
        var = np.sqrt(model.logits_lv.weight.clone().exp_().cpu().data.numpy())
        ax = plt.gca()
        for xy, (w, h), c in zip(embed, var, y):
            e = Ellipse(xy=xy, width=w, height=h, ec=None, lw=0.0)
            e.set_facecolor(plt.cm.Paired(c * 1.0 / y.max()))
            e.set_alpha(0.5)
            ax.add_artist(e)
        ax.set_xlim(-9, 9)
        ax.set_ylim(-9, 9)
        plt.axis('off')
        plt.savefig(args.save_dir+'/scatter_{:03d}.png'.format(itr), bbox_inches='tight')
        plt.close(f)
    if itr%args.save_freq== 0 :
    	torch.save(model,args.save_dir+'/checkpoint.pth.tar') 
    	if itr>0:
            wrap.optimizer.param_groups[0]['lr'] = wrap.optimizer.param_groups[0]['lr']*args.weight_decay



