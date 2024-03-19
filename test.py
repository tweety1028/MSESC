import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import utils
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from metrics.cluster.accuracy import clustering_accuracy
import argparse
import random
import os
from dataloader import load_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):

    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=False):
        super(MLP, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.output_dims = out_dims
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.input_dims, self.hid_dims[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(hid_dims) - 1):
            self.layers.append(nn.Linear(self.hid_dims[i], self.hid_dims[i + 1]))
            self.layers.append(nn.ReLU())

        self.out_layer = nn.Linear(self.hid_dims[-1], self.output_dims)
        if kaiming_init:
            self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
                init.zeros_(layer.bias)
        init.xavier_uniform_(self.out_layer.weight)
        init.zeros_(self.out_layer.bias)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
        h = self.out_layer(h)
        h = torch.tanh_(h)
        return h


class AdaptiveSoftThreshold(nn.Module):
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.register_parameter("bias", nn.Parameter(torch.from_numpy(np.zeros(shape=[self.dim])).float()))

    def forward(self, c):
        return torch.sign(c) * torch.relu(torch.abs(c) - self.bias)


class MSESC(nn.Module):

    def __init__(self, input_dims, hid_dims, out_dims, data_size, kaiming_init=True):
        super(MSESC, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        self.kaiming_init = kaiming_init
        self.shrink = 1.0 / out_dims

        self.net_q1 = MLP(input_dims=self.input_dims[0],
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims,
                         kaiming_init=self.kaiming_init)


        self.net_q2 = MLP(input_dims=self.input_dims[1],
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims,
                         kaiming_init=self.kaiming_init)

        self.net_q3 = MLP(input_dims=self.input_dims[2],
                          hid_dims=self.hid_dims,
                          out_dims=self.out_dims,
                          kaiming_init=self.kaiming_init)

        self.net_q4 = MLP(input_dims=self.input_dims[3],
                          hid_dims=self.hid_dims,
                          out_dims=self.out_dims,
                          kaiming_init=self.kaiming_init)

        self.net_q5 = MLP(input_dims=self.input_dims[4],
                          hid_dims=self.hid_dims,
                          out_dims=self.out_dims,
                          kaiming_init=self.kaiming_init)

        self.net_q6 = MLP(input_dims=self.input_dims[5],
                          hid_dims=self.hid_dims,
                          out_dims=self.out_dims,
                          kaiming_init=self.kaiming_init)

        self.thres = AdaptiveSoftThreshold(1)


    def embedding(self, queries1, queries2, queries3, queries4, queries5, queries6):
        emb1 = self.net_q1(queries1)
        emb2 = self.net_q2(queries2)
        emb3 = self.net_q3(queries3)
        emb4 = self.net_q4(queries4)
        emb5 = self.net_q5(queries5)
        emb6 = self.net_q6(queries6)
        return emb1, emb2, emb3, emb4, emb5, emb6

    def get_coeff(self, q_emb, k_emb):
        c = self.thres(q_emb.mm(k_emb.t()))
        return self.shrink * c


    def forward(self, queries, keys):
        q = self.embedding(queries)
        k = self.embedding(keys)
        out = self.get_coeff(q_emb=q, k_emb=k)
        return out




def get_sparse_rep(msesc, data1, data2, data3, data4, data5, data6,
                   batch_size=10, chunk_size=100, non_zeros=1000):
    N = data1.shape[0]
    non_zeros = min(N, non_zeros)
    C = torch.empty([batch_size, N])
    if (N % batch_size != 0):
        raise Exception("batch_size should be a factor of dataset size.")
    if (N % chunk_size != 0):
        raise Exception("chunk_size should be a factor of dataset size.")

    val = []
    indicies = []
    with torch.no_grad():
        msesc.eval()
        for i in range(N // batch_size):
            chunk1 = data1[i * batch_size:(i + 1) * batch_size].cuda()
            chunk2 = data2[i * batch_size:(i + 1) * batch_size].cuda()
            chunk3 = data3[i * batch_size:(i + 1) * batch_size].cuda()
            chunk4 = data4[i * batch_size:(i + 1) * batch_size].cuda()
            chunk5 = data5[i * batch_size:(i + 1) * batch_size].cuda()
            chunk6 = data6[i * batch_size:(i + 1) * batch_size].cuda()
            q1, q2, q3, q4, q5, q6 = msesc.embedding(chunk1, chunk2, chunk3, chunk4, chunk5, chunk6)
            for j in range(N // chunk_size):
                chunk_samples1 = data1[j * chunk_size: (j + 1) * chunk_size].cuda()
                chunk_samples2 = data2[j * chunk_size: (j + 1) * chunk_size].cuda()
                chunk_samples3 = data3[j * chunk_size: (j + 1) * chunk_size].cuda()
                chunk_samples4 = data4[j * chunk_size: (j + 1) * chunk_size].cuda()
                chunk_samples5 = data5[j * chunk_size: (j + 1) * chunk_size].cuda()
                chunk_samples6 = data6[j * chunk_size: (j + 1) * chunk_size].cuda()
                k1, k2, k3, k4, k5, k6 = msesc.embedding(chunk_samples1, chunk_samples2, chunk_samples3, chunk_samples4, chunk_samples5, chunk_samples6)
                temp1 = msesc.get_coeff(q1, k1)
                temp2 = msesc.get_coeff(q2, k2)
                temp3 = msesc.get_coeff(q3, k3)
                temp4 = msesc.get_coeff(q4, k4)
                temp5 = msesc.get_coeff(q5, k5)
                temp6 = msesc.get_coeff(q6, k6)

                C[:, j * chunk_size:(j + 1) * chunk_size] = (temp1.cpu() + temp2.cpu() + temp3.cpu() + temp4.cpu() + temp5.cpu() + temp6.cpu()) / 6.0

            rows = list(range(batch_size))
            cols = [j + i * batch_size for j in rows]
            C[rows, cols] = 0.0

            _, index = torch.topk(torch.abs(C), dim=1, k=non_zeros)

            val.append(C.gather(1, index).reshape([-1]).cpu().data.numpy())
            index = index.reshape([-1]).cpu().data.numpy()
            indicies.append(index)

    val = np.concatenate(val, axis=0)
    indicies = np.concatenate(indicies, axis=0)
    indptr = [non_zeros * i for i in range(N + 1)]

    C_sparse = sparse.csr_matrix((val, indicies, indptr), shape=[N, N])
    return C_sparse


def get_knn_Aff(C_sparse_normalized, k=3, mode='symmetric'):
    C_knn = kneighbors_graph(C_sparse_normalized, k, mode='connectivity', include_self=False, n_jobs=10)
    if mode == 'symmetric':
        Aff_knn = 0.5 * (C_knn + C_knn.T)
    elif mode == 'reciprocal':
        Aff_knn = C_knn.multiply(C_knn.T)
    else:
        raise Exception("Mode must be 'symmetric' or 'reciprocal'")
    return Aff_knn



def evaluate(msesc, data1, data2, data3, data4, data5, data6,
             labels, num_subspaces, spectral_dim, non_zeros=1000, n_neighbors=3,batch_size=10000,
             chunk_size=10000, affinity='nearest_neighbor', knn_mode='symmetric'):
    C_sparse = get_sparse_rep(msesc=msesc, data1=data1, data2=data2,data3=data3,
                              data4=data4, data5=data5, data6=data6,
                              batch_size=min(batch_size, 10000),
                              chunk_size= chunk_size, non_zeros=non_zeros)
    C_sparse_normalized = normalize(C_sparse).astype(np.float32)
    if affinity == 'symmetric':
        Aff = 0.5 * (np.abs(C_sparse_normalized) + np.abs(C_sparse_normalized).T)
    elif affinity == 'nearest_neighbor':
        Aff = get_knn_Aff(C_sparse_normalized, k=n_neighbors, mode=knn_mode)
    else:
        raise Exception("affinity should be 'symmetric' or 'nearest_neighbor'")
    preds = utils.spectral_clustering(Aff, num_subspaces, spectral_dim)
    acc = clustering_accuracy(labels, preds)
    nmi = normalized_mutual_info_score(labels, preds, average_method='geometric')
    ari = adjusted_rand_score(labels, preds)
    return acc, nmi, ari


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Caltech101_20")
    parser.add_argument('--num_subspaces', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=200.0)
    parser.add_argument('--lmbd', type=float, default=0.9)
    parser.add_argument('--hid_dims', type=int, default=[1024, 1024, 1024])
    parser.add_argument('--out_dims', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_min', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=2386)
    parser.add_argument('--chunk_size', type=int, default=2386)
    parser.add_argument('--non_zeros', type=int, default=1000)
    parser.add_argument('--n_neighbors', type=int, default=3)
    parser.add_argument('--spectral_dim', type=int, default=15)
    parser.add_argument('--affinity', type=str, default="nearest_neighbor")
    parser.add_argument('--mean_subtract',  action='store_true')
    parser.set_defaults(mean_subtract=True)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    dataset, dims, view, data_size, class_num = load_data(args.dataset)

    fit_msg = "Experiments on {}, numpy_seed={}, total_epoch={}".format(args.dataset, args.seed, args.epoch)
    print(fit_msg)

    same_seeds(args.seed)

    full_samples1 = dataset.data1
    full_samples2 = dataset.data2
    full_samples3 = dataset.data3
    full_samples4 = dataset.data4
    full_samples5 = dataset.data5
    full_samples6 = dataset.data6

    full_labels = dataset.y


    if args.mean_subtract:
        print("Mean Subtraction") #mean subtraction
        full_samples1 = full_samples1 - np.mean(full_samples1, axis=0, keepdims=True)
        full_samples2 = full_samples2 - np.mean(full_samples2, axis=0, keepdims=True)
        full_samples3 = full_samples3 - np.mean(full_samples3, axis=0, keepdims=True)
        full_samples4 = full_samples4 - np.mean(full_samples4, axis=0, keepdims=True)
        full_samples5 = full_samples5 - np.mean(full_samples5, axis=0, keepdims=True)
        full_samples6 = full_samples6 - np.mean(full_samples6, axis=0, keepdims=True)

    full_labels = full_labels - np.min(full_labels)  # 计算sre时需要label的范围是 0 ~ num_subspaces - 1


    samples1, samples2, samples3, samples4, samples5, samples6, labels = full_samples1, \
    full_samples2, full_samples3, full_samples4, full_samples5, full_samples6, full_labels

    data1 = torch.from_numpy(samples1).float()
    data1 = utils.p_normalize(data1)
    data2 = torch.from_numpy(samples2).float()
    data2 = utils.p_normalize(data2)
    data3 = torch.from_numpy(samples3).float ()
    data3 = utils.p_normalize(data3)
    data4 = torch.from_numpy(samples4).float()
    data4 = utils.p_normalize(data4)
    data5 = torch.from_numpy(samples5).float()
    data5 = utils.p_normalize(data5)
    data6 = torch.from_numpy(samples6).float()
    data6 = utils.p_normalize(data6)
    data1 = data1.cuda()
    data2 = data2.cuda()
    data3 = data3.cuda()
    data4 = data4.cuda()
    data5 = data5.cuda()
    data6 = data6.cuda()
    block_size = min(data_size, 10000)
    msesc = MSESC(dims, args.hid_dims, args.out_dims, data_size, kaiming_init=True).cuda()
    checkpoint = torch.load('./models/' + args.dataset + '.pth')
    msesc.load_state_dict(checkpoint)
    acc, nmi, ari = evaluate(msesc, data1=data1, data2=data2, data3=data3, data4=data4, data5=data5, data6=data6,
                                     labels=full_labels, num_subspaces=args.num_subspaces,
                                     affinity=args.affinity, spectral_dim=args.spectral_dim, non_zeros=args.non_zeros,
                                     n_neighbors=args.n_neighbors, batch_size=block_size, chunk_size=block_size,
                                     knn_mode='symmetric')
    print("ACC-{:.6f}, NMI-{:.6f}, ARI-{:.6f}".format(acc, nmi, ari))
