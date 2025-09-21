import torch
import torch.nn.functional as F
from torch.linalg import svd, norm

#
# OMP (for K-SVD) 
def omp(D, y, sparsity, device='cpu'):
    """
    Orthogonal Matching Pursuit algorithm using PyTorch
    """
    residual = y.clone()
    idx = []
    x = torch.zeros(D.shape[1], device=device, dtype=D.dtype)
    x_ls = None

    for _ in range(sparsity):
        proj = D.T @ residual
        atom_index = torch.argmax(torch.abs(proj)).item()
        if atom_index in idx:
            break
        idx.append(atom_index)
        D_selected = D[:, idx]

        x_ls = torch.linalg.lstsq(D_selected, y.unsqueeze(1)).solution.squeeze(1)
        residual = y - D_selected @ x_ls

        if norm(residual) < 1e-8:
            break

    if x_ls is not None:
        x[idx] = x_ls
    return x

def ksvd(Y, n_atoms, sparsity, n_iter=10, device='cpu'):
    """
    K-SVD algorithm implementation using PyTorch
    """
    n_features, n_samples = Y.shape
    Y = Y.to(device)

    n_atoms = min(n_atoms, n_samples)

    # Initialize D: random sample columns
    cols = torch.randperm(n_samples, device=device)[:n_atoms]
    D = Y[:, cols].clone().float()

    D = D / (torch.linalg.norm(D, dim=0, keepdim=True) + 1e-12)

    X = torch.zeros((n_atoms, n_samples), device=device, dtype=D.dtype)

    for it in range(n_iter):
        # Sparse coding (OMP for each column)
        for i in range(n_samples):
            X[:, i] = omp(D, Y[:, i], sparsity, device)

        # Dictionary update
        for k in range(n_atoms):
            omega = torch.nonzero(X[k, :], as_tuple=True)[0]
            if len(omega) == 0:
                continue

            E = Y[:, omega] - D @ X[:, omega] + torch.outer(D[:, k], X[k, omega])

            U, s, Vt = svd(E, full_matrices=False)
            D[:, k] = U[:, 0]
            X[k, omega] = s[0] * Vt[0, :]
            del U
            del s
            del Vt
        D = D / (torch.linalg.norm(D, dim=0, keepdim=True) + 1e-12)

    return D, X

class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        if self.p[a] != a:
            self.p[a] = self.find(self.p[a])
        return self.p[a]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

def cluster_by_cosine_similarity(weights, threshold=0.9, device='cpu'):

    n = len(weights)

    vecs = []
    for w in weights:
        w = w.to(device)
        v = w.flatten().float()
        v = v / (torch.linalg.norm(v) + 1e-12)
        vecs.append(v)

    V = torch.stack(vecs, dim=0)  # (n, dim)
    sim = V @ V.T  # 코사인 유사도
  
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i+1, n):
            if sim[i, j] >= threshold:
                uf.union(i, j)

    clusters = {}
    for i in range(n):
        r = uf.find(i)
        clusters.setdefault(r, []).append(i)

    return list(clusters.values())

def cluster_and_shared_ksvd(weights,
                           threshold=0.9,
                           n_atoms=50,
                           sparsity=5,
                           n_iter=10,
                           device='cpu'):

    if isinstance(weights[0], torch.Tensor):
        weights = [w.to(device) for w in weights]
    else:
        weights = [torch.tensor(w, device=device) for w in weights]

    # 1) Group by shape
    shape_groups = {}
    for idx, W in enumerate(weights):
        shape_groups.setdefault(W.shape, []).append((idx, W))

    all_results = []
    for shape, items in shape_groups.items():
        idxs, mats = zip(*items)
        mats = list(mats) 

        if len(mats) == 1:
            clusters = [[0]]
        else:
            clusters = cluster_by_cosine_similarity(mats, threshold=threshold, device=device)
            print(f'유사도 기반 클러스터별 i번째 expert{clusters}')

        for cl in clusters:
            orig_indices = [idxs[i] for i in cl]   
            cluster_mats = [mats[i] for i in cl]
            out, inp = shape

      
            Y = torch.cat(cluster_mats, dim=1)

            n_atoms_use = min(n_atoms, Y.shape[1])
            if n_atoms_use < 1:
                n_atoms_use = 1

            D, X_cluster = ksvd(Y, n_atoms_use, sparsity, n_iter=n_iter, device=device)

            W_approxs = []
            slices = []
            start = 0
            for _ in cluster_mats:
                end = start + inp
                X_slice = X_cluster[:, start:end]  # n_atoms_use x inp
                W_approx = D @ X_slice            #out x inp
                W_approxs.append(W_approx)
                slices.append((start, end))
                start = end

            result = {
                'shape': shape,
                'indices': orig_indices,
                'D': D,
                'X_cluster': X_cluster,
                'slices': slices,
                Fbb'W_approxs': W_approxs
            }
            all_results.append(result)

    return all_results
