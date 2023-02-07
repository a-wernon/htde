import numpy as np
import torch


class Node():
    @staticmethod
    def build_node(node, level=0, parent=None):
        if not isinstance(node, Node):
            node = Node(node)
        node.level = level
        node.parent = parent
        return node

    @staticmethod
    def random(n, r, level=0, num=0, parent=None, device='cpu'):
        d = len(n)              # Dimension
        q = int(np.log2(d))     # Number of levels

        # Now ranks for one level are the same. In the case of different ranks
        # for one level, we can introduce something like "r[level][num]"

        if 2**q != d:
            # TODO: maybe we should add fictive nodes for this case
            raise NotImplementedError('Dimension should be power of 2')

        if level < q:
            r_inner = r[level-1] if level > 0 else 1
            G = torch.rand(r[level], r_inner, r[level]).to(device)
            Y = Node.build_node(G, level, parent)
            Y.set_children(
                Node.random(n, r, level+1, 2*num, Y, device),
                Node.random(n, r, level+1, 2*num+1, Y, device))
            return Y

        G = torch.rand(r[-1], n[num]).to(device) / 10 # Leaf TODO: remove 10 ;)
        return Node.build_node(G, level, parent)

    def __init__(self, G, level=0, parent=None):
        self.G = G            # Core for node (2D or 3D torch.tensor)
        self.L = None         # Left child node
        self.R = None         # Right child node
        self.level = level    # Level of the node in the tree (root is zero)
        self.parent = parent  # Parent node
        self._convolv = None

    def __repr__(self, level=0, indent=4):
        text = ' ' * indent * level

        if self.is_root:
            text += 'ROOT'
        elif self.is_leaf:
            text += 'LEAF'
        else:
            text += 'NODE'

        s = tuple(self.G.shape)
        m = torch.mean(self.G).item()
        text += ' ' + str(s) + f' [mean: {m:-8.2e}]'

        if self.L is not None:
            text += '\n' + self.L.__repr__(level+1, indent)
        if self.R is not None:
            text += '\n' + self.R.__repr__(level+1, indent)

        return text

    @property
    def convolv(self, force=False):
        if not force and self._convolv is not None:
            return self._convolv

        if self.is_leaf:
            self._convolv = torch.einsum('jk->j', self.G)
        else:
            self._convolv = torch.einsum('ijk,i,k->j',
                self.G, self.L.convolv, self.R.convolv)

        return self._convolv

    @property
    def is_leaf(self):
        return self.L is None or self.R is None

    @property
    def is_root(self):
        return self.parent is None

    def full(self, res=None, num_up=0):
        need_eins = False
        if res is None:
            res = [torch.tensor([1], device=self.G.device), [0]]
            need_eins = True

        if self.is_root:
            res.extend([self.G, [1, 0, 2]])
            self.L.full(res, 1)
            self.R.full(res, 2)
        elif self.is_leaf:
            next_num = _find_next_free_num(res)
            res.extend([self.G, [num_up, next_num]])
        else:
            next_num = _find_next_free_num(res)
            res.extend([self.G, [next_num, num_up, next_num + 1]])
            self.L.full(res, next_num)
            self.R.full(res, next_num + 1)

        if need_eins:
            return torch.einsum(*res)

    def get(self, I, num=0):
        if self.is_leaf:
            return self.G[:, I[:, num]]

        y = torch.einsum('rsq,rk,qk->sk',
            self.G, self.L.get(I, 2*num), self.R.get(I, 2*num+1))

        return y[0] if self.is_root else y

    def sample(self, up_mat=None):
        if up_mat is None: # For root node
            up_mat = torch.ones(self.G.shape[1], device=self.G.device)

        if self.is_leaf:
            p = torch.einsum('rk,r->k', self.G, up_mat)
            p = torch.abs(p)
            p /= p.sum()

            idx = torch.multinomial(p, 1, replacement=True)
            return [idx.item()]

        A = torch.einsum('ijk,j,i,k->ik',
            self.G, up_mat, self.L.convolv, self.R.convolv)
        U, V = _matrix_skeleton(A, e=1.E-8)

        p = torch.einsum('ir,rj->r', U, V)
        p = torch.abs(p)
        p /= p.sum()

        idx = torch.multinomial(p, 1, replacement=True)
        idx1 = self.L.sample(U[:, idx].squeeze())
        idx2 = self.R.sample(V[idx, :].squeeze())
        return idx1 + idx2

        def scalar_product(self):
        if self.is_leaf:
            v = torch.einsum('jk->j', self.G)
        else:
            v = torch.einsum('ijk,i,k->j',
                self.G, self.L.scalar_product(), self.R.scalar_product())

        if self.is_root:
            v = v.item()

        return v

    def set_children(self, node_l, node_r):
        self.L = Node.build_node(node_l, self.level+1, self)
        self.R = Node.build_node(node_r, self.level+1, self)

    def to_core_list(self):
        res = [self.G]
        if self.L is not None:
            res.extend(self.L.to_core_list())
        if self.R is not None:
            res.extend(self.R.to_core_list())
        return res


def _find_next_free_num(arr):
    res = []
    for l in arr[1::2]:
        res.extend(l)
    return max(res) + 1


def _matrix_skeleton(A, e=1.E-10, r=1.E+12):
    U, s, V = torch.linalg.svd(A, full_matrices=False)
    s2 = torch.cumsum(torch.flip(s, dims=(0,))**2, 0)
    where = torch.where(s2 <= e**2)[0]
    dlen = 0 if len(where) == 0 else int(1 + where[-1])
    r = max(1, min(int(r), len(s) - dlen))
    S = torch.diag(torch.sqrt(s[:r]))
    return U[:, :r] @ S, S @ V[:r, :]
