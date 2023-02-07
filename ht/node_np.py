import numpy as np


# Select contract function for big contrunctions:
try:
    import opt_einsum
    contract = opt_einsum.contract
except Exception as e:
    contract = np.einsum


class Node():
    @staticmethod
    def build_node(node, level=0, parent=None):
        if not isinstance(node, Node):
            node = Node(node)
        node.level = level
        node.parent = parent
        return node

    @staticmethod
    def random(n, r, level=0, num=0, parent=None):
        d = len(n)           # Dimension
        q = int(np.log2(d))  # Number of levels

        # Now ranks for one level are the same. In the case of different ranks
        # for one level, we can introduce something like "r[level][num]"

        if 2**q != d:
            # TODO: maybe we should add fictive nodes for this case
            raise NotImplementedError('Dimension should be power of 2')

        if level < q:
            r_inner = r[level-1] if level > 0 else 1
            G = np.random.rand(r[level], r_inner, r[level])
            Y = Node.build_node(G, level, parent)
            Y.set_children(
                Node.random(n, r, level+1, 2*num, Y),
                Node.random(n, r, level+1, 2*num+1, Y))
            return Y

        G = np.random.rand(r[-1], n[num]) / 10 # Leaf TODO: remove 10 ;)
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

        text += ' ' + str(self.G.shape) + f' [mean: {np.mean(self.G):-8.2e}]'

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
            self._convolv = np.einsum('jk->j', self.G)
        else:
            self._convolv = np.einsum('ijk,i,k->j',
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
            res = [np.array([1]), [0]]
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
            return contract(*res)

    def get(self, I, num=0):
        if self.is_leaf:
            return self.G[:, I[:, num]]

        y = np.einsum('rsq,rk,qk->sk',
            self.G, self.L.get(I, 2*num), self.R.get(I, 2*num+1))

        return y[0] if self.is_root else y
        
    def sample(self, up_mat=None):
        if up_mat is None: # For root node
            up_mat = np.ones(self.G.shape[1])

        if self.is_leaf:
            p = contract('ij,i->j', self.G, up_mat)
            p = np.abs(p)
            p /= p.sum()

            idx = np.random.choice(len(p), p=p)

            return [idx]

        A = contract('ijk,j,i,k->ik',
            self.G, up_mat, self.L.convolv, self.R.convolv)
        U, V = _matrix_skeleton(A, e=1.E-8)

        p = contract('ir,rj->r', U, V)
        p = np.abs(p)
        p /= p.sum()

        idx = np.random.choice(len(p), p=p)
        idx1 = self.L.sample(U[:, idx])
        idx2 = self.R.sample(V[idx, :])

        return idx1 + idx2

    def set_children(self, node_l, node_r):
        self.L = Node.build_node(node_l, self.level+1, self)
        self.R = Node.build_node(node_r, self.level+1, self)


def _find_next_free_num(arr):
    res = []
    for l in arr[1::2]:
        res.extend(l)
    return max(res) + 1


def _matrix_skeleton(A, e=1.E-10, r=1.E+12):
    """Construct truncated skeleton decomposition A = U V for the given matrix.

    Function from teneva (https://github.com/AndreiChertkov/teneva) package.

    Args:
        A (np.ndarray): matrix of the shape [m, n].
        e (float): desired approximation accuracy (> 0).
        r (int, float): maximum rank for the SVD decomposition (> 0).

    Returns:
        [np.ndarray, np.ndarray]: factor matrix U of the shape [m, q] and factor
        matrix V of the shape [q, n], where "q" is selected rank (q <= r).

    """
    U, s, V = np.linalg.svd(A, full_matrices=False, hermitian=False)

    where = np.where(np.cumsum(s[::-1]**2) <= e**2)[0]
    dlen = 0 if len(where) == 0 else int(1 + where[-1])
    r = max(1, min(int(r), len(s) - dlen))

    S = np.diag(np.sqrt(s[:r]))
    return U[:, :r] @ S, S @ V[:r, :]
