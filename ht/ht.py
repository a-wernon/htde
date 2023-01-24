import numpy as np
import torch
from functools import partial


class HTuckerNode(object):
    def __init__(self, content):
        # content is either a [left, right] or a leaf of torch.Tensor type
        # now no initialisation is done automatically,
        # so we just pass handcrafted version of content
        # content should contain only nested lists of 3-tensors
        if isinstance(content, HTuckerNode):
            if isinstance(content.content, torch.Tensor):
                self.content = content.content.detach().clone()
            else:
                self.content = content.content.copy()
            self.is_leaf = content.is_leaf
        else:
            self.content = content
            self.is_leaf = isinstance(self.content, torch.Tensor)
            if not self.is_leaf:
                for i in range(len(self.content)):
                    self.content[i] = HTuckerNode(self.content[i])

        self.verbose = 1

    def __repr__(self, buffer=""):
        if self.is_leaf:
            return buffer + str(self.content.shape) + "\n"
        else:
            buffer_new = buffer + "|"
            l1 = self.content[0].__repr__(buffer=buffer_new)
            l2 = self.content[1].__repr__(buffer=buffer_new)
            return f"{l1}{buffer_new}\n{l2}"

    def get_full(self):
        if self.is_leaf:
            return self.content
        else:
            assert len(self.content) == 2
            full_left = HTuckerNode(self.content[0]).get_full()
            full_right = HTuckerNode(self.content[1]).get_full()
            if self.verbose >= 2:
                print(full_left.shape, full_right.shape)
            return torch.tensordot(full_left, full_right, dims=[[-1], [0]])

    def scalar_product(self, right_node: "HTuckerNode"):
        # right Node is also of H Tucker format
        # returns 4-tensor
        if self.is_leaf:
            ans = torch.einsum("ijk, ljm -> ilkm", self.content, right_node.content)
            # ans = torch.tensordot(self.content, right_node.content, dims=[1, 1])
            if self.verbose >= 2:
                print(ans.shape, "ans.shape")
            assert len(ans.shape) == 4
            return ans
        else:
            assert len(self.content) == 2 and len(right_node.content) == 2
            l_result = self.content[0].scalar_product(right_node.content[0])
            r_result = self.content[1].scalar_product(right_node.content[1])
            if self.verbose >= 2:
                print(l_result.shape, r_result.shape, "result shapes")
            return torch.tensordot(l_result, r_result, dims=[[-2, -1], [0, 1]])
            # should check order, seems legit for now
            # order is like this [1_l, 2_l, 1_r, 2_r]

    def get_params_for_optim(self):
        if self.is_leaf:
            return [self.content]
        else:
            return (
                self.content[0].get_params_for_optim()
                + self.content[1].get_params_for_optim()
            )


class FuncList(list):
    """
    simple list wrapper for holding function objects,
    rewrite in a normal version
    """

    def __init__(self, content):
        super().__init__(content)
        self.shape = len(content)


class FunctionalHTuckerNode(HTuckerNode):
    def __init__(self, content):
        if isinstance(content, FunctionalHTuckerNode):
            self.content = content.content.detach().clone()
            self.is_leaf = content.is_leaf
        else:
            self.content = content
            self.is_leaf = isinstance(self.content, FuncList)
            if not self.is_leaf:
                for i in range(len(self.content)):
                    self.content[i] = FunctionalHTuckerNode(self.content[i])

        self.verbose = 1

    def get_dimension(self):
        """
        returns output tensor dimension
        """

        if self.is_leaf:
            return 1
        else:
            return sum([node.get_dimension() for node in self.content])

    def get_val(self, x):
        """
        return content for h_t_node with value
        equal to execution of functional tensor in x

        should be written in a recursive fashion
        """

        if x.shape[0] != self.get_dimension():
            print(x.shape[0], self.get_dimension())
            raise ValueError

        if self.is_leaf:
            return torch.Tensor([f(x) for f in self.content]).reshape(1, -1, 1)

        else:
            left_size = self.content[0].get_dimension()
            x_left = x[:left_size]
            x_right = x[left_size:]
            return [self.content[0].get_val(x_left), self.content[1].get_val(x_right)]


def nsin(n, x):
    return torch.sin(torch.Tensor(np.array([n * x])))


def ncos(n, x):
    return torch.cos(torch.Tensor(np.array([n * x])))


def fourier_basis(n):
    if n % 2 == 0:
        return partial(ncos, n // 2)
    else:
        return partial(nsin, (n + 1) // 2)
