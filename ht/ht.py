import numpy as np
import torch
from functools import partial


class HTuckerNode(object):
    def __init__(self, content):
        # content is either a [left, core, right] or a leaf of torch.Tensor type
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
                assert len(self.content) == 3
                # for i in range(len(self.content)):
                self.content[0] = HTuckerNode(self.content[0])
                self.content[-1] = HTuckerNode(self.content[-1])

        self.verbose = 1

    def __repr__(self, buffer=""):
        if self.is_leaf:
            return buffer + str(self.content.shape) + "\n"
        else:
            buffer_new = buffer + "|" + str(self.content[1].shape)
            l1 = self.content[0].__repr__(buffer=buffer_new)
            l2 = self.content[-1].__repr__(buffer=buffer_new)
            return f"{l1}{buffer_new}\n{l2}"

    def get_full(self, core_dimension_placement="left"):
        # core dimension placement is either left of right
        # for left tensor placement is right in vice versa
        # what is done:
        # (1, k, ..., k, r) * (r, r, r) * (r, k, ..., k, 1) = (1, k,..., k, r)
        # it was a right placement example
        if self.is_leaf:
            return self.content
        else:
            assert len(self.content) == 3
            full_left = HTuckerNode(self.content[0]).get_full(
                core_dimension_placement="right"
            )
            full_right = HTuckerNode(self.content[-1]).get_full(
                core_dimension_placement="left"
            )
            core = self.content[1]
            if self.verbose >= 2:
                print(full_left.shape, core.shape, full_right.shape)

            if core_dimension_placement == "left":
                # there should be no other dims of size 1
                l_res = torch.tensordot(full_left, core, dims=[[-1], [0]])
                # shape (1, ..., r, r)
                dims = list(range(len(l_res.shape)))
                dims[0], dims[-2] = dims[-2], dims[0]
                if self.verbose >= 2:
                    print(dims)

                l_res_prm = torch.permute(l_res, dims)
                # shape (r, ..., 1, r)

                l_res_prm = torch.squeeze(l_res_prm)
                # shape (r, ..., r)
                result = torch.tensordot(l_res_prm, full_right, dims=[[-1], [0]])

                # shape (r, ..., 1)
                return result

            if core_dimension_placement == "right":
                # there should be no other dims of size 1
                r_res = torch.tensordot(core, full_right, dims=[[-1], [0]])
                # shape (r, r, ..., 1)
                dims = list(range(len(r_res.shape)))
                dims[1], dims[-1] = dims[-1], dims[1]
                if self.verbose >= 2:
                    print(dims)

                r_res_prm = torch.permute(r_res, dims)
                # shape (r, 1, ..., r)
                r_res_prm = torch.squeeze(r_res_prm)
                # shape (r, ..., r)
                result = torch.tensordot(full_left, r_res_prm, dims=[[-1], [0]])

                # shape (1, ..., r)
                return result

    def scalar_product(
        self, right_node: "HTuckerNode", core_dimension_placement="left"
    ):
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
            assert len(self.content) == 3 and len(right_node.content) == 3
            l_result = self.content[0].scalar_product(
                right_node.content[0], core_dimension_placement="right"
            )
            # shape (1, 1, r_1, r_2)
            r_result = self.content[-1].scalar_product(
                right_node.content[-1], core_dimension_placement="left"
            )
            # shape (r_1, r_2, 1, 1)
            first_core = self.content[1]
            # shape (r_1, r_1, r_1)
            second_core = right_node.content[1]
            # shape (r_2, r_2, r_2)
            if self.verbose >= 2:
                print(
                    l_result.shape,
                    r_result.shape,
                    first_core.shape,
                    second_core.shape,
                    "result shapes",
                )

            if core_dimension_placement == "left":
                return torch.einsum(
                    "abij, ikl, jmn, lncd -> kmcd",
                    l_result,
                    first_core,
                    second_core,
                    r_result,
                )

            if core_dimension_placement == "right":
                return torch.einsum(
                    "abij, ikl, jmn, lncd -> abkm",
                    l_result,
                    first_core,
                    second_core,
                    r_result,
                )
            # should check order, seems legit for now
            # order is like this [1_l, 2_l, 1_r, 2_r]

    def get_params_for_optim(self):
        if self.is_leaf:
            return [self.content]
        else:
            return (
                self.content[0].get_params_for_optim()
                + [self.content[1]]
                + self.content[2].get_params_for_optim()
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
            # this is a tensor of rank one, so middle core is very simple
            return [
                self.content[0].get_val(x_left),
                torch.ones(1, 1, 1),
                self.content[1].get_val(x_right),
            ]


def nsin(n, x):
    return torch.sin(torch.Tensor(np.array([n * x])))


def ncos(n, x):
    return torch.cos(torch.Tensor(np.array([n * x])))


def fourier_basis(n):
    if n % 2 == 0:
        return partial(ncos, n // 2)
    else:
        return partial(nsin, (n + 1) // 2)
