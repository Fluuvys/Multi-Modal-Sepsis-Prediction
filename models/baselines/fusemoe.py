import sys
import math
import copy
import pdb
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import BCELoss, CrossEntropyLoss

# Code adapted from the fairseq repo.
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AdamW,
    BertTokenizer,
    BertModel,
    get_scheduler,
    set_seed,
    BertPreTrainedModel
)

#-------------------#
# Config

class MoEConfig:
    def __init__(
        self, 
        num_experts, 
        moe_input_size,
        moe_hidden_size,
        moe_output_size,
        router_type,
        gating='softmax',
        num_modalities=1,
        vocab_size=100,
        num_tasks=2, 
        top_k=4,
        disjoint_top_k=2,
        noisy_gating=True,
        max_position_embeddings=512,
        type_vocab_size=2,
        modality_type_vocab_size=2,
        hidden_dim=768, 
        num_layers=8, 
        dropout=0.1, 
        hidden_dropout_prob=0.1, 
        pre_lnorm=True,
        n_heads=8,
        image_size=224,
        patch_size=16,
        num_channels=3,
        max_image_length=-1,
        layer_norm_eps=1e-5,
        expert_activation=nn.ReLU(), 
        task_activation=nn.ReLU(), 
        output_activation=nn.Sigmoid(),
        hidden_act="gelu",
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
        is_decoder=False,
    ):
        # Input
        self.vocab_size = vocab_size
        self.hidden_size = hidden_dim
        self.type_vocab_size = type_vocab_size
        self.modality_type_vocab_size = modality_type_vocab_size

        # MoE
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.top_k = top_k
        self.disjoint_top_k = disjoint_top_k
        self.noisy_gating = noisy_gating
        self.moe_input_size = moe_input_size
        self.moe_hidden_size = moe_hidden_size
        self.moe_output_size = moe_output_size
        self.router_type = router_type
        self.num_modalities = num_modalities
        self.gating = gating

        # image
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.max_image_length = max_image_length

        # Transformer
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.pre_lnorm = pre_lnorm
        self.n_heads = n_heads
        self.d_heads = int(hidden_dim / n_heads)

        # LayerNorm
        self.layer_norm_eps = layer_norm_eps

        # Activations
        self.expert_activation = expert_activation
        self.task_activation = task_activation
        self.output_activation = output_activation
        self.hidden_act = hidden_act

        # Other
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict
        self.is_decoder = is_decoder


# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py



class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(self, config:MoEConfig, input_size:int, output_size:int, hidden_size:int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = ACT2FN[config.hidden_act]
        self.log_soft = nn.LogSoftmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.log_soft(out)
        return out


class HierarchicalMoE(nn.Module):

    """Implementation of Hierarchcial Mixture-of-Experts (HME) with two levels.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, config: MoEConfig):
        super(HierarchicalMoE, self).__init__()
        self.noisy_gating = config.noisy_gating
        
        self.output_size = config.moe_output_size
        self.input_size = config.moe_input_size
        self.hidden_size = config.moe_hidden_size
        self.router_type = config.router_type
        self.num_modalities = config.num_modalities
        self.num_experts = config.num_experts
        self.k = config.top_k
        self.gating = config.gating
        # instantiate experts
        self.experts = nn.ModuleList(
            nn.ModuleList([MLP(config, self.input_size, self.output_size, self.hidden_size) for _ in range(self.num_experts[1])])
            for _ in range(self.num_experts[0])
        )
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k[0] <= self.num_experts[0])
        assert(self.k[1] <= self.num_experts[1])

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values, level):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k[level]
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def _get_logits(self, x, train, level, noise_epsilon):
        w_gate = nn.Parameter(torch.zeros(self.input_size, self.num_experts[level]), requires_grad=True).to(x.device)
        w_noise = nn.Parameter(torch.zeros(self.input_size, self.num_experts[level]), requires_grad=True).to(x.device)
        if self.gating[level] == 'softmax':
            clean_logits = x @ w_gate
        elif self.gating[level] == 'laplace':
            clean_logits = -torch.cdist(x, torch.t(w_gate))
        elif self.gating[level] == 'gaussian':
            clean_logits = -torch.pow(torch.cdist(x, torch.t(w_gate)), 2)

        if self.noisy_gating:
            raw_noise_stddev = x @ w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        return logits, clean_logits, noisy_logits, noise_stddev

    def _top_k_gating(self, logits, clean_logits, noisy_logits, noise_stddev, level):
        top_logits, top_indices = logits.topk(min(self.k[level] + 1, self.num_experts[level]), dim=1)
        top_k_logits = top_logits[:, :self.k[level]]
        top_k_indices = top_indices[:, :self.k[level]]
        if self.gating[level] == 'softmax':
            top_k_gates = self.softmax(top_k_logits)
        elif self.gating[level] == 'laplace' or self.gating[level] == 'gaussian':
            top_k_gates = torch.exp(top_k_logits - torch.logsumexp(top_k_logits, dim=1, keepdim=True))

        zeros = torch.zeros_like(logits, requires_grad=True)
        # map the sorted gate to their original positions
        # obtain gating weights with expert position information
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k[level] < self.num_experts[level]:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits, level)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def noisy_top_k_gating(self, x, train, level, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            level: int - 0 indicates outer, 1 indicates inner
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        all_logits = self._get_logits(x, train, level, noise_epsilon)
        logits, clean_logits, noisy_logits, noise_stddev = all_logits[0], all_logits[1], all_logits[2], all_logits[3]
        gates, load = self._top_k_gating(logits, clean_logits, noisy_logits, noise_stddev, level)
        
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        return gates, loss

    def forward(self, x, train=True, loss_coef=1e-2, modalities=None):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        if isinstance(x, list):
            x = torch.concat(x, dim=1)

        gates_outer, loss_outer = self.noisy_top_k_gating(x, train, 0)
        loss_outer *= loss_coef

        dispatcher_outer = SparseDispatcher(self.num_experts[0], gates_outer)
        expert_inputs_outer = dispatcher_outer.dispatch(x)

        all_inner_loss, all_expert_group_output = 0, []
        for j, exp_inp in enumerate(expert_inputs_outer):
            # TODO: importance weighting in this line
            # sequential input inside top_k gating and dispatcher
            gates_inner, loss_inner = self.noisy_top_k_gating(exp_inp, train, 1)
            all_inner_loss += loss_inner
            dispatcher_inner = SparseDispatcher(self.num_experts[1], gates_inner)
            expert_inputs_inner = dispatcher_inner.dispatch(exp_inp)
            # TODO: add refactoring code here if needed
            expert_outputs_inner = [self.experts[j][i](expert_inputs_inner[i]) for i in range(self.num_experts[1])]
            all_expert_group_output.append(dispatcher_inner.combine(expert_outputs_inner))

        all_inner_loss /= self.num_experts[0]
        all_inner_loss *= loss_coef
        y = dispatcher_outer.combine(all_expert_group_output)
        return y, loss_outer + all_inner_loss

# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py

from torch.distributions.normal import Normal
from transformers.activations import ACT2FN


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates, router_type):
        """Create a SparseDispatcher."""
        # each forward pass initialize the sparse dispatcher once
        self._gates = gates
        self._num_experts = num_experts
        self._router_type = router_type
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        # _batch_index: sample index inside a batch that is assigned to particular expert, concatenated
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        # from e.g., [64, 4] -> [256, 4], collection of samples assigned to each gate
        gates_exp = gates[self._batch_index.flatten()]
        # difference between torch.nonzero(gates) and self._nonzero_gates
        # the first one is index and the second one is actural weights!
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        if isinstance(inp, list) and self._router_type == 'joint':
            inp = torch.concat(inp, dim=1)
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        # concat all sample outputs from each expert
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        # this is the weighted combination step
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        ## combined[combined == 0] = np.finfo(float).eps # Fix
        combined = combined.clamp(min=1e-5) # Fix
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(self, config:MoEConfig, input_size:int, output_size:int, hidden_size:int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = ACT2FN[config.hidden_act]
        self.log_soft = nn.LogSoftmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.log_soft(out)
        return out


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, config: MoEConfig):
        super(MoE, self).__init__()
        self.noisy_gating = config.noisy_gating
        self.num_experts = config.num_experts
        self.output_size = config.moe_output_size
        self.input_size = config.moe_input_size
        self.hidden_size = config.moe_hidden_size
        self.k = config.top_k
        self.disjoint_k = config.disjoint_top_k
        self.router_type = config.router_type
        self.num_modalities = config.num_modalities
        self.gating = config.gating

        # instantiate experts
        if self.router_type == 'disjoint':
            self.w_gate = [nn.Parameter(torch.zeros(self.input_size//self.num_modalities, self.num_experts//self.num_modalities), requires_grad=True) for _ in range(self.num_modalities)]
            self.w_noise = [nn.Parameter(torch.zeros(self.input_size//self.num_modalities, self.num_experts//self.num_modalities), requires_grad=True) for _ in range(self.num_modalities)]
        elif self.router_type == 'permod':
            self.w_gate = [nn.Parameter(torch.zeros(self.input_size//self.num_modalities, self.num_experts), requires_grad=True) for _ in range(self.num_modalities)]
            self.w_noise = [nn.Parameter(torch.zeros(self.input_size//self.num_modalities, self.num_experts), requires_grad=True) for _ in range(self.num_modalities)]
        else:
            self.w_gate = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
            self.w_noise = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
        if self.router_type == 'disjoint':
            self.experts = nn.ModuleList(
                nn.ModuleList([MLP(config, self.input_size//self.num_modalities, self.output_size, self.hidden_size) for _ in range(self.num_experts//self.num_modalities)])
                for _ in range(self.num_modalities)
            )
        elif self.router_type == 'permod':
            self.experts = nn.ModuleList([MLP(config, self.input_size//self.num_modalities, self.output_size, self.hidden_size) for _ in range(self.num_experts)])
        else:
            self.experts = nn.ModuleList([MLP(config, self.input_size, self.output_size, self.hidden_size) for _ in range(self.num_experts)])
        
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        if self.router_type == 'disjoint':
            threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.disjoint_k
        else:
            threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)

        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def _get_logits(self, x, train, noise_epsilon, idx=None):
        if idx is not None:
            w_gate = self.w_gate[idx].to(x.device)
            w_noise = self.w_noise[idx].to(x.device)
        else:
            w_gate = self.w_gate
            w_noise = self.w_noise
        if self.gating == 'softmax':
            clean_logits = x @ w_gate
        elif self.gating == 'laplace':
            clean_logits = -torch.cdist(x, torch.t(w_gate))
        elif self.gating == 'gaussian':
            clean_logits = -torch.pow(torch.cdist(x, torch.t(w_gate)), 2)

        if self.noisy_gating:
            raw_noise_stddev = x @ w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        return logits, clean_logits, noisy_logits, noise_stddev

    def _top_k_gating(self, logits, clean_logits, noisy_logits, noise_stddev, k):
        top_logits, top_indices = logits.topk(min(k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :k]
        top_k_indices = top_indices[:, :k]
        if self.gating == 'softmax':
            top_k_gates = self.softmax(top_k_logits)
        elif self.gating == 'laplace' or self.gating == 'gaussian':
            top_k_gates = torch.exp(top_k_logits - torch.logsumexp(top_k_logits, dim=1, keepdim=True))

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and k < self.num_experts:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2, modalities=None):
        """Multimodal noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        if self.router_type == 'joint':
            if isinstance(x, list):
                embeddings = torch.concat(x, dim=1)
            else:
                embeddings = x
            all_logits = self._get_logits(embeddings, train, noise_epsilon)
            logits, clean_logits, noisy_logits, noise_stddev = all_logits[0], all_logits[1], all_logits[2], all_logits[3]
            gates, load = self._top_k_gating(logits, clean_logits, noisy_logits, noise_stddev, self.k)
            return gates, load
        else:
            all_gates, all_loads = [], []
            for i in range(self.num_modalities):
                all_logits = self._get_logits(x[i], train, noise_epsilon, idx=i)
                logits, clean_logits, noisy_logits, noise_stddev = all_logits[0], all_logits[1], all_logits[2], all_logits[3]
                if self.router_type == 'permod':
                    gates, load = self._top_k_gating(logits, clean_logits, noisy_logits, noise_stddev, self.k)
                else:
                    gates, load = self._top_k_gating(logits, clean_logits, noisy_logits, noise_stddev, self.disjoint_k)
                all_gates.append(gates)
                all_loads.append(load)
            return all_gates, all_loads

    def forward(self, x, train=True, loss_coef=1e-2, modalities=None):
        """Args:
        x: tensor shape [batch_size, input_size]
        gating: type of gating function
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, train, modalities=modalities)
        # calculate importance loss
        if isinstance(gates, list):
            loss, y, sub_experts = 0, 0, self.num_experts//self.num_modalities
            for g, l in zip(gates, load):
                loss += self.cv_squared(g.sum(0)) + self.cv_squared(l)
            loss *= loss_coef
            for j, g in enumerate(gates):
                dispatcher = SparseDispatcher(self.num_experts, g, self.router_type)
                expert_inputs = dispatcher.dispatch(x[j])
                if self.router_type == 'permod':
                    expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
                elif self.router_type == 'disjoint':
                    expert_outputs = [self.experts[j][i](expert_inputs[i]) for i in range(sub_experts)]
                y += dispatcher.combine(expert_outputs)
        else:
            loss = self.cv_squared(gates.sum(0)) + self.cv_squared(load)
            loss *= loss_coef
            dispatcher = SparseDispatcher(self.num_experts, gates, self.router_type)
            expert_inputs = dispatcher.dispatch(x)
            # gates = dispatcher.expert_to_gates() # how is this line be used?
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
            y = dispatcher.combine(expert_outputs)
        return y, loss


#-------------------#
# Config

class MoEConfig:
    def __init__(
        self, 
        num_experts, 
        moe_input_size,
        moe_hidden_size,
        moe_output_size,
        router_type,
        gating='softmax',
        num_modalities=1,
        vocab_size=100,
        num_tasks=2, 
        top_k=4,
        disjoint_top_k=2,
        noisy_gating=True,
        max_position_embeddings=512,
        type_vocab_size=2,
        modality_type_vocab_size=2,
        hidden_dim=768, 
        num_layers=8, 
        dropout=0.1, 
        hidden_dropout_prob=0.1, 
        pre_lnorm=True,
        n_heads=8,
        image_size=224,
        patch_size=16,
        num_channels=3,
        max_image_length=-1,
        layer_norm_eps=1e-5,
        expert_activation=nn.ReLU(), 
        task_activation=nn.ReLU(), 
        output_activation=nn.Sigmoid(),
        hidden_act="gelu",
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
        is_decoder=False,
    ):
        # Input
        self.vocab_size = vocab_size
        self.hidden_size = hidden_dim
        self.type_vocab_size = type_vocab_size
        self.modality_type_vocab_size = modality_type_vocab_size

        # MoE
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.top_k = top_k
        self.disjoint_top_k = disjoint_top_k
        self.noisy_gating = noisy_gating
        self.moe_input_size = moe_input_size
        self.moe_hidden_size = moe_hidden_size
        self.moe_output_size = moe_output_size
        self.router_type = router_type
        self.num_modalities = num_modalities
        self.gating = gating

        # image
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.max_image_length = max_image_length

        # Transformer
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.pre_lnorm = pre_lnorm
        self.n_heads = n_heads
        self.d_heads = int(hidden_dim / n_heads)

        # LayerNorm
        self.layer_norm_eps = layer_norm_eps

        # Activations
        self.expert_activation = expert_activation
        self.task_activation = task_activation
        self.output_activation = output_activation
        self.hidden_act = hidden_act

        # Other
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict
        self.is_decoder = is_decoder
    

# ==========================================
# 1. HELPER FUNCTIONS & BASIC LAYERS
# ==========================================

def hold_out(mask, perc=0.2):
    """To implement the autoencoder component of the loss, we introduce a set
    of masking variables mr (and mr1) for each data point. If drop_mask = 0,
    then we removecthe data point as an input to the interpolation network,
    and includecthe predicted value at this time point when assessing
    the autoencoder loss. In practice, we randomly select 20% of the
    observed data points to hold out from
    every input time series."""
    mask = mask.cpu().detach().numpy()
    drop_mask = np.ones_like(mask)
    drop_mask *= mask
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            count = np.sum(mask[i, j], dtype='int')
            if int(0.20 * count) > 1:
                index = 0
                r = np.ones((count, 1))
                b = np.random.choice(count, int(0.20 * count), replace=False)
                r[b] = 0
                for k in range(mask.shape[2]):
                    if mask[i, j, k] > 0:
                        drop_mask[i, j, k] = r[index]
                        index += 1
    return drop_mask

def recon_loss(x_ts, m1, m2, ypred, num_features):
    """ Autoencoder loss """
    y = x_ts.transpose(1, 2)
    m1 = m1.transpose(1, 2)
    m2 = m2.transpose(1, 2)
    m2 = 1 - m2
    m = m1 * m2
    ypred = ypred[:, :num_features, :]
    x = (y - ypred) * (y - ypred)
    x = x * m
    count = torch.sum(m, dim=2)
    count = torch.where(count > 0, count, torch.ones_like(count))
    x = torch.sum(x, dim=2) / count
    x = torch.sum(x, dim=1) / num_features
    return torch.mean(x)

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)

def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1 + abs(dim2 - dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


# ==========================================
# 2. INTERPOLATION MODULES
# ==========================================

class S_Interp(nn.Module):
    def __init__(self, args, device, orig_d_ts):
        super(S_Interp, self).__init__()
        self.tt_max = args.tt_max
        self.device = device
        self.ref_t = torch.linspace(0, 1., self.tt_max).to(self.device)
        self.d_dim = orig_d_ts
        self.output = nn.Linear(args.embed_dim, args.embed_dim)
        self.kernel = Parameter(torch.zeros(self.d_dim))

    def forward(self, x_ts, x_ts_mask, ts_tt_list, rec_mask, reconstruction=False):
        x_ts = x_ts.transpose(1, 2)
        x_ts_mask = x_ts_mask.transpose(1, 2)
        tt_len = ts_tt_list.shape[-1]
        d = ts_tt_list.unsqueeze(1)
        d = d.repeat(1, self.d_dim, 1)
        if reconstruction:
            output_dim = tt_len
            m = rec_mask.transpose(1, 2)
            ref_t = d.unsqueeze(-2).repeat(1, 1, output_dim, 1)
        else:
            m = x_ts_mask
            ref_t = self.ref_t.unsqueeze(0)
            output_dim = self.tt_max

        d = d.unsqueeze(-1).repeat(1, 1, 1, output_dim)
        mask = m.unsqueeze(-1).repeat(1, 1, 1, output_dim)
        x_ts = x_ts.unsqueeze(-1).repeat(1, 1, 1, output_dim)

        norm = (d - ref_t) * (d - ref_t)
        a = torch.ones([self.d_dim, tt_len, output_dim]).to(self.device)

        pos_kernel = torch.log(1 + torch.exp(self.kernel))
        alpha = a * pos_kernel.unsqueeze(-1).unsqueeze(-1)
        w = torch.logsumexp(-alpha * norm + torch.log(mask + 1e-12), dim=2)
        w1 = w.unsqueeze(2).repeat(1, 1, tt_len, 1)
        w1 = torch.exp(-alpha * norm + torch.log(mask + 1e-12) - w1)
        y = torch.sum(w1 * x_ts, dim=2)

        w_t = torch.logsumexp(-10.0 * alpha * norm + torch.log(mask + 1e-12), dim=2)  # kappa = 10
        w_t = w.unsqueeze(2).repeat(1, 1, tt_len, 1)
        w_t = torch.exp(-10.0 * alpha * norm + torch.log(mask + 1e-12) - w_t)
        y_trans = torch.sum(w_t * x_ts, dim=2)
        rep1 = torch.cat([y, w, y_trans], dim=1)

        return rep1

class Cross_Interp(nn.Module):
    def __init__(self, args, device, orig_d_ts):
        super(Cross_Interp, self).__init__()
        self.device = device
        self.d_dim = orig_d_ts
        self.activation = nn.Sigmoid()
        self.cross_channel_interp = torch.empty(self.d_dim, self.d_dim).to(self.device)
        nn.init.eye_(self.cross_channel_interp)

    def forward(self, x, reconstruction=False):
        self.output_dim = x.shape[-1]
        cross_channel_interp = self.cross_channel_interp
        y = x[:, :self.d_dim, :]
        w = x[:, self.d_dim:2 * self.d_dim, :]  # x
        intensity = torch.exp(w)
        y = y.permute(0, 2, 1)
        w = w.permute(0, 2, 1)
        w2 = w
        w = w.unsqueeze(-1).repeat(1, 1, 1, self.d_dim)
        den = torch.logsumexp(w, dim=2)
        w = torch.exp(w2 - den)
        mean = torch.mean(y, dim=1)
        mean = mean.unsqueeze(1).repeat(1, self.output_dim, 1)
        w2 = torch.matmul(w * (y - mean), cross_channel_interp) + mean
        rep1 = w2.permute(0, 2, 1)
        if reconstruction is False:
            y_trans = x[:, 2 * self.d_dim:3 * self.d_dim, :]
            y_trans = y_trans - rep1  # subtracting smooth from transient part
            rep1 = torch.cat([rep1, intensity, y_trans], 1)
        return rep1


# ==========================================
# 3. BASE NEURAL NETWORK MODULES
# ==========================================

class Outer(nn.Module):
    def __init__(self, inp1_size: int = 128, inp2_size: int = 128, n_neurons: int = 128):
        super(Outer, self).__init__()
        self.inp1_size = inp1_size
        self.inp2_size = inp2_size
        self.feedforward = nn.Sequential(
            nn.Linear((inp1_size + 1) * (inp2_size + 1), n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(),
        )

    def forward(self, inp1, inp2):
        batch_size = inp1.size(0)
        append = torch.ones((batch_size, 1)).type_as(inp1)
        inp1 = torch.cat([inp1, append], dim=-1)
        inp2 = torch.cat([inp2, append], dim=-1)
        fusion = torch.zeros((batch_size, self.inp1_size + 1, self.inp2_size + 1)).type_as(inp1)
        for i in range(batch_size):
            fusion[i] = torch.outer(inp1[i], inp2[i])
        fusion = fusion.flatten(1)
        return self.feedforward(fusion)

class MAGGate(nn.Module):
    def __init__(self, inp1_size, inp2_size, dropout):
        super(MAGGate, self).__init__()
        self.fc1 = nn.Linear(inp1_size + inp2_size, 1)
        self.fc3 = nn.Linear(inp2_size, inp1_size)
        self.beta = nn.Parameter(torch.randn((1,)))
        self.norm = nn.LayerNorm(inp1_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp1, inp2):
        w2 = torch.sigmoid(self.fc1(torch.cat([inp1, inp2], -1)))
        adjust = self.fc3(w2 * inp2)
        one = torch.tensor(1).type_as(adjust)
        alpha = torch.min(torch.norm(inp1) / torch.norm(adjust) * self.beta, one)
        output = inp1 + alpha * adjust
        output = self.dropout(self.norm(output))
        return output

class gateMLP(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Sigmoid()
        )
        self._initialize()

    def _initialize(self):
        for model in [self.gate]:
            for layer in model:
                if type(layer) in [nn.Linear]:
                    torch.nn.init.xavier_normal_(layer.weight)

    def forward(self, hidden_states):
        gate_logits = self.gate(hidden_states)
        return gate_logits

class TimeSeriesCnnModel(nn.Module):
    def __init__(self, input_size, n_filters, filter_size, dropout, length, n_neurons, layers):
        super().__init__()
        padding = int(np.floor(filter_size / 2))
        self.layers = layers
        if layers >= 1:
            self.conv1 = nn.Conv1d(input_size, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)
        if layers >= 2:
            self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool2 = nn.MaxPool1d(2, 2)
        if layers >= 3:
            self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool3 = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(int(length * n_filters / (2 ** layers)), n_neurons)
        self.fc1_drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.layers >= 1:
            x = self.pool1(F.relu(self.conv1(x)))
        if self.layers >= 2:
            x = self.pool2(F.relu(self.conv2(x)))
        if self.layers >= 3:
            x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_drop(self.fc1(x)))
        return x

class multiTimeAttention(nn.Module):
    "mTAND module"
    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim * num_heads, nhidden)])

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -10000)
        p_attn = F.softmax(scores, dim=-2)
        p_attn = torch.nan_to_num(p_attn, nan=0.0) # Fix
        if dropout is not None:
            p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=0.1):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x)


# ==========================================
# 4. TRANSFORMER & ATTENTION COMPONENTS
# ==========================================

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """
    def __init__(self, embed_dim, num_heads, attn_dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if qkv_same:
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, args, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False, num_modalities=2):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.pre_self_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])

        self.self_attns = nn.ModuleList([MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        ) for _ in range(num_modalities)])

        self.post_self_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])
        self.pre_encoder_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])

        self.cross_attn_1 = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )

        self.cross_attn_2 = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )

        self.post_encoder_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])

        self.attn_mask = attn_mask
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.pre_ffn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])
        self.fc1 = nn.ModuleList([nn.Linear(self.embed_dim, 4 * self.embed_dim) for _ in range(num_modalities)])
        self.fc2 = nn.ModuleList([nn.Linear(4 * self.embed_dim, self.embed_dim) for _ in range(num_modalities)])
        self.pre_ffn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])
        
        if args.cross_method == 'moe':
            moe_config = MoEConfig(
                num_experts=args.num_of_experts[0],
                moe_input_size=args.tt_max * args.embed_dim * num_modalities,
                moe_hidden_size=args.hidden_size,
                moe_output_size=args.tt_max * args.embed_dim * num_modalities,
                top_k=args.top_k[0],
                router_type=args.router_type,
                num_modalities=args.num_modalities,
                gating=args.gating_function[0]
            )
            self.moe = MoE(moe_config)
            device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.moe = self.moe.to(device_str)
        elif args.cross_method == 'hme':
            moe_config = MoEConfig(
                num_experts=args.num_of_experts,
                moe_input_size=args.tt_max * args.embed_dim * num_modalities,
                moe_hidden_size=args.hidden_size,
                moe_output_size=args.tt_max * args.embed_dim * num_modalities,
                top_k=args.top_k,
                router_type=args.router_type,
                num_modalities=args.num_modalities,
                gating=args.gating_function
            )
            self.moe = HierarchicalMoE(moe_config)
            device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.moe = self.moe.to(device_str)
        
    def forward(self, x_list, modality):
        residual = x_list
        seq_len, bs = x_list[0].shape[0], x_list[0].shape[1]
        balance_loss = None

        x_list = [l(x) for l, x in zip(self.pre_self_attn_layer_norm, x_list)]

        output = [l(query=x, key=x, value=x) for l, x in zip(self.self_attns, x_list)]
        x_list = [x for x, _ in output]
        x_list = [F.dropout(x, p=self.res_dropout, training=self.training) for x in x_list]
        x_list = [r + x for r, x in zip(residual, x_list)]

        # moe or cross attn
        residual = x_list
        x_list = [l(x) for l, x in zip(self.pre_encoder_attn_layer_norm, x_list)]
        x_list = [torch.nan_to_num(x, nan=0.0) for x in x_list] # Fix
        if self.args.cross_method in ["moe", "hme"]:
            x_mod_in = [torch.reshape(x, (bs, -1)) for x in x_list]
            embd_len_list = [0] + list(np.cumsum([x.shape[1] for x in x_mod_in]))
            embeddings = torch.concat(x_mod_in, dim=1)
            if torch.isnan(embeddings).any():
                return None, None
            moe_out, balance_loss = self.moe(x_mod_in, modalities=modality)
            x_mod_out = [moe_out[:, embd_len_list[i]:embd_len_list[i + 1]] for i in range(len(embd_len_list) - 1)]
            x_allmod_output = [torch.reshape(x, (seq_len, bs, -1)) for x in x_mod_out]
            moe_output = [F.dropout(x, p=self.res_dropout, training=self.training) for x in x_allmod_output]
            x_list = [r + x for r, x in zip(residual, moe_output)]

        if self.args.cross_method == "self_cross":
            assert self.num_modalities == 2, 'Input modality should be 2 if using cross attention method.'
            x_txt, x_ts = x_list
            x_ts_to_txt, _ = self.cross_attn_1(query=x_txt, key=x_ts, value=x_ts)
            x_txt_to_ts, _ = self.cross_attn_2(query=x_ts, key=x_txt, value=x_txt)

            x_ts_to_txt = F.dropout(x_ts_to_txt, p=self.res_dropout, training=self.training)
            x_txt_to_ts = F.dropout(x_txt_to_ts, p=self.res_dropout, training=self.training)
            x_list = [r + x for r, x in zip(residual, (x_ts_to_txt, x_txt_to_ts))]

        # FNN
        residual = x_list
        x_list = [l(x) for l, x in zip(self.pre_ffn_layer_norm, x_list)]
        x_list = [F.relu(l(x)) for l, x in zip(self.fc1, x_list)]
        x_list = [F.dropout(x, p=self.relu_dropout, training=self.training) for x in x_list]
        x_list = [l(x) for l, x in zip(self.fc2, x_list)]
        x_list = [F.dropout(x, p=self.res_dropout, training=self.training) for x in x_list]
        x_list = [r + x for r, x in zip(residual, x_list)]
        return x_list, balance_loss


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, device, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False, learn_embed=True, q_seq_len=None, kv_seq_len=None):
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.device = device
        self.q_seq_len = q_seq_len
        self.kv_seq_len = kv_seq_len
        if learn_embed:
            if self.q_seq_len != None:
                self.embed_positions_q = nn.Embedding(self.q_seq_len, embed_dim, padding_idx=0)
                nn.init.normal_(self.embed_positions_q.weight, std=0.02)

            if self.kv_seq_len != None:
                self.embed_positions_kv = nn.Embedding(self.kv_seq_len, embed_dim)
                nn.init.normal_(self.embed_positions_kv.weight, std=0.02)
        else:
            # Assuming SinusoidalPositionalEmbedding is available from fairseq/transformers if not defined
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)

        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        x = x_in
        length_x = x.size(0)
        x = self.embed_scale * x_in
        if self.q_seq_len is not None:
            position_x = torch.arange(length_x, dtype=torch.long).to(self.device)
            x += (self.embed_positions_q(position_x).unsqueeze(0)).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            length_kv = x_in_k.size(0)
            position_kv = torch.arange(length_kv, dtype=torch.long).to(self.device)

            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.kv_seq_len is not None:
                x_k += (self.embed_positions_kv(position_kv).unsqueeze(0)).transpose(0, 1)
                x_v += (self.embed_positions_kv(position_kv).unsqueeze(0)).transpose(0, 1)
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)
        return x

    def max_positions(self):
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TransformerCrossEncoder(nn.Module):
    def __init__(self, args, embed_dim, num_heads, layers, device, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False, q_seq_len_1=None, q_seq_len_2=None, num_modalities=2):
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.device = device
        self.q_seq_len_1 = q_seq_len_1
        self.q_seq_len_2 = q_seq_len_2
        self.num_modalities = num_modalities
        self.embed_positions_q_1 = nn.Embedding(self.q_seq_len_1, embed_dim, padding_idx=0)
        nn.init.normal_(self.embed_positions_q_1.weight, std=0.02)

        if self.q_seq_len_2 != None:
            self.embed_positions_q_2 = nn.Embedding(self.q_seq_len_2, embed_dim, padding_idx=0)
            nn.init.normal_(self.embed_positions_q_2.weight, std=0.02)
            self.embed_positions_q = nn.ModuleList([self.embed_positions_q_1, self.embed_positions_q_2])
        else:
            self.embed_positions_q = nn.ModuleList([self.embed_positions_q_1 for _ in range(num_modalities)])

        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerCrossEncoderLayer(args,
                                                     embed_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout,
                                                     relu_dropout=relu_dropout,
                                                     res_dropout=res_dropout,
                                                     attn_mask=attn_mask,
                                                     num_modalities=num_modalities)
            self.layers.append(new_layer)

        self.normalize = True
        if self.normalize:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_modalities)])

    def forward(self, x_in_list, modality):
        x_list = x_in_list
        lengths, positions = [], []
        total_balance_loss = None

        for i in range(self.num_modalities):
            lengths.append(x_list[i].size(0))
        x_list = [self.embed_scale * x_in for x_in in x_in_list]
        if self.q_seq_len_1 is not None:
            for length in lengths:
                positions.append(torch.arange(length, dtype=torch.long).to(self.device))
            x_list = [l(position_x).unsqueeze(0).transpose(0, 1) + x for l, x, position_x in zip(self.embed_positions_q, x_list, positions)]
            x_list = [F.dropout(x, p=self.dropout, training=self.training) for x in x_list]

        for layer in self.layers:
            x_list, balance_loss = layer(x_list, modality)
            if x_list is None:
                return None, None
            if balance_loss is not None:
                if total_balance_loss is None:
                    total_balance_loss = balance_loss
                else:
                    total_balance_loss = total_balance_loss + balance_loss

        if self.normalize:
            x_list = [l(x) for l, x in zip(self.layer_norm, x_list)]
        return x_list, total_balance_loss


# ==========================================
# 5. MAIN MODELS
# ==========================================

class BertForRepresentation(nn.Module):
    """
    This class represents a BERT model for text representation.
    """
    def __init__(self, args, BioBert):
        super().__init__()
        self.bert = BioBert
        self.dropout = torch.nn.Dropout(BioBert.config.hidden_dropout_prob)
        self.model_name = args.model_name

    def forward(self, input_ids_sequence, attention_mask_sequence, sent_idx_list=None, doc_idx_list=None):
        txt_arr = []
        for input_ids, attention_mask in zip(input_ids_sequence, attention_mask_sequence):
            if 'Longformer' in self.model_name:
                attention_mask -= 1
                text_embeddings = self.bert(input_ids, global_attention_mask=attention_mask)
            else:
                text_embeddings = self.bert(input_ids, attention_mask=attention_mask)
            text_embeddings = text_embeddings[0][:, 0, :]
            text_embeddings = self.dropout(text_embeddings)
            txt_arr.append(text_embeddings)

        txt_arr = torch.stack(txt_arr)
        return txt_arr

class TextModel(nn.Module):
    def __init__(self, args, device, orig_d_txt=768, Biobert=None):
        super(TextModel, self).__init__()
        self.device = device
        self.task = args.task
        self.out_dropout = args.dropout
        self.orig_d_txt = orig_d_txt
        self.d_txt = args.embed_dim
        self.bertrep = BertForRepresentation(args, Biobert)
        self.proj_txt = nn.Linear(self.orig_d_txt, self.d_txt)

        output_dim = args.num_labels
        self.proj1 = nn.Linear(self.d_txt, self.d_txt)
        self.proj2 = nn.Linear(self.d_txt, self.d_txt)
        self.out_layer = nn.Linear(self.d_txt, output_dim)

        if 'ihm' in self.task:
            self.loss_fct1 = CrossEntropyLoss()
        elif 'pheno' in self.task:
            self.loss_fct1 = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def forward(self, input_ids_sequences, attn_mask_sequences, labels=None):
        x_txt = self.bertrep(input_ids_sequences, attn_mask_sequences)
        x_txt = torch.mean(x_txt, dim=1)
        proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(x_txt)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(proj_x_txt)), p=self.out_dropout, training=self.training))
        last_hs_proj += proj_x_txt
        output = self.out_layer(last_hs_proj)

        if 'ihm' in self.task:
            if labels != None:
                return self.loss_fct1(output, labels)
            return torch.nn.functional.softmax(output, dim=-1)[:, 1]
        elif 'pheno' in self.task:
            if labels != None:
                labels = labels.float()
                return self.loss_fct1(output, labels)
            return torch.nn.functional.sigmoid(output)

class MULTCrossModel(nn.Module):
    def __init__(self, args, device, modeltype=None, orig_d_ts=None, orig_reg_d_ts=None, orig_d_txt=None, ts_seq_num=None, text_seq_num=None, Biobert=None):
        super(MULTCrossModel, self).__init__()
        if modeltype != None:
            self.modeltype = modeltype
        else:
            self.modeltype = args.modeltype
        self.num_heads = args.num_heads
        self.args = args
        self.layers = args.layers
        self.device = device
        self.kernel_size = args.kernel_size
        self.dropout = args.dropout
        self.attn_mask = False
        self.irregular_learn_emb_ts = args.irregular_learn_emb_ts
        self.irregular_learn_emb_text = args.irregular_learn_emb_text
        self.irregular_learn_emb_cxr = args.irregular_learn_emb_cxr
        self.irregular_learn_emb_ecg = args.irregular_learn_emb_ecg
        self.reg_ts = args.reg_ts
        self.TS_mixup = args.TS_mixup
        self.mixup_level = args.mixup_level
        self.task = args.task
        self.tt_max = args.tt_max
        self.cross_method = args.cross_method
        self.num_modalities = args.num_modalities
        self.use_pt_text_embeddings = args.use_pt_text_embeddings
        self.token_type_embeddings = nn.Embedding(args.num_modalities, args.embed_dim)

        if self.irregular_learn_emb_ts or self.irregular_learn_emb_text:
            self.time_query = torch.linspace(0, 1., self.tt_max)
            self.periodic = nn.Linear(1, args.embed_time - 1)
            self.linear = nn.Linear(1, 1)

        if "TS" in self.modeltype:
            self.orig_d_ts = orig_d_ts
            self.d_ts = args.embed_dim
            self.ts_seq_num = ts_seq_num

            if self.irregular_learn_emb_ts:
                self.time_attn_ts = multiTimeAttention(self.orig_d_ts * 2, self.d_ts, args.embed_time, 8)

            if self.reg_ts:
                self.orig_reg_d_ts = orig_reg_d_ts
                self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size - 1) / 2), bias=False)

            if self.TS_mixup:
                if self.mixup_level == 'batch':
                    self.moe = gateMLP(input_dim=self.d_ts * 2, hidden_size=args.embed_dim, output_dim=1, dropout=args.dropout)
                elif self.mixup_level == 'batch_seq':
                    self.moe = gateMLP(input_dim=self.d_ts * 2, hidden_size=args.embed_dim, output_dim=1, dropout=args.dropout)
                elif self.mixup_level == 'batch_seq_feature':
                    self.moe = gateMLP(input_dim=self.d_ts * 2, hidden_size=args.embed_dim, output_dim=self.d_ts, dropout=args.dropout)
                else:
                    raise ValueError("Unknown mixedup type")

        if "Text" in self.modeltype:
            self.orig_d_txt = orig_d_txt
            self.d_txt = args.embed_dim
            self.text_seq_num = text_seq_num
            self.bertrep = BertForRepresentation(args, Biobert)

            if self.irregular_learn_emb_text:
                self.time_attn_text = multiTimeAttention(768, self.d_txt, args.embed_time, 8)
            else:
                self.proj_txt = nn.Conv1d(self.orig_d_txt, self.d_txt, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size - 1) / 2), bias=False)

        if "CXR" in self.modeltype:
            self.orig_d_cxr = 1024
            self.d_cxr = args.embed_dim
            self.cxr_seq_num = 5

            if self.irregular_learn_emb_cxr:
                self.time_attn_cxr = multiTimeAttention(1024, self.d_cxr, args.embed_time, 8)
            else:
                self.proj_cxr = nn.Conv1d(self.orig_d_cxr, self.d_cxr, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size - 1) / 2), bias=False)

        if "ECG" in self.modeltype:
            self.orig_d_ecg = 256
            self.d_ecg = args.embed_dim
            self.ecg_seq_num = 5

            if self.irregular_learn_emb_ecg:
                self.time_attn_ecg = multiTimeAttention(256, self.d_ecg, args.embed_time, 8)
            else:
                self.proj_ecg = nn.Conv1d(self.orig_d_ecg, self.d_ecg, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size - 1) / 2), bias=False)

        output_dim = args.num_labels
        if self.cross_method in ["self_cross", "moe", "hme"]:
            self.trans_self_cross_ts_txt = self.get_cross_network(args, layers=args.cross_layers)
            dim = 0
            if "TS" in self.modeltype:
                dim += self.d_ts
            if "Text" in self.modeltype:
                dim += self.d_txt
            if "CXR" in self.modeltype:
                dim += self.d_cxr
            if "ECG" in self.modeltype:
                dim += self.d_ecg            

            self.proj1 = nn.Linear(dim, dim)
            self.proj2 = nn.Linear(dim, dim)
            self.out_layer = nn.Linear(dim, output_dim)
        else:
            self.d_txt = args.embed_dim
            self.trans_ts_mem = self.get_network(self_type='ts_mem', layers=args.layers)
            self.trans_txt_mem = self.get_network(self_type='txt_mem', layers=args.layers)

            if self.cross_method == "MulT":
                self.trans_txt_with_ts = self.get_network(self_type='txt_with_ts', layers=args.cross_layers)
                self.trans_ts_with_txt = self.get_network(self_type='ts_with_txt', layers=args.cross_layers)
                self.proj1 = nn.Linear((self.d_ts + self.d_txt), (self.d_ts + self.d_txt))
                self.proj2 = nn.Linear((self.d_ts + self.d_txt), (self.d_ts + self.d_txt))
                self.out_layer = nn.Linear((self.d_ts + self.d_txt), output_dim)
            elif self.cross_method == "MAGGate":
                self.gate_fusion = MAGGate(inp1_size=self.d_txt, inp2_size=self.d_ts, dropout=self.dropout)
                self.proj1 = nn.Linear(self.d_txt, self.d_txt)
                self.proj2 = nn.Linear(self.d_txt, self.d_txt)
                self.out_layer = nn.Linear(self.d_txt, output_dim)
            elif self.cross_method == "Outer":
                self.outer_fusion = Outer(inp1_size=self.d_txt, inp2_size=self.d_ts)
                self.proj1 = nn.Linear(self.d_txt, self.d_txt)
                self.proj2 = nn.Linear(self.d_txt, self.d_txt)
                self.out_layer = nn.Linear(self.d_txt, output_dim)
            else:
                self.proj1 = nn.Linear(self.d_ts + self.d_txt, self.d_ts + self.d_txt)
                self.proj2 = nn.Linear(self.d_ts + self.d_txt, self.d_ts + self.d_txt)
                self.out_layer = nn.Linear(self.d_ts + self.d_txt, output_dim)

        if 'ihm' in self.task or 'los' in self.task:
            self.loss_fct1 = nn.CrossEntropyLoss()
        elif 'pheno' in self.task:
            self.loss_fct1 = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def get_network(self, self_type='ts_mem', layers=-1):
        if self_type == 'ts_mem':
            if self.irregular_learn_emb_ts:
                embed_dim, q_seq_len, kv_seq_len = self.d_ts, self.tt_max, None
            else:
                embed_dim, q_seq_len, kv_seq_len = self.d_ts, self.ts_seq_num, None
        elif self_type == 'txt_mem':
            if self.irregular_learn_emb_text:
                embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.tt_max, None
            else:
                embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.text_seq_num, None
        elif self_type == 'txt_with_ts':
            if self.irregular_learn_emb_ts:
                embed_dim, q_seq_len, kv_seq_len = self.d_ts, self.tt_max, self.tt_max
            else:
                embed_dim, q_seq_len, kv_seq_len = self.d_ts, self.text_seq_num, self.ts_seq_num
        elif self_type == 'ts_with_txt':
            if self.irregular_learn_emb_text:
                embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.tt_max, self.tt_max
            else:
                embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.ts_seq_num, self.text_seq_num
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  device=self.device,
                                  attn_dropout=self.dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask,
                                  q_seq_len=q_seq_len,
                                  kv_seq_len=kv_seq_len)

    def get_cross_network(self, args, layers=-1):
        embed_dim, q_seq_len = self.d_ts, self.tt_max
        return TransformerCrossEncoder(args=args,
                                       embed_dim=embed_dim,
                                       num_heads=self.num_heads,
                                       layers=layers,
                                       device=self.device,
                                       attn_dropout=self.dropout,
                                       relu_dropout=self.dropout,
                                       res_dropout=self.dropout,
                                       embed_dropout=self.dropout,
                                       attn_mask=self.attn_mask,
                                       q_seq_len_1=q_seq_len,
                                       num_modalities=self.num_modalities)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def _missing_indices(self, missing_idx):
        all_indices = torch.arange(len(missing_idx))
        missing_indices = torch.nonzero(missing_idx).squeeze(1)
        missing_mask = torch.ones(len(missing_idx), dtype=torch.bool)
        missing_mask[missing_indices] = False
        non_missing = all_indices[missing_mask]
        return missing_indices, non_missing

    def forward(self, x_ts, x_ts_mask, ts_tt_list, cxr_missing=None, text_missing=None, ecg_missing=None, input_ids_sequences=None,
                attn_mask_sequences=None, text_emb=None, note_time_list=None, note_time_mask_list=None,
                labels=None, reg_ts=None, cxr_feats=None, cxr_time=None, cxr_time_mask=None, ecg_feats=None,
                ecg_time=None, ecg_time_mask=None):
        if "TS" in self.modeltype:
            if self.irregular_learn_emb_ts:
                time_key_ts = self.learn_time_embedding(ts_tt_list).to(self.device)
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
                x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)
                proj_x_ts_irg = self.time_attn_ts(time_query, time_key_ts, x_ts_irg, x_ts_mask)
                proj_x_ts_irg = proj_x_ts_irg.transpose(0, 1)

            if self.reg_ts and reg_ts != None:
                x_ts_reg = reg_ts.transpose(1, 2)
                proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts == self.d_ts else self.proj_ts(x_ts_reg)
                proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

            if self.TS_mixup:
                if self.mixup_level == 'batch':
                    g_irg = torch.max(proj_x_ts_irg, dim=0).values
                    g_reg = torch.max(proj_x_ts_reg, dim=0).values
                    moe_gate = torch.cat([g_irg, g_reg], dim=-1)
                elif self.mixup_level == 'batch_seq' or self.mixup_level == 'batch_seq_feature':
                    moe_gate = torch.cat([proj_x_ts_irg, proj_x_ts_reg], dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")
                mixup_rate = self.moe(moe_gate)
                proj_x_ts = mixup_rate * proj_x_ts_irg + (1 - mixup_rate) * proj_x_ts_reg
            else:
                if self.irregular_learn_emb_ts:
                    proj_x_ts = proj_x_ts_irg
                elif self.reg_ts:
                    proj_x_ts = proj_x_ts_reg
                else:
                    raise ValueError("Unknown time series type")
            proj_x_ts += self.token_type_embeddings(torch.zeros((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))

        mod_count = 1
        if "Text" in self.modeltype:
            if self.use_pt_text_embeddings:
                x_txt = text_emb
            else:
                x_txt = self.bertrep(input_ids_sequences, attn_mask_sequences)

            if self.irregular_learn_emb_text:
                time_key = self.learn_time_embedding(note_time_list).to(self.device)
                if not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
                proj_x_txt = self.time_attn_text(time_query, time_key, x_txt, note_time_mask_list)
                proj_x_txt = proj_x_txt.transpose(0, 1)
            else:
                x_txt = x_txt.transpose(1, 2)
                proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(x_txt)
                proj_x_txt = proj_x_txt.permute(2, 0, 1)
            if text_missing is None or torch.all(text_missing == 0):
                proj_x_txt += self.token_type_embeddings(torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))
            elif not torch.all(text_missing == 0):
                missing_indices, non_missing = self._missing_indices(text_missing)
                proj_x_txt[:, non_missing, :] += self.token_type_embeddings(torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
                proj_x_txt[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float32, device=x_ts.device)
            mod_count += 1

        if "CXR" in self.modeltype:
            if self.irregular_learn_emb_cxr:
                time_key = self.learn_time_embedding(cxr_time).to(self.device)
                if not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
                proj_x_cxr = self.time_attn_cxr(time_query, time_key, cxr_feats, cxr_time_mask)
                proj_x_cxr = proj_x_cxr.transpose(0, 1)
            else:
                cxr_feats = cxr_feats.transpose(1, 2)
                proj_x_cxr = cxr_feats if self.orig_d_cxr == self.d_cxr else self.proj_cxr(cxr_feats)
                proj_x_cxr = proj_x_cxr.permute(2, 0, 1)
            if cxr_missing is None or torch.all(cxr_missing == 0):
                proj_x_cxr += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))
            elif not torch.all(cxr_missing == 0):
                missing_indices, non_missing = self._missing_indices(cxr_missing)
                proj_x_cxr[:, non_missing, :] += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
                proj_x_cxr[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float32, device=x_ts.device)
            mod_count += 1

        if "ECG" in self.modeltype:
            if self.irregular_learn_emb_cxr:
                time_key = self.learn_time_embedding(ecg_time).to(self.device)
                if not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
                proj_x_ecg = self.time_attn_ecg(time_query, time_key, ecg_feats, ecg_time_mask)
                proj_x_ecg = proj_x_ecg.transpose(0, 1)
            else:
                ecg_feats = ecg_feats.transpose(1, 2)
                proj_x_ecg = ecg_feats if self.orig_d_ecg == self.d_ecg else self.proj_ecg(ecg_feats)
                proj_x_ecg = proj_x_ecg.permute(2, 0, 1)
            
            if ecg_missing is None or torch.all(ecg_missing == 0):
                proj_x_ecg += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))
            elif not torch.all(ecg_missing == 0):
                missing_indices, non_missing = self._missing_indices(ecg_missing)
                proj_x_ecg[:, non_missing, :] += self.token_type_embeddings(torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
                proj_x_ecg[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float32, device=x_ts.device)
            mod_count += 1

        balance_loss = None
        if self.cross_method in ["self_cross", "moe", "hme"]:
            if self.modeltype == "TS_Text":
                hiddens, balance_loss = self.trans_self_cross_ts_txt([proj_x_txt, proj_x_ts], ['txt', 'ts'])
            elif self.modeltype == "TS_CXR":
                hiddens, balance_loss = self.trans_self_cross_ts_txt([proj_x_cxr, proj_x_ts], ['cxr', 'ts'])
            elif self.modeltype == "TS_CXR_Text":
                hiddens, balance_loss = self.trans_self_cross_ts_txt([proj_x_ts, proj_x_cxr, proj_x_txt], ['ts', 'cxr', 'txt'])
            elif self.modeltype == "TS_CXR_Text_ECG":
                hiddens, balance_loss = self.trans_self_cross_ts_txt([proj_x_ts, proj_x_cxr, proj_x_txt, proj_x_ecg], ['ts', 'cxr', 'txt', 'ecg'])

            if hiddens is None:
                return None
            last_hs = torch.cat([hid[-1] for hid in hiddens], dim=1)
        else:
            if 'CXR' in self.modeltype:
                proj_x_txt = proj_x_cxr
            if self.cross_method == "MulT":
                h_txt_with_ts = self.trans_txt_with_ts(proj_x_txt, proj_x_ts, proj_x_ts)
                h_ts_with_txt = self.trans_ts_with_txt(proj_x_ts, proj_x_txt, proj_x_txt)
                proj_x_ts = self.trans_ts_mem(h_txt_with_ts)
                proj_x_txt = self.trans_txt_mem(h_ts_with_txt)
                last_h_ts = proj_x_ts[-1]
                last_h_txt = proj_x_txt[-1]
                last_hs = torch.cat([last_h_ts, last_h_txt], dim=1)
            else:
                proj_x_ts = self.trans_ts_mem(proj_x_ts)
                proj_x_txt = self.trans_txt_mem(proj_x_txt)
                if self.cross_method == "MAGGate":
                    last_hs = self.gate_fusion(proj_x_txt[-1], proj_x_ts[-1])
                elif self.cross_method == "Outer":
                    last_hs = self.outer_fusion(proj_x_txt[-1], proj_x_ts[-1])
                else:
                    last_hs = torch.cat([proj_x_txt[-1], proj_x_ts[-1]], dim=1)
        
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        if 'ihm' in self.task or 'los' in self.task:
            if labels != None:
                task_loss = self.loss_fct1(output, labels)
                return task_loss, balance_loss
            return torch.nn.functional.softmax(output, dim=-1)[:, 1]
        elif 'pheno' in self.task:
            if labels != None:
                labels = labels.float()
                task_loss = self.loss_fct1(output, labels)
                return task_loss, balance_loss
            return torch.nn.functional.sigmoid(output)

class TSMixed(nn.Module):
    def __init__(self, args, device, modeltype=None, orig_d_ts=None, orig_reg_d_ts=None, ts_seq_num=None):
        super(TSMixed, self).__init__()
        if modeltype != None:
            self.modeltype = modeltype
        else:
            self.modeltype = args.modeltype
        self.num_heads = args.num_heads
        self.attn_mask = False
        self.layers = args.layers
        self.device = device
        self.kernel_size = args.kernel_size
        self.dropout = args.dropout
        self.irregular_learn_emb_ts = args.irregular_learn_emb_ts
        self.irregular_learn_emb_text = args.irregular_learn_emb_text
        self.Interp = args.Interp
        self.reg_ts = args.reg_ts
        self.TS_mixup = args.TS_mixup
        self.mixup_level = args.mixup_level
        self.task = args.task
        self.TS_model = args.TS_model
        self.tt_max = args.tt_max

        self.time_query = torch.linspace(0, 1., self.tt_max)
        self.periodic = nn.Linear(1, args.embed_time - 1)
        self.linear = nn.Linear(1, 1)
        output_dim = args.num_labels
        self.orig_d_ts = orig_d_ts
        self.d_ts = args.embed_dim
        self.ts_seq_num = ts_seq_num

        if self.Interp:
            self.s_intp = S_Interp(args, self.device, self.orig_d_ts)
            self.c_intp = Cross_Interp(args, self.device, self.orig_d_ts)
            self.proj_ts_intp = nn.Conv1d(self.orig_d_ts * 3, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size - 1) / 2), bias=False)

        if self.irregular_learn_emb_ts:
            self.time_attn_ts = multiTimeAttention(self.orig_d_ts * 2, self.d_ts, args.embed_time, 8)

        if self.reg_ts:
            self.orig_reg_d_ts = orig_reg_d_ts
            self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size - 1) / 2), bias=False)

        if self.TS_mixup:
            if self.mixup_level == 'batch':
                self.moe = gateMLP(input_dim=self.d_ts * 2, hidden_size=args.embed_dim, output_dim=1, dropout=self.dropout)
            elif self.mixup_level == 'batch_seq':
                self.moe = gateMLP(input_dim=self.d_ts * 2, hidden_size=args.embed_dim, output_dim=1, dropout=self.dropout)
            elif self.mixup_level == 'batch_seq_feature':
                self.moe = gateMLP(input_dim=self.d_ts * 2, hidden_size=args.embed_dim, output_dim=self.d_ts, dropout=self.dropout)
            else:
                raise ValueError("Unknown mixedup type")

        if self.TS_model == 'LSTM':
            self.trans_ts_mem = nn.LSTM(input_size=self.d_ts, hidden_size=self.d_ts, num_layers=args.layers, dropout=self.dropout, bidirectional=True)
        elif self.TS_model == 'CNN':
            self.trans_ts_mem = TimeSeriesCnnModel(input_size=self.d_ts, n_filters=self.d_ts, filter_size=self.kernel_size,
                                                   dropout=self.dropout, length=self.tt_max, n_neurons=self.d_ts, layers=args.layers)
        elif self.TS_model == 'Atten':
            self.trans_ts_mem = self.get_network(self_type='ts_mem', layers=args.layers)
        
        self.proj1 = nn.Linear(self.d_ts, self.d_ts)
        self.proj2 = nn.Linear(self.d_ts, self.d_ts)
        self.out_layer = nn.Linear(self.d_ts, output_dim)

        if 'ihm' in self.task:
            self.loss_fct1 = nn.CrossEntropyLoss()
        elif 'pheno' in self.task:
            self.loss_fct1 = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def get_network(self, self_type='ts_mem', layers=-1):
        embed_dim = self.d_ts
        if self_type == 'ts_mem':
            if self.irregular_learn_emb_ts:
                q_seq_len = self.tt_max
            else:
                q_seq_len = self.ts_seq_num
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  device=self.device,
                                  attn_dropout=self.dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask,
                                  q_seq_len=q_seq_len,
                                  kv_seq_len=None)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x_ts, x_ts_mask, ts_tt_list, labels=None, reg_ts=None):
        if "TS" in self.modeltype:
            if self.Interp:
                x_ts_mask_interp = copy.deepcopy(x_ts_mask)
                x_ts_interp = copy.deepcopy(x_ts)
                recon_m = hold_out(x_ts_mask_interp)
                recon_m = torch.Tensor(recon_m).to(self.device)
                proj_x_ts_interp = self.proj_ts_intp(self.c_intp(self.s_intp(x_ts_interp, x_ts_mask_interp, ts_tt_list, recon_m)))
                proj_x_ts_interp = proj_x_ts_interp.permute(2, 0, 1)
                recon_interp = self.c_intp(self.s_intp(x_ts_interp, x_ts_mask_interp, ts_tt_list, recon_m, reconstruction=True), reconstruction=True)

            if self.irregular_learn_emb_ts:
                time_key_ts = self.learn_time_embedding(ts_tt_list).to(self.device)
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
                x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)
                proj_x_ts_irg = self.time_attn_ts(time_query, time_key_ts, x_ts_irg, x_ts_mask)
                proj_x_ts_irg = proj_x_ts_irg.transpose(0, 1)

            if self.reg_ts and reg_ts != None:
                x_ts_reg = reg_ts.transpose(1, 2)
                proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts == self.d_ts else self.proj_ts(x_ts_reg)
                proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

            if self.TS_mixup:
                if self.Interp and not self.irregular_learn_emb_ts and self.reg_ts:
                    proj_x_ts_irg = proj_x_ts_interp
                if self.Interp and self.irregular_learn_emb_ts and not self.reg_ts:
                    proj_x_ts_reg = proj_x_ts_interp
                if self.mixup_level == 'batch':
                    g_irg = torch.max(proj_x_ts_irg, dim=0).values
                    g_reg = torch.max(proj_x_ts_reg, dim=0).values
                    moe_gate = torch.cat([g_irg, g_reg], dim=-1)
                elif self.mixup_level == 'batch_seq' or self.mixup_level == 'batch_seq_feature':
                    moe_gate = torch.cat([proj_x_ts_irg, proj_x_ts_reg], dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")

                mixup_rate = self.moe(moe_gate)
                proj_x_ts = mixup_rate * proj_x_ts_irg + (1 - mixup_rate) * proj_x_ts_reg
            else:
                if self.irregular_learn_emb_ts:
                    proj_x_ts = proj_x_ts_irg
                elif self.reg_ts:
                    proj_x_ts = proj_x_ts_reg
                else:
                    raise ValueError("Unknown time series type")

            if self.TS_model == 'CNN':
                proj_x_ts = proj_x_ts.permute(1, 2, 0)
                proj_x_ts = self.trans_ts_mem(proj_x_ts)
            elif self.TS_model == 'LSTM':
                _, (proj_x_ts, _) = self.trans_ts_mem(proj_x_ts)
            else:
                proj_x_ts = self.trans_ts_mem(proj_x_ts)
                
            if self.TS_model != 'CNN':
                last_h_ts = proj_x_ts[-1]
            else:
                last_h_ts = proj_x_ts

            if self.modeltype == "TS":
                last_hs = last_h_ts
            else:
                raise ValueError("Unknown model type")
                        
            last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_h_ts)), p=self.dropout, training=self.training))
            last_hs_proj += last_hs
            output = self.out_layer(last_hs_proj)

        if self.Interp:
            reconloss_interp = recon_loss(x_ts_interp, x_ts_mask_interp, recon_m, recon_interp, self.d_ts)

        if 'ihm' in self.task:
            if labels != None:
                if self.Interp:
                    return self.loss_fct1(output, labels) + reconloss_interp
                else:
                    return self.loss_fct1(output, labels)
            return torch.nn.functional.softmax(output, dim=-1)[:, 1]
        elif 'pheno' in self.task:
            if labels != None:
                labels = labels.float()
                if self.Interp:
                    return self.loss_fct1(output, labels) + reconloss_interp
                else:
                    return self.loss_fct1(output, labels)
            return torch.nn.functional.sigmoid(output)

# =============================================================================
# PROJECT-SPECIFIC ADDITION -- FuseMoE baseline adapter (Milestone 3)
#
# Everything above this line is the original, unmodified fusemoe.py (vendored
# reproduction of the FuseMoE paper's MULTCrossModel). Everything below is new
# code for our sepsis-onset project. It translates between our standardized
# SepsisDataset / collate_sepsis_batch batch format (dataset.py) and the raw
# argument signature MULTCrossModel.forward() expects, so train.py can call
# FuseMoEBaseline exactly like every other model in MODEL_REGISTRY.
#
# Paste everything below this comment block onto the bottom of fusemoe.py.
# =============================================================================

import warnings
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# dataset.py defines the fixed 17-variable TS vocabulary this adapter needs
# to know the cardinality of (to build the [B, T, D] grid MULTCrossModel's
# mTAND branch expects). train.py's own imports rely on `experiments/` being
# on sys.path (auto-added when `python experiments/train.py` is the entry
# point) -- see train.py's REPO_ROOT comment. We rely on the same mechanism
# here. If this file is ever imported standalone (e.g. running it directly
# for the dummy test below) that path won't be set up, so we fall back to
# the known cardinality of dataset.py's ALL_17_VARIABLES rather than crash.
# --------------------------------------------------------------------------
try:
    from dataset import VARIABLE_VOCAB
    _N_TS_VARS_DEFAULT = len(VARIABLE_VOCAB)
except ImportError:
    _N_TS_VARS_DEFAULT = 17  # ASSUMPTION: mirrors dataset.py's ALL_17_VARIABLES count


class _MinimalBertConfig:
    """Stand-in for a real HF PretrainedConfig. MULTCrossModel's __init__ always
    builds a BertForRepresentation when 'Text' is in modeltype -- even though we
    set use_pt_text_embeddings=True below, which means BertForRepresentation is
    constructed but its .forward() is never actually called. BertForRepresentation
    .__init__ still does `BioBert.config.hidden_dropout_prob`, so *some* object with
    that attribute is required at construction time regardless. This avoids
    downloading/loading a real pretrained transformer just to satisfy that
    unused attribute access.

    ASSUMPTION: if you ever flip use_pt_text_embeddings to False (to fine-tune
    text end-to-end instead of consuming precomputed embeddings), replace this
    with a real `AutoModel.from_pretrained(text_model_name)` -- this mock will
    NOT work for that path since BertForRepresentation.forward() calls
    `self.bert(input_ids, attention_mask=...)` on it.
    """
    hidden_dropout_prob = 0.1


class _MinimalBert:
    config = _MinimalBertConfig()


class FuseMoEBaseline(nn.Module):
    """
    Adapter around the raw `MULTCrossModel` (FuseMoE paper reproduction) that
    speaks our project's standardized batch format, for Milestone 3 baseline
    reproduction (PROJECT_CONTEXT.md rule #1: baseline numbers must come from
    our own re-run, never copied from the paper).

    Configured for exactly 3 modalities -- TS, Text (notes), CXR -- with
    cross_method='moe' (sparse mixture-of-experts fusion), per the task spec.
    We do NOT reproduce ECG; ECG-related args to MULTCrossModel are left None
    and 'ECG' is never in the mocked modeltype string, so those code paths in
    the raw model are simply skipped.

    ---------------------------------------------------------------------
    EXPECTED `batch` DICT KEYS (see dataset.py's collate_sepsis_batch)
    ---------------------------------------------------------------------
    Keys marked **NEW** are NOT produced by collate_sepsis_batch today (it
    deliberately keeps notes/CXR as raw text / image paths, per its own
    docstring: "tokenization / JPEG loading is model-specific"). You'll need
    an upstream step (either inside a DataLoader wrapper, or a one-time
    precompute pass over notes.parquet / cxr_metadata.parquet) that attaches
    these before batches reach this adapter.

        batch["label"]              float32 [B]              0/1 sepsis-within-horizon label

        batch["ts"]["var_idx"]      int64   [B, T_ts]         VARIABLE_VOCAB index per raw TS
                                                                event; -1 marks a padded slot
        batch["ts"]["value"]        float32 [B, T_ts]         raw (unnormalized) value per event
        batch["ts"]["hours"]        float32 [B, T_ts]         hours-since-admission per event
        batch["ts"]["mask"]         bool    [B, T_ts]         True where a real (non-pad) event
                                                                exists at that slot

        batch["notes"]["emb"]     **float32 [B, T_notes, 768]  precomputed BioBERT / Clinical-
                                                                Longformer embedding per note
                                                                (pooled/[CLS], NOT raw token ids --
                                                                see "Text encoder" note below)
        batch["notes"]["hours"]     float32 [B, T_notes]      hours-since-admission per note
        batch["notes"]["mask"]      bool    [B, T_notes]      True where a real note exists

        batch["cxr"]["feats"]     **float32 [B, T_cxr, 1024]   precomputed CXR image embedding
                                                                per study (NOT raw JPEG paths)
        batch["cxr"]["hours"]       float32 [B, T_cxr]        hours-since-admission per study
        batch["cxr"]["mask"]        bool    [B, T_cxr]        True where a real study exists

    ---------------------------------------------------------------------
    OUTPUT (forward() return value)
    ---------------------------------------------------------------------
        {"loss": task_loss, "logits": logits, "balance_loss": balance_loss}

        loss          scalar tensor, BCE-with-logits task loss
        logits        float32 [B], RAW pre-sigmoid logits (see "Logit capture" below)
        balance_loss  scalar tensor (MoE load-balancing aux loss) or possibly None
                      -- see "Known quirks" below

    NOTE ON train.py: run_epoch() currently does `logits = model(batch)` and
    computes loss itself via an external `nn.BCEWithLogitsLoss()`. This
    adapter intentionally does NOT match that convention (the task spec for
    this adapter requires the {"loss","logits","balance_loss"} dict). You'll
    need to special-case the 'fusemoe' entry in train.py's run_epoch (e.g.
    `out = model(batch); logits = out["logits"]; loss = out["loss"] + aux_weight
    * out["balance_loss"]`) rather than reusing the bare-logits path used by
    utde / sanity_baseline.

    ---------------------------------------------------------------------
    ASSUMPTIONS TO CHECK / ADJUST (flagged per the task spec)
    ---------------------------------------------------------------------
    1. Text encoder: adapter uses `use_pt_text_embeddings=True`, so it expects
       precomputed 768-dim note embeddings rather than loading/fine-tuning a
       live BioBERT inside this adapter. This is a deliberate scope choice --
       instantiating a real BertForRepresentation here would need network
       access to pull weights and would make this baseline far more expensive
       to iterate on than the paper's own setup implies. Flip
       use_pt_text_embeddings=False and swap `_MinimalBert()` for a real
       `AutoModel.from_pretrained(text_model_name)` if you want end-to-end
       text fine-tuning instead.
    2. Hardcoded embedding dims in the RAW model (not this adapter -- these
       live in fusemoe.py's MULTCrossModel.__init__ and cannot be overridden
       via args): text embeddings MUST be exactly 768-dim
       (`multiTimeAttention(768, ...)` for text is hardcoded, independent of
       orig_d_txt) and CXR embeddings MUST be exactly 1024-dim
       (`self.orig_d_cxr = 1024` is hardcoded, never read from args or the
       constructor). batch["notes"]["emb"] / batch["cxr"]["feats"] must match
       these exactly or you'll get a shape-mismatch deep inside multiTimeAttention.
    3. TS event->grid expansion: dataset.py gives a flat, lossless
       (var_idx, value, hours) event stream, not the [B, T, D] "one row per
       observation time, several variables per row" grid mTAND-style models
       usually assume. This adapter expands each raw event into its own row
       (one populated variable column per row) rather than grouping
       same-timestamp events together -- see `_events_to_grid()`. This keeps
       full fidelity of dataset.py's lossless event stream (PROJECT_CONTEXT.md:
       dataset.py "does NOT bin, truncate, or reduce anything") and pushes all
       reduction into this baseline's own adapter, where it belongs.
    4. Time normalization: raw hours-since-admission are divided by
       `lookback_hours` (default 48.0, matching utde.yaml's convention) and
       clamped to [0, 1] to match MULTCrossModel's `time_query = linspace(0,1,tt_max)`
       reference grid. If your admissions can exceed 48h of relevant history,
       raise lookback_hours accordingly.
    5. Task / loss head: mocked args.task='pheno' to hit MULTCrossModel's
       BCEWithLogitsLoss branch with num_labels=1 (single-logit binary
       classification), matching train.py's run_epoch convention
       (`nn.BCEWithLogitsLoss()`, float labels) even though 'pheno' originally
       named MIMIC-III's multi-label phenotyping task in the source paper.
       Labels are reshaped to [B, 1] to match the raw model's [B, num_labels]
       output.
    6. MoE hyperparameters (num_of_experts, top_k, router_type,
       gating_function): utils/config.py's MoEConfig and core/sparse_moe.py's
       MoE weren't in the provided context, so valid values for `router_type`
       / `gating_function` are a best guess ('joint' / 'softmax'). If
       construction fails on these, check those two files for the actual
       accepted values.
    7. `n_ts_vars` defaults to len(VARIABLE_VOCAB) from dataset.py (17) via a
       best-effort import; override explicitly if your TS vocabulary changes.

    ---------------------------------------------------------------------
    KNOWN QUIRKS IN THE VENDORED MODEL (not fixed here -- fusemoe.py is not
    to be rewritten, per the task spec; documented so they aren't a surprise)
    ---------------------------------------------------------------------
    - `TransformerCrossEncoderLayer.__init__` unconditionally does
      `self.moe = self.moe.to('cuda:0')` when cross_method='moe'. This means
      MULTCrossModel construction ITSELF requires a CUDA device to exist on
      the machine, regardless of what `device` you pass to this adapter --
      even if you intend to eventually run everything on CPU, construction
      will raise before you get the chance. We `.to(device)` the whole model
      again after construction to fix final placement, but that can't help
      the construction-time failure. See the warning emitted in __init__.
    - The missing-modality zero-fill inside MULTCrossModel.forward hardcodes
      `dtype=torch.float16` for the zeroed-out slice regardless of the
      model's actual working dtype (float32 by default). PyTorch generally
      handles the implicit cast on indexed assignment fine, but it's worth
      knowing about if you see dtype warnings.
    - If TransformerCrossEncoderLayer detects NaNs in the concatenated
      modality embeddings mid-fusion, MULTCrossModel.forward returns a bare
      `None` instead of `(task_loss, balance_loss)`. This adapter checks for
      that and raises a clearer RuntimeError instead of letting an opaque
      "cannot unpack non-iterable NoneType" propagate.
    """

    def __init__(
        self,
        device: str = "cpu",
        # -- core FuseMoE / mTAND hyperparams (names mirror utde.yaml / example_config.yaml) --
        embed_dim: int = 64,
        embed_time: int = 16,
        tt_max: int = 48,
        layers: int = 2,
        cross_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        kernel_size: int = 3,
        # -- sparse-MoE cross-modal fusion hyperparams (cross_method='moe') --
        hidden_size: int = 128,
        num_of_experts: Sequence[int] = (4,),
        top_k: Sequence[int] = (2,),
        router_type: str = "joint",          # ASSUMPTION -- see class docstring point 6
        gating_function: Sequence[str] = ("softmax",),  # ASSUMPTION -- see class docstring point 6
        # -- project-specific --
        lookback_hours: float = 48.0,
        n_ts_vars: Optional[int] = None,
        num_labels: int = 1,
        text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    ):
        super().__init__()

        self.device = device
        self.embed_dim = embed_dim
        self.embed_time = embed_time
        self.tt_max = tt_max
        self.layers = layers
        self.cross_layers = cross_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_of_experts = list(num_of_experts)
        self.top_k = list(top_k)
        self.router_type = router_type
        self.gating_function = list(gating_function)
        self.lookback_hours = lookback_hours
        self.n_ts_vars = n_ts_vars if n_ts_vars is not None else _N_TS_VARS_DEFAULT
        self.num_labels = num_labels
        self.text_model_name = text_model_name

        is_cuda_device = str(device).startswith("cuda")
        if not is_cuda_device:
            warnings.warn(
                "FuseMoEBaseline: constructing MULTCrossModel with cross_method='moe' "
                "on device=%r. The vendored fusemoe.py code hardcodes "
                "`self.moe = self.moe.to('cuda:0')` inside TransformerCrossEncoderLayer, "
                "so a CUDA device must be physically present for construction to "
                "succeed at all, even though you asked for %r. If no CUDA device is "
                "available this will raise a RuntimeError momentarily, from inside "
                "MULTCrossModel.__init__, not from this adapter." % (device, device)
            )

        args = self._build_mult_args()

        # modeltype string is checked via substring ("TS" in modeltype, "Text" in
        # modeltype, "CXR" in modeltype) AND via exact match in the moe/hme/self_cross
        # fusion branch ( `elif self.modeltype == "TS_CXR_Text":` ) -- this exact
        # string is required for both checks to line up.
        self.model = MULTCrossModel(
            args=args,
            device=device,
            modeltype="TS_CXR_Text",
            orig_d_ts=self.n_ts_vars,
            orig_reg_d_ts=None,   # reg_ts=False in mocked args -- unused
            orig_d_txt=768,       # see class docstring point 2 -- functionally hardcoded anyway
            ts_seq_num=None,      # unused: irregular_learn_emb_ts=True uses tt_max instead
            text_seq_num=None,    # unused: irregular_learn_emb_text=True uses tt_max instead
            Biobert=_MinimalBert(),
        )
        self.model = self.model.to(device)

        # --- Logit capture via forward hook -------------------------------------
        # MULTCrossModel.forward only returns (task_loss, balance_loss) when labels
        # are given, or post-activation probabilities (softmax/sigmoid) when they
        # aren't -- it never directly returns raw logits. We hook self.model.out_layer
        # (the final nn.Linear producing pre-activation `output` in the raw model's
        # forward) to capture the exact tensor that feeds the loss / activation, so we
        # can report true logits regardless of which branch executes.
        self._captured_logits: Optional[torch.Tensor] = None
        self.model.out_layer.register_forward_hook(self._capture_logits_hook)

    def _capture_logits_hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
        self._captured_logits = output

    def _build_mult_args(self) -> SimpleNamespace:
        """Mocks the argparse-style `args` namespace MULTCrossModel expects.
        See class docstring for which values are confirmed-necessary vs.
        best-guess placeholders."""
        return SimpleNamespace(
            num_heads=self.num_heads,
            layers=self.layers,
            cross_layers=self.cross_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            # -- irregular ("mTAND-style") time embedding, one flag per modality --
            irregular_learn_emb_ts=True,     # our TS is a sparse event stream, not fixed-grid
            irregular_learn_emb_text=True,   # notes arrive at irregular times
            irregular_learn_emb_cxr=True,    # CXRs arrive at irregular times -- this is the
                                              # whole point of PROJECT_CONTEXT.md's cxr_linking.py
                                              # note to "keep FULL timestamped sequence, not just
                                              # latest" (unlike MedPatch's most-recent-only gap)
            irregular_learn_emb_ecg=False,   # unused -- ECG not one of our 3 modalities, but
                                              # MULTCrossModel.__init__ reads this attribute
                                              # unconditionally so it must exist
            reg_ts=False,                    # we have no fixed-grid resampled TS, only irregular
            TS_mixup=False,                  # only meaningful when reg_ts=True as well
            mixup_level="batch",             # unused (TS_mixup=False); kept for attribute safety
            task="pheno",                    # -> BCEWithLogitsLoss branch; see docstring point 5
            tt_max=self.tt_max,
            cross_method="moe",
            num_modalities=3,                # TS, Text, CXR
            use_pt_text_embeddings=True,     # see docstring point 1
            embed_dim=self.embed_dim,
            embed_time=self.embed_time,
            num_labels=self.num_labels,
            model_name=self.text_model_name, # only read by BertForRepresentation, which is
                                              # constructed but never called (see _MinimalBert)
            # -- sparse MoE fusion (cross_method='moe') --
            hidden_size=self.hidden_size,
            num_of_experts=self.num_of_experts,
            top_k=self.top_k,
            router_type=self.router_type,
            gating_function=self.gating_function,
        )

    def _events_to_grid(
        self,
        var_idx: torch.Tensor,
        value: torch.Tensor,
        hours: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expands dataset.py's flat (var_idx, value, hours) event stream --
        already padded to [B, T_ts] by collate_sepsis_batch's `_pad_stack` --
        into the [B, T_ts, D] "one raw event per row" grid MULTCrossModel's
        mTAND branch (multiTimeAttention) expects. Only the observed
        variable's column is populated per row; all others are zero and
        masked out. See class docstring point 3 for why we don't merge
        same-timestamp multi-variable events into shared rows.
        """
        B, T = value.shape
        D = self.n_ts_vars
        # collate_sepsis_batch pads var_idx with -1; clamp before one_hot so padded
        # rows don't error (they get zeroed out below via `mask` anyway).
        var_idx_safe = var_idx.clamp(min=0)
        one_hot = F.one_hot(var_idx_safe, num_classes=D).to(value.dtype)  # [B, T, D]
        x_ts = one_hot * value.unsqueeze(-1)                              # [B, T, D]
        x_ts_mask = one_hot * mask.unsqueeze(-1).to(value.dtype)          # [B, T, D], pad rows -> all-zero
        ts_tt_list = (hours / self.lookback_hours).clamp(0.0, 1.0)        # [B, T]
        return x_ts, x_ts_mask, ts_tt_list

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor: # Đổi kiểu trả về thành torch.Tensor
        device = self.device
        ts = batch["ts"]
        notes = batch["notes"]
        cxr = batch["cxr"]

        label = batch["label"].to(device)  # [B], float32, 0/1

        # ---- TS: dataset.py's flat event stream -> MULTCrossModel's [B, T, D] grid ----
        x_ts, x_ts_mask, ts_tt_list = self._events_to_grid(
            ts["var_idx"].to(device),
            ts["value"].to(device),
            ts["hours"].to(device),
            ts["mask"].to(device),
        )

        # =====================================================================
        # FIX 1: TỰ ĐỘNG BÙ ĐẮP DỮ LIỆU BỊ THIẾU TỪ TRONG MODEL
        # Bỏ đi các dòng raise KeyError và thay bằng zero tensor
        # =====================================================================
        # ---- Notes ----
        if "emb" not in notes:
            B, T_notes = notes["mask"].shape
            text_emb = torch.zeros((B, T_notes, 768), dtype=torch.float32, device=device)
        else:
            text_emb = notes["emb"].to(device)

        note_time_list = (notes["hours"].to(device) / self.lookback_hours).clamp(0.0, 1.0)
        note_time_mask_list = notes["mask"].to(device)
        text_missing = (~notes["mask"].to(device).any(dim=1)).float()

        # ---- CXR ----
        if "feats" not in cxr:
            B, T_cxr = cxr["mask"].shape
            cxr_feats = torch.zeros((B, T_cxr, 1024), dtype=torch.float32, device=device)
        else:
            cxr_feats = cxr["feats"].to(device)

        cxr_time = (cxr["hours"].to(device) / self.lookback_hours).clamp(0.0, 1.0)
        cxr_time_mask = cxr["mask"].to(device)
        cxr_missing = (~cxr["mask"].to(device).any(dim=1)).float()
        # =====================================================================

        # BCEWithLogitsLoss (task='pheno' branch) expects output/labels shapes to
        # match; out_layer produces [B, num_labels] so labels needs the same rank.
        labels_for_model = label.unsqueeze(-1)  # [B, 1] (num_labels=1 by default)

        self._captured_logits = None
        raw_output = self.model(
            x_ts=x_ts,
            x_ts_mask=x_ts_mask,
            ts_tt_list=ts_tt_list,
            cxr_missing=cxr_missing,
            text_missing=text_missing,
            ecg_missing=None,
            input_ids_sequences=None,     # unused: use_pt_text_embeddings=True
            attn_mask_sequences=None,     # unused: use_pt_text_embeddings=True
            text_emb=text_emb,
            note_time_list=note_time_list,
            note_time_mask_list=note_time_mask_list,
            labels=labels_for_model,
            reg_ts=None,                  # unused: reg_ts=False
            cxr_feats=cxr_feats,
            cxr_time=cxr_time,
            cxr_time_mask=cxr_time_mask,
        )

        if raw_output is None:
            # See class docstring "Known quirks": TransformerCrossEncoderLayer
            # returns (None, None) on a NaN guard, and MULTCrossModel.forward's moe
            # branch propagates that as a bare `None` instead of a 2-tuple.
            raise RuntimeError(
                "MULTCrossModel.forward returned None. Per the vendored fusemoe.py "
                "code this means NaNs were detected in one of the modality "
                "embeddings mid-fusion (TransformerCrossEncoderLayer's "
                "`if torch.isnan(embeddings).any(): return None, None` guard) and "
                "the forward pass aborted. Check x_ts / text_emb / cxr_feats for "
                "NaNs -- a common cause is an all-padding modality that wasn't "
                "reflected in text_missing / cxr_missing."
            )
        task_loss, balance_loss = raw_output

        if self._captured_logits is None:
            raise RuntimeError(
                "Forward hook on self.model.out_layer never fired -- MULTCrossModel's "
                "forward may not have reached out_layer for this modeltype/cross_method "
                "combination. Re-check that modeltype=='TS_CXR_Text' and cross_method=='moe'."
            )
        logits = self._captured_logits.squeeze(-1)  # [B, 1] -> [B]

        # =====================================================================
        # FIX 2: ÉP KIỂU TRẢ VỀ ĐỂ KHỚP VỚI train.py
        # Bỏ đi việc trả về dictionary
        # =====================================================================
        return logits

# =============================================================================
# Dummy test block
# =============================================================================
if __name__ == "__main__":
    # Construction itself requires a CUDA device to exist (see class docstring
    # "Known quirks" -- the vendored TransformerCrossEncoderLayer hardcodes
    # `.to('cuda:0')` for its internal MoE submodule when cross_method='moe').
    # We skip gracefully rather than crash on a CPU-only dev machine / CI runner.
    if not torch.cuda.is_available():
        print(
            "Skipping FuseMoEBaseline dummy test: no CUDA device available. "
            "fusemoe.py's vendored MoE fusion layer hardcodes `.to('cuda:0')` at "
            "construction time regardless of the device you request -- see the "
            "class docstring's 'Known quirks' section."
        )
    else:
        torch.manual_seed(0)
        _device = "cuda:0"
        _B, _T_TS, _T_NOTES, _T_CXR = 4, 20, 3, 2
        _D_TS = _N_TS_VARS_DEFAULT

        dummy_batch = {
            "label": torch.randint(0, 2, (_B,)).float(),
            "ts": {
                "var_idx": torch.randint(0, _D_TS, (_B, _T_TS)),
                "value": torch.randn(_B, _T_TS),
                "hours": torch.rand(_B, _T_TS) * 48.0,
                "mask": torch.ones(_B, _T_TS, dtype=torch.bool),
            },
            "notes": {
                "emb": torch.randn(_B, _T_NOTES, 768),
                "hours": torch.rand(_B, _T_NOTES) * 48.0,
                "mask": torch.ones(_B, _T_NOTES, dtype=torch.bool),
            },
            "cxr": {
                "feats": torch.randn(_B, _T_CXR, 1024),
                "hours": torch.rand(_B, _T_CXR) * 48.0,
                "mask": torch.ones(_B, _T_CXR, dtype=torch.bool),
            },
        }
        # Mark one patient in the batch as CXR-missing entirely, to exercise the
        # missing-modality zero-fill path.
        dummy_batch["cxr"]["mask"][0, :] = False

        model = FuseMoEBaseline(device=_device, embed_dim=32, hidden_size=64, tt_max=16).to(_device)
        dummy_batch = {
            k: ({kk: vv.to(_device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(_device))
            for k, v in dummy_batch.items()
        }

        out = model(dummy_batch)
        assert set(out.keys()) == {"loss", "logits", "balance_loss"}, out.keys()
        assert out["logits"].shape == (_B,), out["logits"].shape
        assert torch.isfinite(out["loss"]), out["loss"]
        print("loss:", out["loss"].item())
        print("logits:", out["logits"].detach().cpu().tolist())
        print("balance_loss:", None if out["balance_loss"] is None else out["balance_loss"].item())

        out["loss"].backward()
        print("backward() OK -- gradients flow through the adapter.")
        print("FuseMoEBaseline dummy test passed.")