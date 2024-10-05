import torch
import numpy as np
import pickle
import pandas as pd


class WeightInitialization:
    def __init__(
            self,
            shape=None,
            noise_init_scale=1e-2,
            scale='log',  # lin (-inf, inf), log (0, inf), sig (0, 1)
            requires_grad=True,
            random=True,
            seed=1234,
    ):
        '''
        class for weight initialization.
        :param num_cells: how many cell types (to expand float inits).
        :param noise_init_scale: standard deviation of noise initialization.
        :param log: boolean. return parameter in log scale
        :param requires_grad: whether to fit.
        :param random: if False: no random initialization takes place.
        '''
        self.shape = shape
        self.noise_init_scale = noise_init_scale
        self.scale = scale
        self.requires_grad = requires_grad
        self.random = random
        self.seed = seed

    def initialize(self, init, shape=None, scale=None, requires_grad=None,
                   random=None):
        """
        for weight initialization. scales noise up if init>1.
        :param init: initializing value, tensor, np.array or float
        :param scale: which scale to use
        :param fit: if False: no random initialization takes place and no grad.
        :return: noisy initial values
        """
        if shape is None:
            shape = self.shape
        if scale is None:
            scale = self.scale
        if requires_grad is None:
            requires_grad = self.requires_grad
        if random is None:
            random = self.random
        init = torch.tensor(init, dtype=torch.float32)
        if not init.shape:
            init = init * torch.ones(shape)
        if random:
            torch.manual_seed(self.seed)
            noise = torch.randn(init.shape) * self.noise_init_scale
            # if largest absolute value is > 1, scale up noise
            noise *= 1 + torch.relu(torch.max(torch.abs(init)) - 1)
            init += noise
        if scale == 'lin':
            pass
        elif scale == 'log':
            init = torch.log(init.clamp(min=1e-6))
        elif scale == 'sig':
            init = init.clamp(min=1e-6, max=1 - 1e-6)
            init = torch.log(init / (1 - init))
        self.seed += 1  # change seed for next call
        return torch.nn.Parameter(data=init, requires_grad=requires_grad)


def get_iGluSnFR_kernel(
        duration=0.6,  # how long the kernel will be
        dt=1 / 30,  # One over sampling frequency
        tau_r=-0.09919711,  # rise time constant
        tau_d=-0.04098927):  # decay time constant
    # from https://github.com/berenslab/abc-ribbon/blob/master/standalone_model/ribbonv2.py
    t_kernel = np.arange(0, duration, dt)
    kernel = np.exp(t_kernel / tau_d) - np.exp(
        (tau_d + tau_r) / (tau_d * tau_r) * t_kernel)
    kernel /= np.sum(kernel)
    kernel = kernel.astype(np.float32)[None, None, ::-1]  # out x in x depth
    return kernel.copy()


def get_GCaMP6f_singleAP_kernel(
        duration=0.6,  # how long the kernel will be
        dt=1 / 30,  # One over sampling frequency
        tau_r=-0.030747978105432443,  # rise time constant
        tau_d=-0.10249318503347234):  # decay time constant
    t_kernel = np.arange(0, duration, dt)
    kernel = np.exp(t_kernel / tau_d) - np.exp(
        (tau_d + tau_r) / (tau_d * tau_r) * t_kernel)
    kernel /= np.sum(kernel)
    kernel = kernel.astype(np.float32)[None, None, ::-1]  # out x in x depth
    return kernel.copy()


def precomputed_sigmoid(x, sigmoid_offset, sigmoid_slope):
    """
    Same as sigmoid of the LNR model, but precomputed to be faster in loop.
    :param x: C
    :param: sigmoid_offset
    :param sigmoid_slope
    :return:
    """
    return torch.sigmoid((x - sigmoid_offset) * sigmoid_slope)


def torch_correlation(x, y):
    """
    returns correlation.
    """
    x = x - torch.mean(x, dim=1, keepdim=True)
    x = x / (torch.norm(x, 2, dim=1, keepdim=True) + 1e-10)
    y = y - torch.mean(y, dim=1, keepdim=True)
    y = y / (torch.norm(y, 2, dim=1, keepdim=True) + 1e-10)
    return torch.mean(torch.sum(x * y, dim=1))