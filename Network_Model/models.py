import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import WeightInitialization, get_iGluSnFR_kernel, \
    get_GCaMP6f_singleAP_kernel, precomputed_sigmoid


class AmacrineCellFeedback(nn.Module):
    """
    models ACs as a linear-nonlinear model
    which get inputs from all BCs, linearly weighted by weights
    with correct corresponding shapes nAC>1 are modelled
    """
    def __init__(
            self,
            bc_ac_weight_init,
            ac_tau_rise_init,
            ac_tau_decay_init,
            ac_sigmoid_slope_init,
            ac_sigmoid_offset_init,
            noise_init_scale=1e-2,
            input_frequency=64,  # in Hz
            random_init=True,
            seed=1234,
    ):
        super().__init__()

        self.initializer = WeightInitialization(
            bc_ac_weight_init.shape[0], noise_init_scale=noise_init_scale,
            random=random_init, seed=seed)

        # input weights from BC to AC
        self.log_bc_ac_weight = self.initializer.initialize(
            bc_ac_weight_init)

        # linear filter
        self.log_ac_tau_rise = self.initializer.initialize(ac_tau_rise_init)
        self.log_ac_tau_decay = self.initializer.initialize(ac_tau_decay_init)

        # non-linearity
        self.log_ac_sigmoid_slope = self.initializer.initialize(
            ac_sigmoid_slope_init)
        self.ac_sigmoid_offset = self.initializer.initialize(
            ac_sigmoid_offset_init, scale='lin')

        # check correct input dimensions for AC
        assert self.log_ac_tau_rise.shape == self.log_ac_tau_decay.shape
        assert self.log_ac_tau_rise.shape == self.log_ac_sigmoid_slope.shape
        assert self.log_ac_tau_rise.shape == self.ac_sigmoid_offset.shape
        assert self.log_bc_ac_weight.shape[0] == self.log_ac_tau_rise.shape[0]

        # Static Parameters
        self.input_frequency = input_frequency

        # kernel parameters (linear filter: double exponential kernel)
        self.kernel_time = torch.nn.Parameter(
            data=torch.arange(0.8, 0, step=-1 / self.input_frequency)[None, :],
            requires_grad=False)
        self.kernel_shape = self.kernel_time.shape

    def forward(self, x, bc_ac_weight, ac_kernel, ac_sigmoid_slope):
        """
        alias run_one_step (has to be fast because in loop!)
        :param: x shape: (nBC_helper, kernel.shape[1])
        :param: bc_ac_weight: precomputed bc_ac_weight
        :param: ac_kernel: precomputed ac_kernel
        :param: ac_sigmoid_slope
        self.bc_ac_weight: (nAC, nBC_helper)
        :return: shape(nAC)
        """
        # compute linear combination for inputs
        x = torch.mm(bc_ac_weight, x)  # (nAC, history_len)

        # zero padding for starting
        if x.shape[-1] < self.kernel_shape[-1]:
            x = F.pad(x, (self.kernel_shape[-1] - x.shape[1], 0),
                      mode='constant', value=0)  # x[0])
            # ToDo: can this below be removed?
            #  not sure. the one below would be the better(?) option
            # option for padding not 0 but first value
            # not yet working for empty tensors (if i==0)
            # x[:, :self.kernel.shape[-1]] = (x[:, :self.kernel.shape[-1]].T
            # * x[:, self.kernel.shape[-1]]).T

        # linear filtering
        x = torch.sum(ac_kernel * x, dim=1)

        # non-linearity
        x = precomputed_sigmoid(x, self.ac_sigmoid_offset, ac_sigmoid_slope)
        return x

    def compute_ac_kernel(self):
        """
        models the linear filter of an AC.
        as double exponential kernel.
        :return: nAC x kernels
        """
        # all variables in sec
        # ToDo: aren't these scalars? why the transpose? no. shape = (nAC,1)
        tau_rise = torch.exp(self.log_ac_tau_rise).T[:, None]
        tau_decay = torch.exp(self.log_ac_tau_decay).T[:, None]

        # shape: (nAC, tpts_kernel)
        kernel = torch.exp(- self.kernel_time / tau_decay) - torch.exp(
            -(tau_decay + tau_rise) / (tau_decay * tau_rise) * self.kernel_time)

        # normalize kernel
        kernel = kernel / torch.norm(kernel, dim=1, keepdim=True)
        return kernel

    
class FullBCModel(nn.Module):
    def __init__(
        self,
        cell_types=np.ones((24,2)),  # 1=On, -1=OFF cell
        num_pr = 2, # Number of photoreceptors per BC
        sigmoid_slope_init=15.0,  # float or torch tensor
        sigmoid_offset_init=0.5,
        change_prob01_init=1e-1,
        change_prob12_init=1e-1,
        intermediate_pool_capacity_init=50.0,
        release_pool_capacity_init=10.0,
        input_frequency=64,  # in Hz
        steady_state_steps=0,  # in seconds
        ip_steady=np.ones(14),  # intermediate pool, full at start fraction
        rrp_steady=np.ones(14),  # ready release pool
        noise_init_scale=1e-2,
        kernel_speed_init=1,
        random_init=True,
        seed=1234,
        fit_linear_filter=True,
        fit_non_linearity=True,
        fit_release_parameters=True,
        fit_steady_state=True,
        pr_bc_weight_init =1 / 6,
        # AC feedback
        num_acl=45,
        bc_acl_weight_init=1 / 14,
        acl_tau_rise_init=0.1,
        acl_tau_decay_init=0.1,
        acl_sigmoid_slope_init=2.0,
        acl_sigmoid_offset_init=1.5,
        acl_bc_weight_init=1e-2,
        acl_acl_weight_init=1/25,
        initial_affine=True,
        kernel_type='biphasic',
    ):
        super().__init__()

        # General Static Parameters
        self.num_bc = cell_types.shape[0]
        self.num_pr = num_pr
        self.cell_types = torch.nn.Parameter(
            data=torch.tensor(cell_types.astype(np.float32)),
            requires_grad=False)
        self.input_frequency = input_frequency
        self.iglusnfr_kernel = torch.nn.Parameter(
            data=torch.tensor(get_iGluSnFR_kernel(dt=1 / input_frequency)),
            requires_grad=False)
        self.GCaMP6f_kernel = torch.nn.Parameter(
            data=torch.tensor(get_GCaMP6f_singleAP_kernel(dt=1 / input_frequency)),
            requires_grad=False)
        self.kernel_0_size = len(np.arange(0, 0.3, 1 / self.input_frequency))
        self.kernel_1_size = self.iglusnfr_kernel.shape[-1]
        assert self.kernel_1_size == self.GCaMP6f_kernel.shape[-1]
        self.steady_state_steps = steady_state_steps * self.input_frequency
        self.padding = self.steady_state_steps
        self.padding += self.kernel_0_size - 1
        self.padding += self.kernel_1_size - 1
        self.noise_init_scale = noise_init_scale
        self.kernel_type = kernel_type

        ### Local Model ###
        self.initializer = WeightInitialization(
            shape=self.num_bc, noise_init_scale=self.noise_init_scale,
            random=random_init, seed=seed)
        
        ### PR - BC Weight ###
        self.log_pr_bc_weight = self.initializer.initialize(
            pr_bc_weight_init * np.ones((self.num_bc, self.num_pr)),
            random=random_init)

        # Feedforward Drive
        self.initial_affine = initial_affine
        if initial_affine:
            self.stimulus_bias = self.initializer.initialize(
                0, shape=1, scale='lin', requires_grad=fit_non_linearity,
                random=fit_non_linearity)
            self.stimulus_scale = self.initializer.initialize(
                1, shape=1, scale='lin', requires_grad=fit_non_linearity,
                random=fit_non_linearity)
        self.log_kernel_speed = self.initializer.initialize(
            kernel_speed_init, shape=(self.num_bc, num_pr),
            requires_grad=fit_linear_filter, random=fit_linear_filter)
        if self.kernel_type == 'nonparametric': #TODO compute_kernel needs index
            self.temporal_kernel = self.initializer.initialize(
                torch.zeros_like(self.compute_kernel()), shape=1, scale='lin',
                requires_grad=fit_linear_filter, random=fit_linear_filter)

        # Non-linearity
        self.sigmoid_offset = self.initializer.initialize(
            sigmoid_offset_init, scale='lin', requires_grad=fit_non_linearity,
            random=fit_non_linearity)
        self.log_sigmoid_slope = self.initializer.initialize(
            sigmoid_slope_init, requires_grad=fit_non_linearity,
            random=fit_non_linearity)

        # Release Machinery
        self.log_change_prob01 = self.initializer.initialize(
            change_prob01_init, requires_grad=fit_release_parameters,
            random=fit_release_parameters)
        self.log_change_prob12 = self.initializer.initialize(
            change_prob12_init, requires_grad=fit_release_parameters,
            random=fit_release_parameters)
        self.log_intermediate_pool_capacity = self.initializer.initialize(
            intermediate_pool_capacity_init,
            requires_grad=fit_release_parameters,
            random=fit_release_parameters)
        self.log_release_pool_capacity = self.initializer.initialize(
            release_pool_capacity_init, requires_grad=fit_release_parameters,
            random=fit_release_parameters)
        self.sig_ip_steady = self.initializer.initialize(
            ip_steady, requires_grad=fit_steady_state, scale='sig',
            random=fit_steady_state)
        self.sig_rrp_steady = self.initializer.initialize(
            rrp_steady, requires_grad=fit_steady_state, scale='sig',
            random=fit_steady_state)

        # Feedback
        self.num_acl = num_acl
        self.glycinergic_amacrine_cells = AmacrineCellFeedback(
            bc_acl_weight_init * np.ones((self.num_acl, self.num_bc)),
            acl_tau_rise_init,
            acl_tau_decay_init,
            acl_sigmoid_slope_init,
            acl_sigmoid_offset_init,
            noise_init_scale=self.noise_init_scale,
            input_frequency=self.input_frequency,
            random_init=random_init,
            seed=seed,
        )
        self.log_acl_bc_weight = self.initializer.initialize(
            acl_bc_weight_init * np.ones((self.num_bc, self.num_acl)),
            random=random_init)
        
        self.log_acl_acl_weight = self.initializer.initialize(
            acl_acl_weight_init * np.ones((self.num_acl, self.num_acl)),
            random=random_init)

    def forward(self, x):
        """
        :param x: raw light input dim: D x num_pr
        :return:
        """

        assert x.shape[-1] == self.num_pr, 'Wrong input dimension'
        pr_out = [] # Output of photoreceptors
        for pr in range(self.num_pr): # Loop through photoreceptors
            # linear filtering of light input with biphasic kernel
            pr_out.append(torch.exp(self.log_pr_bc_weight[:,pr])*self.linear_filter(x[:,pr], pr)) # DC
        x = sum(pr_out) # Sum across PRs for each BC

        # initialize release machinery and tracking
        lnr_state = self.init_state(x)

        # precompute log weights (compute before loop for speed up)
        sigmoid_slope = torch.exp(self.log_sigmoid_slope)
        # LNR
        acl_bc_weight = - torch.exp(self.log_acl_bc_weight)
        acl_acl_weight = - torch.exp(self.log_acl_acl_weight)
        bc_acl_weight = torch.exp(
            self.glycinergic_amacrine_cells.log_bc_ac_weight)
        acl_kernel = self.glycinergic_amacrine_cells.compute_ac_kernel()
        acl_sigmoid_slope = torch.exp(
            self.glycinergic_amacrine_cells.log_ac_sigmoid_slope)

        for i, xi in enumerate(x):  # over time steps
            # Local Chirp
            lnr_state['track_ac_input'].append(
                lnr_state['track_release'][i - 1])
            ac_in = torch.stack(  # only relevant history
                lnr_state['track_ac_input'][-acl_kernel.shape[1]:], dim=1)
            acl_out = self.glycinergic_amacrine_cells.forward(
                ac_in, bc_acl_weight, acl_kernel, acl_sigmoid_slope)
            acl_out = acl_out + torch.matmul(acl_acl_weight, acl_out)
            lnr_state['track_acl_output'].append(acl_out)
            local_feedback = torch.matmul(acl_bc_weight, acl_out)  # (nBC)
            lnr_state['track_feedback'][i] = local_feedback
            local_release_prob = precomputed_sigmoid(
                xi + local_feedback, self.sigmoid_offset, sigmoid_slope)
            lnr_state = self.release_step(i, local_release_prob, lnr_state)

        # final iGluSNFR filter and normalization
        y_lnr = self.final_transformation(lnr_state['track_release'])
        y_calcium = self.calcium_transformation(lnr_state['track_acl_output'])

        return y_lnr, lnr_state, y_calcium

    def linear_filter(self, x, i):
        """
        linear part of the LNR model
        :param x: raw light stimulus
        :return: with kernel convolved trace
        """
        x = F.pad(x, (self.padding, 0),
                  mode='constant', value=x[0])
        if self.initial_affine:
            x = x * self.stimulus_scale
            x = x + self.stimulus_bias
        x = x[None, None, :]  # BCD

        # feedforward drive
        if self.kernel_type == 'biphasic':
            kernel = self.compute_kernel(i)  # OID
        elif self.kernel_type == 'nonparametric':
            kernel = self.temporal_kernel

        return F.conv1d(x, kernel)[0].T  # DC

    def final_transformation(self, track_release):
        """
        final convolution with iGlusNFr kernel
        :param track_release:
        :return:
        """
        # convolve with iGluSNFR kernel
        # treat channel as batch dimension, because all get same iGlu kernel.
        # ToDo: change this when using batched inputs!
        x = track_release.T[:, None, :]
        x = F.conv1d(x, self.iglusnfr_kernel)
        x = x[:, 0, self.steady_state_steps:]  # CD

        return x
    
    def calcium_transformation(self, acl_output):
        """
        final convolution with GCaMP6f kernel
        :param track_release:
        :return:
        """
        # convolve with GCaMP6f kernel
        # treat channel as batch dimension, because all get same GCaMP6f kernel.
        # ToDo: change this when using batched inputs!
        x = torch.stack(acl_output).T[:, None, :]
        x = F.conv1d(x, self.GCaMP6f_kernel)
        x = x[:, 0, self.steady_state_steps:]  # CD

        return x

    def init_state(self, x):
        """
        initializing pool states and compute once what can be outside loop.
        :param release probability.
        :return:
        """
        # fractions steady state [0,1]
        ip_steady = torch.sigmoid(self.sig_ip_steady)
        rrp_steady = torch.sigmoid(self.sig_rrp_steady)

        # capacities minimum 1
        release_pool_capacity = torch.exp(
            self.smooth_clamp(self.log_release_pool_capacity, high=1e6, low=0))
        intermediate_pool_capacity = torch.exp(
            self.smooth_clamp(
                self.log_intermediate_pool_capacity, high=1e6, low=0))

        return dict(
            track_release_prob=torch.zeros_like(x),
            intermediate_pool_state=ip_steady * torch.exp(
                self.log_intermediate_pool_capacity),
            release_pool_state=rrp_steady * torch.exp(
                self.log_release_pool_capacity),
            track_intermediate_pool=torch.zeros_like(x),
            track_release_pool=torch.zeros_like(x),
            track_release=torch.zeros_like(x),
            change_prob12=torch.exp(self.log_change_prob12),
            release_pool_capacity=release_pool_capacity,
            change_prob01=torch.exp(self.log_change_prob01),
            intermediate_pool_capacity=intermediate_pool_capacity,
            track_feedback=torch.zeros_like(x),
            track_ac_input=[],
            track_acl_output=[],
        )

    def release_step(self, i, release_prob, state):
        """
        tracking and updating pools for one one release step
        :param i index of release steps
        :param release_prob
        :param state
        :return: updated state
        """
        # track current state of pools
        state['track_release_prob'][i] = release_prob
        state['track_intermediate_pool'][i] = state['intermediate_pool_state']
        state['track_release_pool'][i] = state['release_pool_state']
        state['track_release'][i] = release_prob * state['release_pool_state']

        # update release pool state (actual release) (1/2)
        released = state['release_pool_state'] - state['track_release'][i]

        # update release pool state (refilling) (2/2)
        state['release_pool_state'] = self.smooth_clamp(
            released + state['change_prob12'] * state[
                'intermediate_pool_state'], state['release_pool_capacity'])

        # get movement of vesicles from intermediate to release pool
        transfer = state['release_pool_state'] - released

        # update intermediate pool state (1/1)
        state['intermediate_pool_state'] = self.smooth_clamp(
            state['intermediate_pool_state'] + state['change_prob01'] -
            transfer, state['intermediate_pool_capacity'])

        return state

    def smooth_clamp(self, x, high, low=0):
        """
        :param low: lower bound of clamping
        :param high: upper bound of clamping
        :return: smooth clamping
        """
        #x = F.relu(x)  # hard clamp
        x = F.elu(x - low - 1) + low + 1
        x = F.elu(high - 1 - x) - high + 1
        return - x

    def compute_kernel(self, i):
        """
        PR biphasic kernel
        :return: normalized kernel with dt=1/scan_frequency
        (from https://github.com/berenslab/abc-ribbon/blob/master
        /standalone_model/ribbonv2.py)
        """
        # all variables in sec
        device = self.log_kernel_speed.device
        t = torch.arange(0.3, 0, step=-1 / self.input_frequency).to(device)
        kernel_speed = torch.exp(self.log_kernel_speed[:,i])
        t = t[None, :]
        kernel_speed = kernel_speed[:, None]
        tau_r = 0.05 * kernel_speed
        tau_d = 0.05 * kernel_speed
        phi = - np.pi * (0.2 / 1.4) * kernel_speed
        tau_phase = 100
        kernel = - (t / tau_r) ** 3 / (1 + t / tau_r) * torch.exp(
            - (t / tau_d) ** 2) * torch.cos(2 * np.pi * t / phi + tau_phase)
        # set kernel L2 norm = 1
        kernel = kernel / torch.norm(kernel, dim=1, keepdim=True)
        kernel = - kernel * self.cell_types[:,i][:, None]
        return kernel[:, None, :]  # out_channel x in_channel x duration
