from typing import Callable

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from easydict import EasyDict

from scipy.stats import qmc
from .ZO_utils import split_model, split_named_model, SplitedLayer, SplitedParam, default_create_fwd_hook_add_perturbation, default_create_fwd_hook_get_out_dimension
from .ZO_utils import build_rand_gen_fn
from .ZO_utils import default_wp_add_perturbation, default_wp_remove_perturbation, default_wp_gen_grad
# from .QMC_sampler import sphere_n, coord_basis, block_mask_generator, layer_mask_generator

DEBUG = False
# DEBUG = True

class ZO_Estim_MC(nn.Module):
    """
    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
    """
    
    def __init__(
        self,
        model: nn.Module,
        obj_fn_type: str,
        
        # For param perturb ZO. A list of SplitedParam. Specifies what Tensors should be optimized.
        splited_param_list: list = None,
        # For actv  perturb ZO. A list of SplitedLayer. Specifies what layers' activations should be perturbed.
        splited_layer_list: list = None,
        
        config = EasyDict({
            "sigma": 0.1,
            "n_sample": 20,
            "signsgd": False,
            "quantized": False,
            "estimate_method": 'forward',
            "sample_method": 'gaussian',
            "normalize_perturbation": False,
            "scale": None,
            "en_layerwise_perturbation": True,
            "en_partial_forward": True,
            "en_param_commit": False,
          }),
        ):
        super().__init__()

        self.model = model
        self.obj_fn_type = obj_fn_type

        self.splited_param_list = splited_param_list
        self.splited_layer_list = splited_layer_list
        
        self.splited_modules_list = split_model(model)
    
        self.sigma = config.sigma
        self.n_sample = config.n_sample
        self.signsgd = config.signsgd

        self.quantized = config.quantized
        self.estimate_method = config.estimate_method
        self.sample_method = config.sample_method
        self.normalize_perturbation = config.normalize_perturbation
        
        self.scale =config.scale
        
        self.en_layerwise_perturbation = config.en_layerwise_perturbation
        self.en_partial_forward = config.en_partial_forward
        self.en_param_commit = config.en_param_commit
        
        self.config = config

        self.device = next(model.parameters()).device
        
        self.rand_gen_fn = build_rand_gen_fn(self.sample_method, self.device)
        
        self.ZO_dimension = None
        
        self.forward_counter = 0
            
    ### Generate random vectors from a normal distribution
    def _generate_random_vector(self, shape, sample_method, device):
        
        dimension = np.prod(shape)

        if self.quantized == True:
            if sample_method == 'bernoulli':
                sample = torch.ones(shape, device=device) - 2*torch.bernoulli(0.5*torch.ones(shape, device=device))
            else:
                return NotImplementedError('Unlnown sample method', self.sample_method)
        else:
            if sample_method == 'uniform':
                sample = torch.randn(shape, device=device)
                sample = torch.nn.functional.normalize(sample, p=2, dim=0)
            elif sample_method == 'gaussian':
                sample = torch.randn(shape, device=device)
                # sample = torch.randn(shape, device=device) / dimension
            elif sample_method == 'bernoulli':
                ### Rademacher
                sample = torch.ones(shape, device=device) - 2*torch.bernoulli(0.5*torch.ones(shape, device=device))
            elif sample_method in ('sobol', 'halton'):
                if self.sampler == None:
                    raise ValueError('Need sampler input')
                else:
                    sample = torch.Tensor(self.sampler.random(1)).squeeze()
                    sample = 2*sample-torch.ones_like(sample)
                    sample = torch.nn.functional.normalize(sample, p=2, dim=0)
                    sample = sample.to(device)
            elif sample_method == 'sphere_n':
                sample = next(self.sampler)
                sample = sample.to(device)
            else:
                return NotImplementedError('Unlnown sample method', sample_method)
            
        return sample
    
    def get_single_param_ZO_gradient(self, splited_param, block_in, old_loss, sigma, estimate_method, sample_method):
        idx = splited_param.idx
        param = splited_param.param

        param_dim = param.numel()
        param_shape = param.shape
        param_vec = param.view(-1)
        old_param_vec = param_vec.clone().detach()

        param_grad_vec = torch.zeros_like(param_vec, device=self.device)
        
        if self.sample_method == 'coord_basis':
            for idx in range(param_dim):
                param_vec[idx].data.add_(sigma)
                if self.en_param_commit:
                    # if hasattr(splited_param.layer, 'commit_all'):
                    #     splited_param.layer.commit_all()
                    if hasattr(splited_param, 'commit_fn'):
                        splited_param.commit_fn()
                _, pos_loss = self.obj_fn()
                self.forward_counter += 1
                
                if estimate_method == 'forward':
                    param_vec[idx].data.copy_(old_param_vec[idx].data)
                    if self.en_param_commit:
                        # if hasattr(splited_param.layer, 'commit_all'):
                        #     splited_param.layer.commit_all()
                        if hasattr(splited_param, 'commit_fn'):
                            splited_param.commit_fn()
                    param_grad_vec[idx] = (pos_loss - old_loss) / sigma
                elif estimate_method == 'antithetic':
                    param_vec[idx].data.copy_(old_param_vec[idx].data)
                    param_vec[idx].data.sub_(sigma)
                    if self.en_param_commit:
                        # if hasattr(splited_param.layer, 'commit_all'):
                        #     splited_param.layer.commit_all()
                        if hasattr(splited_param, 'commit_fn'):
                            splited_param.commit_fn()
                    _, neg_loss = self.obj_fn()
                    self.forward_counter += 1
                    
                    param_vec[idx].data.copy_(old_param_vec[idx].data)
                    if self.en_param_commit:
                        # if hasattr(splited_param.layer, 'commit_all'):
                        #     splited_param.layer.commit_all()
                        if hasattr(splited_param, 'commit_fn'):
                            splited_param.commit_fn()
                    param_grad_vec[idx] = (pos_loss - neg_loss) / 2 / sigma
        else:
            splited_param.old_param = splited_param.param.clone().detach()
            
            for i in range(self.n_sample):
                ### Generate random perturbation with the same shape as the parameter
                if sample_method == 'coord_basis':
                    u = torch.zeros(param_dim, device=self.device)
                    u[i] = 1
                else:
                    u = self._generate_random_vector(param_vec.shape, sample_method, self.device)

                if self.normalize_perturbation:
                    p_sigma = sigma / torch.linalg.vector_norm(u)
                else:
                    p_sigma = sigma
                
                ### Add perturbation to the parameter
                # pos
                param_vec.add_(u * p_sigma)
                if self.en_param_commit:
                    # if hasattr(splited_param.layer, 'commit_all'):
                    #     splited_param.layer.commit_all()
                    if hasattr(splited_param, 'commit_fn'):
                        splited_param.commit_fn()
                if block_in is not None:
                    _, pos_loss = self.obj_fn(starting_idx=idx, input=block_in, return_loss_reduction='mean')
                else:
                    _, pos_loss = self.obj_fn()
                
                self.forward_counter += 1

                ### Estimate gradient
                if estimate_method == 'forward':
                    param_vec.data.copy_(old_param_vec)
                    if self.en_param_commit:
                      # if hasattr(splited_param.layer, 'commit_all'):
                      #     splited_param.layer.commit_all()
                      if hasattr(splited_param, 'commit_fn'):
                          splited_param.commit_fn()

                    param_grad_vec += (pos_loss - old_loss) / sigma * u
                elif estimate_method == 'antithetic':
                    param_vec.data.copy_(old_param_vec)
                    param_vec.sub_(u * p_sigma)
                    if self.en_param_commit:
                      # if hasattr(splited_param.layer, 'commit_all'):
                      #     splited_param.layer.commit_all()
                      if hasattr(splited_param, 'commit_fn'):
                          splited_param.commit_fn()
                        
                    if block_in is not None:
                        _, neg_loss = self.obj_fn(starting_idx=idx, input=block_in, return_loss_reduction='mean')
                    else:
                        _, neg_loss = self.obj_fn()
                    
                    self.forward_counter += 1
                    
                    param_vec.data.copy_(old_param_vec)
                    if self.en_param_commit:
                      # if hasattr(splited_param.layer, 'commit_all'):
                      #     splited_param.layer.commit_all()
                      if hasattr(splited_param, 'commit_fn'):
                          splited_param.commit_fn()

                    param_grad_vec += (pos_loss - neg_loss) / 2 / sigma * u
              
                param_grad_vec = param_grad_vec / self.n_sample
              
                if self.signsgd is True:
                    param_grad_vec = torch.sign(param_grad_vec)
                ### No scaling
                if self.scale is None:
                    pass
                elif self.scale == 'sqrt_dim':
                    param_grad_vec = param_grad_vec * math.sqrt(self.n_sample / (self.n_sample+param_dim-1))
                elif self.scale == 'dim':
                    param_grad_vec = param_grad_vec * (self.n_sample / (self.n_sample+param_dim-1))

        param_ZO_grad = param_grad_vec.view(param_shape)
        return param_ZO_grad
    
    def get_all_param_ZO_gradient(self, old_loss, sigma, estimate_method, sample_method):
        dimension = 0
        for splited_param in self.splited_param_list:
            dimension += splited_param.param.numel()
            splited_param.param.grad = torch.zeros_like(splited_param.param)
        
        if self.sample_method == 'coord_basis':
            for splited_param in self.splited_param_list:
                param_dim = splited_param.param.numel()
                param_vec = splited_param.param.view(-1)
                param_grad_vec = torch.zeros_like(param_vec)
                for idx in range(param_dim):
                    param_vec[idx].data.add_(sigma)
                    if self.en_param_commit:
                        if 'voltage' in splited_param.name:
                            splited_param.layer.commit_coordinate(idx)
                    _, pos_loss = self.obj_fn()
                    self.forward_counter += 1
                    
                    if estimate_method == 'forward':
                        param_vec[idx].data.sub_(sigma)
                        if self.en_param_commit:
                            if 'voltage' in splited_param.name:
                                splited_param.layer.commit_coordinate(idx)
                        param_grad_vec[idx] = (pos_loss - old_loss) / sigma
                    elif estimate_method == 'antithetic':
                        param_vec[idx].data.sub_(2*sigma)
                        if self.en_param_commit:
                            if 'voltage' in splited_param.name:
                                splited_param.layer.commit_coordinate(idx)
                        _, neg_loss = self.obj_fn()
                        self.forward_counter += 1
                        param_vec[idx].data.add_(sigma)
                        if self.en_param_commit:
                            if 'voltage' in splited_param.name:
                                splited_param.layer.commit_coordinate(idx)
                        param_grad_vec[idx] = (pos_loss - neg_loss) / 2 / sigma
                
                splited_param.param.grad = param_grad_vec.reshape(splited_param.param.shape)

        else:
            n_sample = self.n_sample
            for splited_param in self.splited_param_list:
                splited_param.old_param = splited_param.param.clone().detach()
            
            for i in range(n_sample):
                ### Generate random perturbation with the same shape as the parameter
                for splited_param in self.splited_param_list:
                    splited_param.u = self._generate_random_vector(splited_param.param.shape, sample_method, self.device)
                
                if self.normalize_perturbation:
                    p_sigma = sigma / torch.linalg.vector_norm(torch.cat([splited_param.u.view(-1) for splited_param in self.splited_param_list]))
                else:
                    p_sigma = sigma
                
                ### Add perturbation to the parameter
                # pos
                for splited_param in self.splited_param_list:
                    splited_param.param.add_(splited_param.u * p_sigma)
                    if self.en_param_commit:
                        if hasattr(splited_param.layer, 'commit_all'):
                            splited_param.layer.commit_all()  
                        # if 'voltage' in splited_param.name:
                        #     splited_param.layer.commit_all()
                    
                _, pos_loss = self.obj_fn()
                self.forward_counter += 1

                ### Estimate gradient
                if estimate_method == 'forward':
                    for splited_param in self.splited_param_list:
                        splited_param.param.copy_(splited_param.old_param)
                        if self.en_param_commit:
                            if hasattr(splited_param.layer, 'commit_all'):
                                splited_param.layer.commit_all()  
                            # if 'voltage' in splited_param.name:
                            #     splited_param.layer.commit_all()
                        
                        splited_param.param.grad += (pos_loss - old_loss) / sigma / n_sample * splited_param.u
                elif estimate_method == 'antithetic':
                    for splited_param in self.splited_param_list:
                        splited_param.param.copy_(splited_param.old_param)
                        splited_param.param.sub_(splited_param.u * p_sigma)

                        if self.en_param_commit:
                            if hasattr(splited_param.layer, 'commit_all'):
                                splited_param.layer.commit_all()  
                            # if 'voltage' in splited_param.name:
                            #     splited_param.layer.commit_all()
                                
                    _, neg_loss = self.obj_fn()
                    self.forward_counter += 1
                    
                    for splited_param in self.splited_param_list:
                        splited_param.param.copy_(splited_param.old_param)
                        if self.en_param_commit:
                            if hasattr(splited_param.layer, 'commit_all'):
                                splited_param.layer.commit_all()  
                            # if 'voltage' in splited_param.name:
                            #     splited_param.layer.commit_all()
                        
                        splited_param.param.grad += (pos_loss - neg_loss) / 2 / sigma / n_sample * splited_param.u
                        
            for splited_param in self.splited_param_list:
                if self.signsgd is True:
                    splited_param.param.grad = torch.sign(splited_param.param.grad)
                ### No scaling
                if self.scale is None:
                    pass
                elif self.scale == 'sqrt_dim':
                    splited_param.param.grad *= math.sqrt(n_sample / (n_sample+dimension-1))
                elif self.scale == 'dim':
                    splited_param.param.grad *= (n_sample / (n_sample+dimension-1))
        
        return None
    
    # def get_param_ZO_gradient(self, old_loss):
    #     if self.en_layerwise_perturbation:
            
    #         for splited_param in self.splited_param_list:
    #             if self.en_partial_forward:
    #                 block_in = self.obj_fn(ending_idx=splited_param.idx, return_loss_reduction='no_loss')
    #             else:
    #                 block_in = None
    #             ### TODO: could further specify sigma, estimate_method, sample_method for different params
    #             param_ZO_grad = self.get_single_param_ZO_gradient(splited_param, block_in, old_loss, self.sigma, self.estimate_method, self.sample_method)
                    
    #             splited_param.param.grad = param_ZO_grad

    #     else:
    #         self.get_all_param_ZO_gradient(old_loss, self.sigma, self.estimate_method, self.sample_method)

    ### memory-efficient implementation
    
    def get_layer_param_ZO_gradient(self, splited_layer, block_in, old_loss):   
      
        sigma = self.sigma     
        
        old_param_list = []
        seed_list = []
        loss_diff_list = []
        
        param_dim = 0
        
        if self.sample_method == 'coord_basis':
            raise NotImplementedError
        else:
            for param in splited_layer.layer.parameters():
                if param.requires_grad:
                    old_param_list.append(param.data.clone().detach())
                    param_dim += param.numel()
            
            wp_add_perturbation = getattr(splited_layer.layer, 'wp_add_perturbation', default_wp_add_perturbation)
            wp_remove_perturbation = getattr(splited_layer.layer, 'wp_remove_perturbation', default_wp_remove_perturbation)
            wp_gen_grad = getattr(splited_layer.layer, 'wp_gen_grad', default_wp_gen_grad)
            
            for i in range(self.n_sample):       
                ZO_random_seed = torch.randint(0, 100000, (1,))      
                seed_list.append(ZO_random_seed)   
                ### Add perturbation to the parameter
                # pos
                
                wp_add_perturbation(splited_layer.layer, sigma, self.rand_gen_fn, seed=ZO_random_seed)

                if block_in is not None:
                    _, pos_loss = self.obj_fn(starting_idx=splited_layer.idx, input=block_in, return_loss_reduction='mean')
                else:
                    _, pos_loss = self.obj_fn()
                
                self.forward_counter += 1

                ### Estimate gradient
                if self.estimate_method == 'forward':
                    wp_remove_perturbation(splited_layer.layer, old_param_list)
                    loss_diff = (pos_loss - old_loss) / self.n_sample / sigma

                elif self.estimate_method == 'antithetic':
                    wp_add_perturbation(splited_layer.layer, -2*sigma, self.rand_gen_fn, seed=ZO_random_seed)
                        
                    if block_in is not None:
                        _, neg_loss = self.obj_fn(starting_idx=splited_layer.idx, input=block_in, return_loss_reduction='mean')
                    else:
                        _, neg_loss = self.obj_fn()
                    
                    self.forward_counter += 1
                    
                    wp_remove_perturbation(splited_layer.layer, old_param_list)
                    loss_diff = (pos_loss - neg_loss) / 2.0 / self.n_sample / sigma
                
                ### No scaling
                if self.scale is None:
                    pass
                elif self.scale == 'sqrt_dim':
                    loss_diff = loss_diff * math.sqrt(self.n_sample / (self.n_sample+param_dim-1))
                elif self.scale == 'dim':
                    loss_diff = loss_diff * (self.n_sample / (self.n_sample+param_dim-1))
                
                loss_diff_list.append(loss_diff)
              
            # end if n_samle
            
            # if self.signsgd is True:
            #     param_grad_vec = torch.sign(param_grad_vec)
            
            wp_gen_grad(splited_layer.layer, loss_diff_list, seed_list, self.rand_gen_fn)
            
        return None    
    
    def get_actv_ZO_gradient(self, splited_layer, block_in, old_loss_vec):

        ### Generate random perturbation with the same shape as the parameter
        ### Add perturbation to the parameter
        ### Estimate gradient

        # post_actv_shape = splited_layer.layer.output_shape
        # if 'ATIS' in self.obj_fn_type:
        #     batch_sz = post_actv_shape[1]
        # else:
        #     batch_sz = post_actv_shape[0]
        # ZO_grad = torch.zeros(post_actv_shape, device=self.device)
        
        # mask = torch.ones(post_actv_shape, device=self.device)

        if self.sample_method == 'coord_basis':
            raise NotImplementedError
            # actv_dim = np.prod(post_actv_shape[1:])
            # feature_shape = post_actv_shape[1:]
            # n_sample = actv_dim
        else:
            n_sample = self.n_sample
        
        for i in range(n_sample):
            
            ### Generate random perturbation with the same shape as the parameter
            if self.sample_method == 'coord_basis':   
                raise NotImplementedError                 
                # u = torch.zeros(actv_dim, device=self.device)
                # u[i] = 1
                # u.reshape(feature_shape)
                # u = mask * torch.tile(u, (batch_sz, 1))
                    
            else:
                # if hasattr(self, 'sync_batch_perturb') and self.sync_batch_perturb:
                #     u = mask * torch.tile(self._generate_random_vector(feature_shape, self.sample_method, self.device).unsqueeze(0), (batch_sz, 1))
                # else:
                #     u = mask * self._generate_random_vector(post_actv_shape, self.sample_method, self.device)    
                pass
              
            ### Add perturbation to the parameter
            ZO_random_seed = torch.randint(0, 100000, (1,))
            create_fwd_hook_add_perturbation = getattr(splited_layer.layer, 'create_fwd_hook_add_perturbation', default_create_fwd_hook_add_perturbation)
            # fwd_hook_add_perturbation = create_fwd_hook_add_perturbation(u*self.sigma)
            fwd_hook_add_perturbation = create_fwd_hook_add_perturbation(ZO_random_seed, self.sigma, self.rand_gen_fn)
            fwd_hook_handle = splited_layer.layer.register_forward_hook(fwd_hook_add_perturbation)

            if self.en_partial_forward:
                _, pos_loss = self.obj_fn(starting_idx=splited_layer.idx, input=block_in, return_loss_reduction='none')
            else:
                _, pos_loss = self.obj_fn(return_loss_reduction='none')
                
            fwd_hook_handle.remove()
            self.forward_counter += 1

            ### Estimate gradient
            if self.estimate_method == 'forward':
                loss_diff = pos_loss - old_loss_vec

            elif self.estimate_method == 'antithetic':
                ### Add perturbation to the parameter
                # splited_layer.layer.en_perturb_forward( - u * self.sigma)
                create_fwd_hook_add_perturbation = getattr(splited_layer.layer, 'create_fwd_hook_add_perturbation', default_create_fwd_hook_add_perturbation)
                # fwd_hook_add_perturbation = create_fwd_hook_add_perturbation(- u * self.sigma)
                fwd_hook_add_perturbation = create_fwd_hook_add_perturbation(ZO_random_seed, -self.sigma, self.rand_gen_fn)
                fwd_hook_handle = splited_layer.layer.register_forward_hook(fwd_hook_add_perturbation)

                if self.en_partial_forward:
                    _, neg_loss = self.obj_fn(starting_idx=splited_layer.idx, input=block_in, return_loss_reduction='none')
                else:
                    _, neg_loss = self.obj_fn(return_loss_reduction='none')

                fwd_hook_handle.remove()
                self.forward_counter += 1
                
                loss_diff = (pos_loss - neg_loss) / 2.0
                
            u = splited_layer.layer.perturbation

            ### init
            if i == 0:
                ZO_grad = torch.zeros(u.shape, device=self.device)
                batch_sz = len(loss_diff)
                # if 'ATIS' in self.obj_fn_type:
                #     batch_sz = u.shape[1]
                # else:
                #     batch_sz = u.shape[0]
                actv_dim = np.prod(u.shape) / batch_sz
            
            if batch_sz == 1:
                ZO_grad += loss_diff / self.sigma * u
            else:
                # if 'ATIS' in self.obj_fn_type or 'MNLI' in self.obj_fn_type:
                if any([x in self.obj_fn_type for x in ['ATIS', 'MNLI', 'LM']]):
                    if len(loss_diff) == u.shape[1]:
                    # if type(splited_layer.layer) in self.model.ZO_trainable_layers_dict.values():
                        ZO_grad += torch.einsum('i,si...->si...', (loss_diff / self.sigma, u)) 
                    else:
                        ZO_grad += torch.einsum('i,i...->i...', (loss_diff / self.sigma, u))
                else:
                    ZO_grad += torch.einsum('i,i...->i...', (loss_diff / self.sigma, u))

        if self.sample_method == 'coord_basis':
            ZO_grad = (ZO_grad / batch_sz)
        else:
            ZO_grad = (ZO_grad / self.n_sample / batch_sz)
            
        ### gradient scale
        if self.signsgd is True:
            ZO_grad = torch.sign(ZO_grad)
        ### No scaling
        if self.scale is None:
            pass
        elif self.scale == 'sqrt_dim':
            ZO_grad *= math.sqrt(batch_sz*n_sample / (batch_sz*n_sample+actv_dim-1))
        elif self.scale == 'dim':
            ZO_grad *= (batch_sz*n_sample / (batch_sz*n_sample+actv_dim-1))

        ### Apply estimated gradient
        splited_layer.layer.ZO_grad_output = ZO_grad
        # if type(splited_layer.layer) == nn.Linear:
        #     splited_layer.layer.weight.grad = torch.matmul(ZO_grad.T, splited_layer.layer.in_value)
        #     splited_layer.layer.bias.grad = torch.sum(ZO_grad, dim=0)
        # else:
        #     splited_layer.layer.local_backward(ZO_grad, block_in) 
    
    def get_all_actv_ZO_gradient(self, block_in, old_loss_vec):

        ### Generate random perturbation with the same shape as the parameter
        ### Add perturbation to the parameter
        ### Estimate gradient
        
        if 'LM' in self.obj_fn_type:
            batch_sz = torch.numel(old_loss_vec)
            actv_dim = self.ZO_dimension / old_loss_vec.size(1)
        else:
            batch_sz = len(old_loss_vec)
            actv_dim = self.ZO_dimension
            
        if self.sample_method == 'coord_basis':
            raise NotImplementedError
            # actv_dim = np.prod(post_actv_shape[1:])
            # feature_shape = post_actv_shape[1:]
            # n_sample = actv_dim
        else:
            n_sample = self.n_sample 
        
        ### scaling factor
        if self.sample_method == 'coord_basis':
            scaling_factor = (1 / batch_sz)
        else:
            scaling_factor = (1 / self.n_sample / batch_sz)

        ### No scaling
        if self.scale is None:
            pass
        elif self.scale == 'sqrt_dim':
            scaling_factor *= math.sqrt(batch_sz*n_sample / (batch_sz*n_sample+actv_dim-1))
        elif self.scale == 'dim':
            scaling_factor *= (batch_sz*n_sample / (batch_sz*n_sample+actv_dim-1))     
        
        ### Perturbed forwards
        for i in range(n_sample):
            fwd_hook_handle_list = []
            ### Add perturbation to all activations
            for splited_layer in self.splited_layer_list:
                splited_layer.ZO_random_seed = torch.randint(0, 100000, (1,))
                create_fwd_hook_add_perturbation = getattr(splited_layer.layer, 'create_fwd_hook_add_perturbation', default_create_fwd_hook_add_perturbation)
                # fwd_hook_add_perturbation = create_fwd_hook_add_perturbation(u*self.sigma)
                fwd_hook_add_perturbation = create_fwd_hook_add_perturbation(splited_layer.ZO_random_seed, self.sigma, self.rand_gen_fn)
                fwd_hook_handle = splited_layer.layer.register_forward_hook(fwd_hook_add_perturbation)
                fwd_hook_handle_list.append(fwd_hook_handle)

            if self.en_partial_forward:
                _, pos_loss = self.obj_fn(starting_idx=splited_layer.idx, input=block_in, return_loss_reduction='none')
            else:
                _, pos_loss = self.obj_fn(return_loss_reduction='none')
            
            for fwd_hook_handle in fwd_hook_handle_list:    
                fwd_hook_handle.remove()
                
            self.forward_counter += 1

            if self.estimate_method == 'forward':
                loss_diff = pos_loss - old_loss_vec

            elif self.estimate_method == 'antithetic':
                fwd_hook_handle_list = []
                ### Add perturbation to the parameter
                for splited_layer in self.splited_layer_list:
                    # splited_layer.layer.en_perturb_forward( - u * self.sigma)
                    create_fwd_hook_add_perturbation = getattr(splited_layer.layer, 'create_fwd_hook_add_perturbation', default_create_fwd_hook_add_perturbation)
                    # fwd_hook_add_perturbation = create_fwd_hook_add_perturbation(- u * self.sigma)
                    fwd_hook_add_perturbation = create_fwd_hook_add_perturbation(splited_layer.ZO_random_seed, -self.sigma, self.rand_gen_fn)
                    fwd_hook_handle = splited_layer.layer.register_forward_hook(fwd_hook_add_perturbation)
                    fwd_hook_handle_list.append(fwd_hook_handle)

                if self.en_partial_forward:
                    _, neg_loss = self.obj_fn(starting_idx=splited_layer.idx, input=block_in, return_loss_reduction='none')
                else:
                    _, neg_loss = self.obj_fn(return_loss_reduction='none')

                for fwd_hook_handle in fwd_hook_handle_list:    
                    fwd_hook_handle.remove()
                
                self.forward_counter += 1
                
                loss_diff = (pos_loss - neg_loss) / 2.0
                    
            ### estimate gradient
            for splited_layer in self.splited_layer_list:
                u = splited_layer.layer.perturbation

                ### init
                if i == 0:
                    splited_layer.layer.ZO_grad_output = torch.zeros(u.shape, device=self.device, dtype=u.dtype)
                    
                ### accumulate
                splited_layer.layer.ZO_grad_output += torch.einsum('bs,bsd->bsd', (scaling_factor * loss_diff / self.sigma, u)).to(u.dtype) 
        
        if self.signsgd is True:
            for splited_layer in self.splited_layer_list:
                splited_layer.layer.ZO_grad_output = torch.sign(splited_layer.layer.ZO_grad_output)

        ### Apply estimated gradient
        # splited_layer.layer.ZO_grad_output = ZO_grad
        # if type(splited_layer.layer) == nn.Linear:
        #     splited_layer.layer.weight.grad = torch.matmul(ZO_grad.T, splited_layer.layer.in_value)
        #     splited_layer.layer.bias.grad = torch.sum(ZO_grad, dim=0)
        # else:
        #     splited_layer.layer.local_backward(ZO_grad, block_in) 
    
    def get_pseudo_actv_ZO_gradient(self):

        ### Generate random perturbation with the same shape as the parameter
        ### Add perturbation to the parameter
        ### Estimate gradient
        
        if self.sample_method == 'coord_basis':
            raise NotImplementedError
      
        actv_dim = self.ZO_dimension
        n_sample = self.n_sample
        detach_idx = getattr(self.config, 'pzo_detach_idx', -1)
        
        # (bz, 1, 1, seq_len) -> (seq_len, bz, 1)
        mask = self.obj_fn.get_mask()
        
        # mask = None
        
        if self.estimate_method == 'forward':
            output, output_grad, outputs, loss = self.obj_fn(return_loss_reduction='pzo', detach_idx=detach_idx)
            self.model.zero_grad()
            self.forward_counter += 1
        else:
            pass
        
        with torch.no_grad():
            ### Perturbed forwards
            for i in range(n_sample):
                fwd_hook_handle_list = []
                ### Add perturbation to all activations
                for splited_layer in self.splited_layer_list:
                    splited_layer.ZO_random_seed = torch.randint(0, 100000, (1,))
                    create_fwd_hook_add_perturbation = getattr(splited_layer.layer, 'create_fwd_hook_add_perturbation', default_create_fwd_hook_add_perturbation)
                    # fwd_hook_add_perturbation = create_fwd_hook_add_perturbation(u*self.sigma)
                    fwd_hook_add_perturbation = create_fwd_hook_add_perturbation(splited_layer.ZO_random_seed, self.sigma, self.rand_gen_fn, mask)
                    fwd_hook_handle = splited_layer.layer.register_forward_hook(fwd_hook_add_perturbation)
                    fwd_hook_handle_list.append(fwd_hook_handle)

                if self.estimate_method == 'one_point':
                    pos_output, pos_output_grad, pos_outputs, pos_loss = self.obj_fn(return_loss_reduction='pzo', detach_idx=detach_idx)
                else:
                    pos_output = self.obj_fn(return_loss_reduction='pzo_nograd', detach_idx=detach_idx)
                self.model.zero_grad()
                
                for fwd_hook_handle in fwd_hook_handle_list:    
                    fwd_hook_handle.remove()
                    
                self.forward_counter += 1

                if self.estimate_method == 'antithetic':
                    fwd_hook_handle_list = []
                    ### Add perturbation to the parameter
                    for splited_layer in self.splited_layer_list:
                        # splited_layer.layer.en_perturb_forward( - u * self.sigma)
                        create_fwd_hook_add_perturbation = getattr(splited_layer.layer, 'create_fwd_hook_add_perturbation', default_create_fwd_hook_add_perturbation)
                        # fwd_hook_add_perturbation = create_fwd_hook_add_perturbation(- u * self.sigma)
                        fwd_hook_add_perturbation = create_fwd_hook_add_perturbation(splited_layer.ZO_random_seed, -self.sigma, self.rand_gen_fn, mask)
                        fwd_hook_handle = splited_layer.layer.register_forward_hook(fwd_hook_add_perturbation)
                        fwd_hook_handle_list.append(fwd_hook_handle)

                    neg_output = self.obj_fn(return_loss_reduction='pzo_nograd', detach_idx=detach_idx)
                    self.model.zero_grad()

                    for fwd_hook_handle in fwd_hook_handle_list:    
                        fwd_hook_handle.remove()
                    
                    self.forward_counter += 1
                    
                if self.estimate_method == 'one_point':
                    tilde_o = pos_output
                    output_grad = pos_output_grad
                    outputs = pos_outputs
                    loss = pos_loss
                elif self.estimate_method == 'forward':
                    tilde_o = pos_output - output
                elif self.estimate_method == 'antithetic':
                    tilde_o = (pos_output - neg_output) / 2.0
                
                batch_sz = tilde_o.shape[1]
                seq_len = tilde_o.shape[0]
                
                ### scaling factor
                scaling_factor = (1 / n_sample / batch_sz)

                ### No scaling
                if self.scale is None:
                    pass
                elif self.scale == 'sqrt_dim':
                    scaling_factor *= math.sqrt(batch_sz*n_sample / (batch_sz*n_sample+actv_dim-1))
                elif self.scale == 'dim':
                    scaling_factor *= (batch_sz*n_sample / (batch_sz*n_sample+actv_dim-1))    
                        
                ### estimate gradient
                for splited_layer in self.splited_layer_list:
                    u = splited_layer.layer.perturbation
                    
                    ### merge seq_len and batch_sz  
                    feedback_matrix = torch.einsum('...d,...m->dm', (u, tilde_o))
                    
                    ### each token to [CLS] token
                    # feedback_matrix = torch.einsum('sbd,bm->sdm', (u, tilde_o[0]))
                    
                    ### only merge batch_sz
                    # u = u.permute(1, 0, 2).reshape(batch_sz, -1)
                    # tilde_o = tilde_o.permute(1, 0, 2).reshape(batch_sz, -1)
                    # feedback_matrix = torch.einsum('bd,bm->bdm', (u, tilde_o))
                    
                    if float(self.config.pzo_momentum) > 0: 
                        if hasattr(splited_layer.layer, 'feedback_matrix') is False:
                            # splited_layer.layer.feedback_matrix = feedback_matrix
                            splited_layer.layer.register_buffer('feedback_matrix', feedback_matrix)
                        else:
                            splited_layer.layer.feedback_matrix = self.config.pzo_momentum * splited_layer.layer.feedback_matrix + (1 - self.config.pzo_momentum) * feedback_matrix
                            feedback_matrix = splited_layer.layer.feedback_matrix

                    ### init
                    if i == 0:
                        splited_layer.layer.ZO_grad_output = torch.zeros(u.shape, device=self.device)

                    ### merge seq_len and batch_sz  
                    splited_layer.layer.ZO_grad_output += scaling_factor * torch.einsum('dm,...m->...d', (feedback_matrix, output_grad))
                    
                    ### each token to [CLS] token
                    # splited_layer.layer.ZO_grad_output += scaling_factor * torch.einsum('sdm,bm->sbd', (feedback_matrix, output_grad[0]))
                    
                    ### only merge batch_sz
                    # output_grad = output_grad.permute(1, 0, 2).reshape(batch_sz, -1)
                    # splited_layer.layer.ZO_grad_output += scaling_factor * torch.einsum('bdm,bm->bd', (feedback_matrix, output_grad)).reshape(batch_sz, seq_len, -1).permute(1, 0, 2)

            ### Apply estimated gradient
            # splited_layer.layer.ZO_grad_output = ZO_grad
            # if type(splited_layer.layer) == nn.Linear:
            #     splited_layer.layer.weight.grad = torch.matmul(ZO_grad.T, splited_layer.layer.in_value)
            #     splited_layer.layer.bias.grad = torch.sum(ZO_grad, dim=0)
            # else:
            #     splited_layer.layer.local_backward(ZO_grad, block_in) 
        
        return outputs, loss

    def update_obj_fn(self, obj_fn):
        self.obj_fn = obj_fn
        
        if self.ZO_dimension is None:
            self.ZO_dimension = self.get_ZO_dimension()
    
    def get_forward_cnt(self):
        return self.forward_counter
    
    def get_ZO_dimension(self):
        ZO_dimension = 0
        if self.splited_param_list is not None:
            for splited_param in self.splited_param_list:
                print(f'{splited_param.name} param_dimension={splited_param.param.numel()}')
                ZO_dimension += splited_param.param.numel()
                # param.grad = torch.zeros_like(param)

        if self.splited_layer_list is not None:
            ### get activation dimension
            fwd_hook_handle_list = []
            for splited_layer in self.splited_layer_list:
                create_fwd_hook_get_out_dimension = getattr(splited_layer.layer, 'create_fwd_hook_get_out_dimension', default_create_fwd_hook_get_out_dimension)
                fwd_hook_get_out_dimension = create_fwd_hook_get_out_dimension()
                fwd_hook_handle_list.append(splited_layer.layer.register_forward_hook(fwd_hook_get_out_dimension))
                    
            _, old_loss = self.obj_fn(return_loss_reduction='none')
            
            for splited_layer in self.splited_layer_list:
                if splited_layer.mode == 'actv':
                    print(f'{splited_layer.name} out_dimension={splited_layer.layer.out_dimension}')
                    ZO_dimension += splited_layer.layer.out_dimension

                elif splited_layer.mode == 'param':
                    param_dim = 0
                    for param in splited_layer.layer.parameters():
                        if param.requires_grad:
                            param_dim += param.numel()
                    print(f'{splited_layer.name} param_dimension={param_dim}')
                
                    ZO_dimension += param_dim
                    
            for fwd_hook_handle in fwd_hook_handle_list:
                fwd_hook_handle.remove()  
        
        ZO_dimension = int(ZO_dimension)
        print('ZO_dimension=', ZO_dimension)
        return ZO_dimension
    
    def estimate_grad(self):
        ### modelwise pseudo-ZO node perturbation
        if getattr(self.config, 'en_pseudo_ZO', False):
            outputs, loss = self.get_pseudo_actv_ZO_gradient()
        else:
            with torch.no_grad():
                if self.splited_layer_list is not None:
                
                    outputs, old_loss_vec = self.obj_fn(return_loss_reduction='none')
                    loss = old_loss_vec.mean()
                    
                    ### layerwise node/weight perturbation
                    if self.en_layerwise_perturbation:
                        for splited_layer in self.splited_layer_list:
                            if self.en_partial_forward:
                                block_in = self.obj_fn(ending_idx=splited_layer.idx, return_loss_reduction='no_loss')
                            else:
                                block_in = None
                                
                            if splited_layer.mode == 'actv':
                                self.get_actv_ZO_gradient(splited_layer, block_in, old_loss_vec)
                            elif splited_layer.mode == 'param':
                                self.get_layer_param_ZO_gradient(splited_layer, block_in, loss)
                    ### modelwise node perturbation
                    else:
                        
                        if self.en_partial_forward:
                            block_in = self.obj_fn(ending_idx=self.splited_layer_list[0].idx, return_loss_reduction='no_loss')
                        else:
                            block_in = None
                            
                        self.get_all_actv_ZO_gradient(block_in, old_loss_vec)
            
                if self.splited_param_list is not None:
                    outputs, loss = self.obj_fn()
                    
                    ### layerwise weight perturbation
                    if self.en_layerwise_perturbation:
                    
                        for splited_param in self.splited_param_list:
                            if self.en_partial_forward:
                                block_in = self.obj_fn(ending_idx=splited_param.idx, return_loss_reduction='no_loss')
                            else:
                                block_in = None
                            ### TODO: could further specify sigma, estimate_method, sample_method for different params
                            param_ZO_grad = self.get_single_param_ZO_gradient(splited_param, block_in, loss, self.sigma, self.estimate_method, self.sample_method)
                                
                            splited_param.param.grad = param_ZO_grad

                    ### modelwise weight perturbation
                    else:
                        self.get_all_param_ZO_gradient(loss, self.sigma, self.estimate_method, self.sample_method)
        
        return outputs, loss