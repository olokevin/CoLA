import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitedLayer(nn.Module):
    def __init__(self, idx, name, layer, mode='actv'):
        super().__init__()
        self.idx = idx
        self.name = name
        self.layer = layer
        
        self.mode = mode # 'actv' or 'param'

class SplitedParam(nn.Module):
    def __init__(self, idx, name, layer, param, commit_fn=None):
        super().__init__()
        self.idx = idx
        self.name = name
        assert isinstance(param, torch.Tensor)
        self.layer = layer
        self.param = param
        if commit_fn is not None:
            assert callable(commit_fn)
            self.commit_fn = commit_fn

def fwd_hook_save_value(module, input, output):
    module.in_value = input[0].detach().clone()
    module.out_value = output.detach().clone()
    
    return None

def bwd_hook_save_grad(module, grad_input, grad_output):
    if grad_input[0] is not None:
        module.in_grad = grad_input[0].detach().clone()
    module.out_grad = grad_output[0].detach().clone()
    
    return None

def default_create_fwd_hook_get_out_dimension():
    def fwd_hook(module, input, output):
        # input is a tuple
        module.output_shape = output.shape
        module.out_dimension = output.numel() / output.shape[0]
    return fwd_hook

def default_wp_add_perturbation(module, sigma, rand_gen_fn, seed=None):
    if seed is not None:
        state = torch.get_rng_state()
        torch.manual_seed(seed)
    for param in module.parameters():
        if param.requires_grad:
            perturbation = rand_gen_fn(param.shape)
            param.data += sigma * perturbation
    
    if seed is not None:
        torch.set_rng_state(state)

# def default_wp_remove_perturbation(module, sigma, rand_gen_fn, seed=None):
#     if seed is not None:
#         state = torch.get_rng_state()
#         torch.manual_seed(seed)
#     for param in module.parameters():
#         if param.requires_grad:
#             perturbation = rand_gen_fn(param.shape)
#             param.data -= sigma * perturbation
    
#     if seed is not None:
#         torch.set_rng_state(state)

def default_wp_remove_perturbation(module, old_param_list):
    for idx, param in enumerate(module.parameters()):
        if param.requires_grad:
            param.data.copy_(old_param_list[idx])
            
def default_wp_gen_grad(module, loss_diff_list, seed_list, rand_gen_fn):
    state = torch.get_rng_state()
    
    for idx in range(len(seed_list)):
        seed = seed_list[idx]
        loss_diff = loss_diff_list[idx]
        
        torch.manual_seed(seed)

        for param in module.parameters():
            if param.requires_grad:
                perturbation = rand_gen_fn(param.shape)
                if param.grad is None:
                    param.grad = loss_diff * perturbation
                else:
                    param.grad += loss_diff * perturbation
    
    torch.set_rng_state(state)


# def default_create_fwd_hook_add_perturbation(perturbation):
#     def fwd_hook(module, input, output):
#         # input is a tuple
#         module.in_value = input[0].detach().clone()
#         # output is a tensor. inplace & return modifiled output both owrk
#         # output += perturbation
#         return output + perturbation
#     return fwd_hook

"""
mask: of the same shape as output / perturbation
"""
def default_create_fwd_hook_add_perturbation(seed, sigma, rand_gen_fn, mask=None):
    def fwd_hook(module, input, output):
        # input is a tuple
        # module.in_value = input[0].detach().clone()
        # output is a tensor. inplace & return modifiled output both work
        state = torch.get_rng_state()
        
        if seed is not None:
            torch.manual_seed(seed)
        perturbation = rand_gen_fn(output.shape).to(output.dtype)
        torch.set_rng_state(state)
        
        if mask is not None:
            perturbation *= mask
            
        module.perturbation = perturbation
        
        # output += sigma * perturbation
        return output + sigma * perturbation
    return fwd_hook
    
def default_create_bwd_pre_hook_ZO_grad(ZO_grad_output, debug=False):
    def bwd_pre_hook(module, grad_output):
        if debug:
            ### full gradient
            print(f'{F.cosine_similarity(grad_output[0].reshape(-1), ZO_grad_output.reshape(-1), dim=0)}')
            ### [CLS] token
            # print(f'{F.cosine_similarity(grad_output[0][0].reshape(-1), ZO_grad_output[0].reshape(-1), dim=0)}')
            
            # print(f'{torch.linalg.norm(ZO_grad_output.reshape(-1)) / torch.linalg.norm(grad_output[0].reshape(-1))}')
        return (ZO_grad_output,)
    return bwd_pre_hook

def bwd_pre_hook_only_CLS(module, grad_output):
    mask = torch.zeros_like(grad_output[0])
    mask[0] = 1
    return (grad_output[0] * mask,)

def default_create_bwd_pre_hook_add_noise(target_cosine_similarity, max_iterations=100, debug=False):
    def bwd_pre_hook(module, grad_output):
        # noise_scale = init_noise_scale
        original_grad = grad_output[0].detach().clone()
        dim = original_grad.numel()
        noise_scale = torch.linalg.norm(original_grad.reshape(-1)) * math.sqrt((1/target_cosine_similarity**2 - 1) / dim)
        for _ in range(max_iterations):
            noise = torch.randn_like(original_grad) * noise_scale
            noise[original_grad == 0] = 0
            noisy_grad = original_grad + noise
            cosine_sim = F.cosine_similarity(original_grad.reshape(-1), noisy_grad.reshape(-1), dim=0)
            if cosine_sim < target_cosine_similarity:
                noisy_grad *= torch.linalg.norm(original_grad.reshape(-1)) / torch.linalg.norm(noisy_grad.reshape(-1))
                break
            noise_scale *= 1.1  # Gradually increase noise scale if target not met
        
        if debug:
            print(cosine_sim)
        return (noisy_grad,)
    return bwd_pre_hook

def recursive_getattr(obj, attr):
    attrs = attr.split('.')
    for a in attrs:
        obj = getattr(obj, a)
    return obj
  
def split_model(model, ZO_iterable_block_name=None):
    modules = []
    # full model split
    if ZO_iterable_block_name is None:
        for m in model.children():
            if isinstance(m, (torch.nn.Sequential, torch.nn.ModuleList)):
                modules += split_model(m)
            else:
                modules.append(m)
    # only split iterable block
    else:
        # iterable_block = getattr(model, ZO_iterable_block_name)
        iterable_block = recursive_getattr(model, ZO_iterable_block_name)
        assert isinstance(iterable_block, (torch.nn.Sequential, torch.nn.ModuleList))
        for m in iterable_block.children():
            modules.append(m)
    return modules

def split_named_model(model, parent_name=''):
    named_modules = {}
    for name, module in model.named_children():
    # for name, module in model.named_modules():    # Error: non-stop recursion
        if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
            named_modules.update(split_named_model(module, parent_name + name + '.'))
        # elif hasattr(module, 'conv') and isinstance(module.conv, torch.nn.Sequential):
        #     named_modules.update(split_named_model(module.conv, parent_name + name + '.conv.'))
        else:
            named_modules[parent_name + name] = module
    return named_modules

def build_rand_gen_fn(sample_method, device, sampler=None):
    def _rand_gen_fn(shape):
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
            if sampler == None:
                raise ValueError('Need sampler input')
            else:
                sample = torch.Tensor(sampler.random(1)).squeeze()
                sample = 2*sample-torch.ones_like(sample)
                sample = torch.nn.functional.normalize(sample, p=2, dim=0)
                sample = sample.to(device)
        elif sample_method == 'sphere_n':
            sample = next(sampler)
            sample = sample.to(device)
        else:
            return NotImplementedError('Unlnown sample method', sample_method)
        
        return sample
    return _rand_gen_fn