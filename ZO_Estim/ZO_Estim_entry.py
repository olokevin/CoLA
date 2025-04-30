import torch
import torch.nn as nn

from .ZO_utils import SplitedLayer, SplitedParam, split_model
from .ZO_Estim_MC import ZO_Estim_MC

### Model specific utils ###
# from .ZO_fwd_utils import trainable_layers_dict, get_iterable_block_name, ZO_pre_block_forward, ZO_post_block_forward
from cola.cola_layer import ColaLayer

def create_trainable_layers_list(layer_list, trainable_layers_dict):
    if isinstance(layer_list, str):
        return trainable_layers_dict[layer_list]
    elif isinstance(layer_list, list):
        opt_layers = []
        for layer_str in layer_list:
            opt_layers.append(trainable_layers_dict[layer_str])
        return tuple(opt_layers)
    else:
        raise (ValueError("opt_layers_strs should either be a string of a list of strings"))

def fwd_hook_get_output_shape(module, input, output):
    module.output_shape = output.shape

def build_ZO_Estim(config, model):
    if config.name == 'ZO_Estim_MC':
        ### split model
        ZO_iterable_block_name = getattr(model, 'ZO_iterable_block_name', None)
        split_modules_list = split_model(model, ZO_iterable_block_name)

        splited_param_list = None
        splited_layer_list = None

        ### Param perturb 
        if config.param_perturb_block_idx_list is not None:
            splited_param_list = []
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    splited_param_list.append(SplitedParam(idx=-1, name=param_name, layer=None, param=param)) 
        
        ### Actv perturb 
        if config.actv_perturb_block_idx_list is not None:
            splited_layer_list = []
            
            for layer_name, layer in model.named_modules():
                # if type(layer) in (ColaLayer, nn.Embedding):
                if type(layer) in (ColaLayer,):
                    if all(param.requires_grad for param in layer.parameters()):
                        splited_layer_list.append(SplitedLayer(idx=-1, name=layer_name, layer=layer))
                            
        if splited_param_list is not None:
            for splited_param in splited_param_list:
                print('param', splited_param.name)
            
        ZO_trainable_layers_list_wp = None
        if getattr(config, 'en_wp_np_mixture', False):
            if hasattr(model, 'ZO_trainable_layers_list_wp'):
                ZO_trainable_layers_list_wp = model.ZO_trainable_layers_list_wp
        
        if splited_layer_list is not None:
            ### pseudo ZO do not estimate 
            if getattr(config, 'en_pseudo_ZO', False):
                splited_layer_list = [layer for layer in splited_layer_list if 'classifier' not in layer.name]
            
            for splited_layer in splited_layer_list:
                print('layer', splited_layer.name)

                if ZO_trainable_layers_list_wp is not None:
                    if isinstance(splited_layer.layer, ZO_trainable_layers_list_wp):
                        splited_layer.mode = 'param'
        
        ZO_Estim = ZO_Estim_MC(
            model = model, 
            obj_fn_type = config.obj_fn_type,
            splited_param_list = splited_param_list,
            splited_layer_list = splited_layer_list,
            
            config = config,
        )
        return ZO_Estim
    else:
        return NotImplementedError

def build_obj_fn(obj_fn_type, **kwargs):
    if obj_fn_type == 'LM':
        obj_fn = ObjFnLM(**kwargs)
    elif obj_fn_type == 'LM_fc_layerwise':
        obj_fn = ObjFnLMFcLayerwise(**kwargs)
    elif obj_fn_type == 'LM_layerwise':
        obj_fn = ObjFnLMLayerwise(**kwargs)
    else:
        raise NotImplementedError(f"Unknown obj_fn_type: {obj_fn_type}")
    return obj_fn


class ObjFnLM:
    def __init__(self, model, batch):
        self.model = model
        self.batch = batch
    
    def __call__(self, return_loss_reduction='mean'):
        
        output = self.model(**self.batch)
        
        # import inspect
        # print(inspect.getsource(self.model.module.loss_function))
        # print(self.model.module.loss_function.__module__)
        # print(self.model.module.loss_function.__qualname__)    

        
        if return_loss_reduction == 'mean':
            loss = output.loss
        elif return_loss_reduction == 'none':
            logits = output.logits.float()
            labels = self.batch['labels']
            
            batch_sz = logits.size(0)
            seq_len = logits.size(1)
            vocab_size = logits.size(-1)

            # Shift so that tokens < n predict n
            labels = nn.functional.pad(labels, (0, 1), value=-100)
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            logits = logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(logits.device)
            loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100, reduction='none').reshape(batch_sz, seq_len)
            # print('loss shape', loss.shape)
        
        # elif return_loss_reduction == 'one-token':
        #     logits = output.logits.float()
        #     labels = self.batch['labels']
            
        #     batch_sz = logits.size(0)
        #     seq_len = logits.size(1)
        #     vocab_size = logits.size(-1)

        #     # original logits: [batch_sz, seq_len, vocab_size]
        #     # original labels: [batch_sz, seq_len]

        #     # Shift labels as before:
        #     labels_padded   = nn.functional.pad(self.batch['labels'], (0, 1), value=-100)  # [batch_sz, seq_len+1]
        #     shift_labels    = labels_padded[..., 1:]                                      # [batch_sz, seq_len]

        #     # pick token i
        #     i = 50
        #     logits_i = output.logits[:, i, :].float()     # [batch_sz, vocab_size]
        #     labels_i = shift_labels[:, i].to(logits_i.device)  # [batch_sz]

        #     # compute only that token’s loss
        #     loss = torch.nn.functional.cross_entropy(logits_i, labels_i,
        #                             ignore_index=-100,
        #                             reduction='mean')      # [batch_sz]
        #     # or reduction='mean' to get a scalar
            
        return output, loss


class ObjFnLMFcLayerwise:
    def __init__(self, data, target, model, criterion):
        self.input_ids = data
        self.target_ids = target
        
        self.ZO_iterable_block_name = getattr(model, 'ZO_iterable_block_name', None)
        self.ZO_pre_block_forward = getattr(model, 'ZO_pre_block_forward', None)
        self.ZO_post_block_forward = getattr(model, 'ZO_post_block_forward', None)
        
        self.split_modules_list = split_model(model, self.ZO_iterable_block_name)
        self.model = model
        self.criterion = criterion
    
    def __call__(self, starting_idx=0, ending_idx=None, input=None, return_loss_reduction='mean', detach_idx=None):
        if ending_idx == None:
            ending_idx = len(self.split_modules_list)

        if starting_idx == 0:
            y = self.input_ids
            ### ZO_pre_block_forward when start from image input
            if self.ZO_pre_block_forward is not None:
                hidden_states, attention_mask = self.ZO_pre_block_forward(self.input_ids)
        else:
            assert input is not None
            hidden_states, attention_mask = input

        for i in range(starting_idx, ending_idx):
            hidden_states = self.split_modules_list[i](hidden_states, attention_mask)

        if return_loss_reduction == 'pzo':
            hidden_states, attention_mask = hidden_states.detach(), attention_mask.detach()
            hidden_states.requires_grad = True 
            
            if self.ZO_post_block_forward is not None:
                output = self.ZO_post_block_forward(hidden_states)
            
            loss = self.criterion(output.view(-1, output.size(-1)), self.target_ids.view(-1))
            
            loss.backward()
            return hidden_states.detach(), hidden_states.grad.detach(), output, loss
        
        elif return_loss_reduction == 'no_loss':
            return (hidden_states, attention_mask)
        else:
            ### ZO_post_block_forward when end at classifier head
            if ending_idx == len(self.split_modules_list):
                if self.ZO_post_block_forward is not None:
                    output = self.ZO_post_block_forward(hidden_states)
                    
            if return_loss_reduction == 'mean':
                self.criterion.reduction = 'mean'
                loss = self.criterion(output.view(-1, output.size(-1)), self.target_ids.view(-1))
                return output, loss
            elif return_loss_reduction == 'none':
                self.criterion.reduction = 'none'
                bz, seq_len, vocab_size = output.size()
                loss = self.criterion(output.view(-1, vocab_size), self.target_ids.view(-1))
                loss = loss.view(bz, seq_len)
                loss = loss.mean(dim=1)
                self.criterion.reduction = 'mean'
                return output, loss
            else:
                raise NotImplementedError(f'Unknown {return_loss_reduction}')


class ObjFnLMLayerwise:
    def __init__(self, data, target, model, criterion):
        self.input_ids = data
        self.target_ids = target
        
        self.ZO_iterable_block_name = getattr(model, 'ZO_iterable_block_name', None)
        self.ZO_pre_block_forward = getattr(model, 'ZO_pre_block_forward', None)
        self.ZO_post_block_forward = getattr(model, 'ZO_post_block_forward', None)
        
        self.split_modules_list = split_model(model, self.ZO_iterable_block_name)
        self.model = model
        self.criterion = criterion
    
    def __call__(self, starting_idx=0, ending_idx=None, input=None, return_loss_reduction='mean', detach_idx=None):
        if ending_idx == None:
            ending_idx = len(self.split_modules_list)

        if starting_idx == 0:
            y = self.input_ids
            ### ZO_pre_block_forward when start from image input
            if self.ZO_pre_block_forward is not None:
                hidden_states, attention_mask = self.ZO_pre_block_forward(self.input_ids)
                attention_output = None
        else:
            assert input is not None
            hidden_states, attention_mask, attention_output = input
        
        for i in range(starting_idx, ending_idx):
            hidden_states, attention_mask, attention_output = self.split_modules_list[i](hidden_states, attention_mask, attention_output)

        if return_loss_reduction == 'pzo':
            hidden_states, attention_mask, attention_output = hidden_states.detach(), attention_mask.detach(), attention_output.detach()
            hidden_states.requires_grad = True 
            
            if self.ZO_post_block_forward is not None:
                output = self.ZO_post_block_forward(hidden_states)
            
            loss = self.criterion(output.view(-1, output.size(-1)), self.target_ids.view(-1))
            
            loss.backward()
            return hidden_states.detach(), hidden_states.grad.detach(), output, loss
        elif return_loss_reduction == 'no_loss':
            return (hidden_states, attention_mask, attention_output)
        else:
            ### ZO_post_block_forward when end at classifier head
            if ending_idx == len(self.split_modules_list):
                if self.ZO_post_block_forward is not None:
                    output = self.ZO_post_block_forward(hidden_states)
                    
            if return_loss_reduction == 'mean':
                self.criterion.reduction = 'mean'
                loss = self.criterion(output.view(-1, output.size(-1)), self.target_ids.view(-1))
                return output, loss
            elif return_loss_reduction == 'none':
                self.criterion.reduction = 'none'
                bz, seq_len, vocab_size = output.size()
                loss = self.criterion(output.view(-1, vocab_size), self.target_ids.view(-1))
                loss = loss.view(bz, seq_len)
                loss = loss.mean(dim=1)
                self.criterion.reduction = 'mean'
                return output, loss
            else:
                raise NotImplementedError(f'Unknown {return_loss_reduction}')