import torch
import torch.nn as nn

from .ZO_utils import SplitedLayer, SplitedParam, split_model
from .ZO_Estim_MC import ZO_Estim_MC

### Model specific utils ###
# from .ZO_fwd_utils import trainable_layers_dict, get_iterable_block_name, ZO_pre_block_forward, ZO_post_block_forward
from tensor_transformers.tensor_layers.tensorized_layers import TensorizedLinear

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

# def MZI_commit_fn(layer, param, param_name) -> None:
#     def _commit_fn():
#         if param_name == "phase_U":
#             phase_bias = layer.phase_bias_U
#             delta_list = layer.delta_list_U
#             quantizer = layer.phase_U_quantizer
#             layer.U.data.copy_(
#                 layer.decomposer.reconstruct(
#                     delta_list,
#                     layer.decomposer.v2m(quantizer(param.view(phase_bias.size(0), phase_bias.size(1), -1)) + phase_bias),
#                 )
#             )
#         elif param_name == "phase_V":
#             phase_bias = layer.phase_bias_V
#             delta_list = layer.delta_list_V
#             quantizer = layer.phase_V_quantizer

#             layer.V.data.copy_(
#                 layer.decomposer.reconstruct(
#                     delta_list,
#                     layer.decomposer.v2m(quantizer(param.view(phase_bias.size(0), phase_bias.size(1), -1)) + phase_bias),
#                 )
#             )

#         elif param_name == "phase_S":
#             layer.S.data.copy_(param.data.cos().view_as(layer.S).mul_(layer.S_scale))
#         else:
#             raise ValueError(f"Wrong param_name {param_name}")
#     return _commit_fn

def build_ZO_Estim(config, model):
    if config.name == 'ZO_Estim_MC':
        ### split model
        ZO_iterable_block_name = getattr(model, 'ZO_iterable_block_name', None)
        split_modules_list = split_model(model, ZO_iterable_block_name)

        splited_param_list = None
        splited_layer_list = None

        ### Param perturb 
        if config.param_perturb_block_idx_list is not None:
            if config.param_perturb_block_idx_list == 'all':
                param_perturb_block_idx_list = list(range(len(split_modules_list)))
            else:
                param_perturb_block_idx_list = config.param_perturb_block_idx_list
            
            splited_param_list = []
            
            if config.en_partial_forward:                        
                for block_idx in param_perturb_block_idx_list:
                    if block_idx < 0:
                        block_idx = len(split_modules_list) + block_idx
                    block = split_modules_list[block_idx]
                    
                    for layer_name, layer in block.named_modules():
                        if isinstance(layer, (TensorizedLinear, nn.Linear, nn.Conv2d)):
                            for param_name, param in layer.named_parameters():
                                if param.requires_grad:
                                    splited_param_list.append(SplitedParam(idx=block_idx, name=f'{block_idx}.{layer_name}.{param_name}', layer=layer, param=param))  
            else:
                for param_name, param in model.named_parameters():
                    if param.requires_grad:
                        splited_param_list.append(SplitedParam(idx=-1, name=param_name, layer=None, param=param)) 
                ### select layers then select params
                # for layer_name, layer in model.named_modules():
                #     if isinstance(layer, (TensorizedLinear, nn.Linear, nn.Conv2d)):
                #         for param_name, param in layer.named_parameters():
                #             if param.requires_grad:
                #                 splited_param_list.append(SplitedParam(idx=-1, name=f'{layer_name}.{param_name}', layer=layer, param=param))  
                
        ### Actv perturb 
        if config.actv_perturb_block_idx_list is not None:
            splited_layer_list = []

            if config.actv_perturb_block_idx_list == 'all':
                actv_perturb_block_idx_list = list(range(len(split_modules_list)))
            else:
                actv_perturb_block_idx_list = config.actv_perturb_block_idx_list
            
            # if config.ZO_trainable_layers_list is not None:
            if hasattr(model, 'ZO_trainable_layers_dict'):
                ZO_trainable_layers_dict = model.ZO_trainable_layers_dict
                ZO_trainable_layers_list = create_trainable_layers_list(config.ZO_trainable_layers_list, ZO_trainable_layers_dict)
            else:
                ZO_trainable_layers_list = None

            
            ### model specified actv perturb
            if hasattr(model, 'ZO_trainable_blocks_name_idx'):
                for name, idx in model.ZO_trainable_blocks_name_idx.items():    
                    if idx == -1:
                        idx = len(split_modules_list)
                    splited_layer_list.append(SplitedLayer(idx=idx, name=name, layer=getattr(model, name)))
            
            # if 'ATIS' in config.obj_fn_type:
            #     splited_layer_list.append(SplitedLayer(idx=0, name='embedding', layer=model.embedding))
            
            if config.en_partial_forward:
                
                for block_idx in actv_perturb_block_idx_list:
                    if block_idx < 0:
                        block_idx = len(split_modules_list) + block_idx
                    block = split_modules_list[block_idx]
                    if ZO_trainable_layers_list is not None:
                        if type(block) in ZO_trainable_layers_list:
                            splited_layer_list.append(SplitedLayer(idx=block_idx, name=f'{ZO_iterable_block_name}.{block_idx}', layer=block))
                        else:
                            for name, layer in block.named_children():
                                if type(layer) in ZO_trainable_layers_list:
                                    splited_layer_list.append(SplitedLayer(idx=block_idx, name=f'{ZO_iterable_block_name}.{block_idx}.{name}', layer=layer))
            
            else:
                if ZO_trainable_layers_list is not None:
                    for layer_name, layer in model.named_modules():
                        if any([str(x) in layer_name for x in actv_perturb_block_idx_list]):
                            if type(layer) in ZO_trainable_layers_list:
                                splited_layer_list.append(SplitedLayer(idx=-1, name=layer_name, layer=layer))
                else:
                    for layer_name, layer in model.named_modules():
                        if any([str(x) in layer_name for x in actv_perturb_block_idx_list]):
                            if type(layer) in (TensorizedLinear, nn.Linear, nn.Conv2d):
                                splited_layer_list.append(SplitedLayer(idx=-1, name=layer_name, layer=layer))

            # if 'ATIS' in config.obj_fn_type:
            #     splited_layer_list.append(SplitedLayer(idx=len(split_modules_list), name='classifier', layer=model.classifier))
            #     splited_layer_list.append(SplitedLayer(idx=len(split_modules_list), name='slot_classifier', layer=model.slot_classifier))
                            
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
    if obj_fn_type == 'classifier':
        obj_fn = build_obj_fn_classifier(**kwargs)
    elif obj_fn_type == 'classifier_layerwise':
        obj_fn = build_obj_fn_classifier_layerwise(**kwargs)
    elif obj_fn_type == 'classifier_acc':
        obj_fn = build_obj_fn_classifier_acc(**kwargs)
    elif obj_fn_type == 'pinn':
        obj_fn = build_obj_fn_pinn(**kwargs)
    elif obj_fn_type == 'ATIS':
        obj_fn = build_obj_fn_ATIS(**kwargs)
    elif obj_fn_type == 'ATIS_fc_layerwise':
        obj_fn = build_obj_fn_ATIS_fc_layerwise(**kwargs)
    elif obj_fn_type == 'ATIS_layerwise':
        obj_fn = build_obj_fn_ATIS_layerwise(**kwargs)
    elif obj_fn_type == 'MNLI':
        obj_fn = build_obj_fn_MNLI(**kwargs)
    elif obj_fn_type == 'MNLI_fc_layerwise':
        obj_fn = build_obj_fn_MNLI_fc_layerwise(**kwargs)
        # obj_fn = MNLI_fc_layerwise(**kwargs)
    elif obj_fn_type == 'MNLI_layerwise':
        obj_fn = build_obj_fn_MNLI_layerwise(**kwargs)
    elif obj_fn_type == 'LM':
        obj_fn = build_obj_fn_LM(**kwargs)
    elif obj_fn_type == 'LM_fc_layerwise':
        obj_fn = build_obj_fn_LM_fc_layerwise(**kwargs)
    elif obj_fn_type == 'LM_layerwise':
        obj_fn = build_obj_fn_LM_layerwise(**kwargs)
    else:
        raise NotImplementedError
    return obj_fn

def build_obj_fn_pinn(model, dataset, loss_fn, inputs=None):
    def _obj_fn(return_loss_reduction='mean'):
        train_loss = loss_fn(model=model, dataset=dataset, inputs=inputs, return_loss_reduction=return_loss_reduction)
        y = False
        return y, train_loss

    return _obj_fn
  
def build_obj_fn_classifier(data, target, model, criterion):
    def _obj_fn(return_loss_reduction='mean'):
        y = model(data)
        # return y, criterion(y, target)
      
        if return_loss_reduction == 'mean':
            criterion.reduction = 'mean'
            return y, criterion(y, target)
        elif return_loss_reduction == 'none':
            criterion.reduction = 'none'
            loss = criterion(y, target)
            criterion.reduction = 'mean'
            return y, loss
    
    return _obj_fn

def build_obj_fn_classifier_acc(data, target, model, criterion):
    def _obj_fn():
        outputs = model(data)
        _, predicted = outputs.max(1)
        total = target.size(0)
        correct = predicted.eq(target).sum().item()
        err = 1 - correct / total

        return outputs, err
    
    return _obj_fn

def build_obj_fn_classifier_layerwise(data, target, model, criterion):
    ZO_iterable_block_name = getattr(model, 'ZO_iterable_block_name', None)
    ZO_pre_block_forward = getattr(model, 'ZO_pre_block_forward', None)
    ZO_post_block_forward = getattr(model, 'ZO_post_block_forward', None)
    
    split_modules_list = split_model(model, ZO_iterable_block_name)
    
    # if no attribute for _obj_fn: same as build_obj_fn_classifier
    def _obj_fn(starting_idx=0, ending_idx=None, input=None, return_loss_reduction='mean', detach_idx=None):
        if ending_idx == None:
            ending_idx = len(split_modules_list)

        if starting_idx == 0:
            y = data
            ### ZO_pre_block_forward when start from image input
            if ZO_pre_block_forward is not None:
                y = ZO_pre_block_forward(y)
        else:
            assert input is not None
            y = input
        
        # if input is not None:
        #     y = input
        # else:
        #     assert starting_idx == 0
        #     y = data
        #     ### ZO_pre_block_forward when start from image input
        #     if ZO_pre_block_forward is not None:
        #         y = ZO_pre_block_forward(y)
        
        if detach_idx is not None and detach_idx < 0:
            detach_idx = len(split_modules_list) + detach_idx
        
        for i in range(starting_idx, ending_idx):
            y = split_modules_list[i](y)
            if detach_idx is not None and i == detach_idx:
                y = y.detach()
                y.requires_grad = True
           
        if return_loss_reduction == 'no_loss':
            return y
        else:
            ### ZO_post_block_forward when end at classifier head
            if ending_idx == len(split_modules_list):
                if ZO_post_block_forward is not None:
                    y = ZO_post_block_forward(y)
                    
            if return_loss_reduction == 'mean':
                criterion.reduction = 'mean'
                return y, criterion(y, target)
            elif return_loss_reduction == 'none':
                criterion.reduction = 'none'
                loss = criterion(y, target)
                criterion.reduction = 'mean'
                return y, loss
            else:
                raise NotImplementedError(f'Unknown {return_loss_reduction}')
    
    return _obj_fn

def build_obj_fn_ATIS(data, target, model, criterion):
    def _obj_fn(return_loss_reduction='mean'):
        # optimizer.step((w1,attn,seg),(target,slot_label))
        w1 = data[0]
        attn = data[1]
        seg = data[2]

        target_0 = target[0]
        slot_label = target[1]
        config_forward = target[2]

        pred,pred_slot = model(w1,mask=attn,seg=seg, config_forward=config_forward)

        pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)
        
        if return_loss_reduction == 'mean':
            criterion.reduction = 'mean'
            loss_MLM =  criterion(pred_slot, slot_label)
            loss = criterion(pred,target_0)  + loss_MLM
        elif return_loss_reduction == 'none':
            criterion.reduction = 'none'
            batch_size = pred.size(0)
            loss_MLM =  criterion(pred_slot, slot_label).view(batch_size, -1).mean(dim=1)
            loss = criterion(pred,target_0)  + loss_MLM
            criterion.reduction = 'mean'
        
        return (pred, pred_slot), loss
    return _obj_fn

def build_obj_fn_ATIS_fc_layerwise(data, target, model, criterion):
    w1 = data[0]
    attn = data[1]
    seg = data[2]

    target_0 = target[0]
    slot_label = target[1]
    config_forward = target[2]
    
    ZO_iterable_block_name = getattr(model, 'ZO_iterable_block_name', None)
    ZO_pre_block_forward = getattr(model, 'ZO_pre_block_forward', None)
    ZO_post_block_forward = getattr(model, 'ZO_post_block_forward', None)
    
    split_modules_list = split_model(model, ZO_iterable_block_name)
    
    # if no attribute for _obj_fn: same as build_obj_fn_classifier
    def _obj_fn(starting_idx=0, ending_idx=None, input=None, return_loss_reduction='mean', detach_idx=None):
        if ending_idx == None:
            ending_idx = len(split_modules_list)

        if starting_idx == 0:
            y = data
            ### ZO_pre_block_forward when start from image input
            if ZO_pre_block_forward is not None:
                hidden_states, attention_mask = ZO_pre_block_forward(w1,mask=attn,seg=seg, config_forward=config_forward)
        else:
            assert input is not None
            hidden_states, attention_mask = input
        
        # if detach_idx is not None and detach_idx < 0:
        #     detach_idx = len(split_modules_list) + detach_idx
        
        for i in range(starting_idx, ending_idx):
            hidden_states = split_modules_list[i](hidden_states, attention_mask)
            # if detach_idx is not None and i == detach_idx:
            #     hidden_states, attention_mask, attention_output = hidden_states.detach(), attention_mask.detach(), attention_output.detach()
            #     hidden_states.requires_grad = True

        if return_loss_reduction == 'pzo':
            hidden_states, attention_mask = hidden_states.detach(), attention_mask.detach()
            hidden_states.requires_grad = True 
            
            if ZO_post_block_forward is not None:
                pred,pred_slot = ZO_post_block_forward(hidden_states)
                pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)
            
            loss_MLM =  criterion(pred_slot, slot_label)
            loss = criterion(pred,target_0)  + loss_MLM
            
            loss.backward()
            return hidden_states.detach(), hidden_states.grad.detach(), (pred,pred_slot), loss
            
        elif return_loss_reduction == 'no_loss':
            return (hidden_states, attention_mask)
        else:
            ### ZO_post_block_forward when end at classifier head
            if ending_idx == len(split_modules_list):
                if ZO_post_block_forward is not None:
                    pred,pred_slot = ZO_post_block_forward(hidden_states)
                    pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)
                    
            if return_loss_reduction == 'mean':
                criterion.reduction = 'mean'
                loss_MLM =  criterion(pred_slot, slot_label)
                loss = criterion(pred,target_0)  + loss_MLM
                return (pred, pred_slot), loss
            elif return_loss_reduction == 'none':
                criterion.reduction = 'none'
                batch_size = pred.size(0)
                loss_MLM =  criterion(pred_slot, slot_label).view(batch_size, -1).mean(dim=1)
                loss = criterion(pred,target_0)  + loss_MLM
                criterion.reduction = 'mean'
                return (pred, pred_slot), loss
            else:
                raise NotImplementedError(f'Unknown {return_loss_reduction}')
    
    return _obj_fn
  
def build_obj_fn_ATIS_layerwise(data, target, model, criterion):
    w1 = data[0]
    attn = data[1]
    seg = data[2]

    target_0 = target[0]
    slot_label = target[1]
    config_forward = target[2]
    
    ZO_iterable_block_name = getattr(model, 'ZO_iterable_block_name', None)
    ZO_pre_block_forward = getattr(model, 'ZO_pre_block_forward', None)
    ZO_post_block_forward = getattr(model, 'ZO_post_block_forward', None)
    
    split_modules_list = split_model(model, ZO_iterable_block_name)
    
    # if no attribute for _obj_fn: same as build_obj_fn_classifier
    def _obj_fn(starting_idx=0, ending_idx=None, input=None, return_loss_reduction='mean', detach_idx=None):
        if ending_idx == None:
            ending_idx = len(split_modules_list)

        if starting_idx == 0:
            y = data
            ### ZO_pre_block_forward when start from image input
            if ZO_pre_block_forward is not None:
                hidden_states, attention_mask = ZO_pre_block_forward(w1,mask=attn,seg=seg, config_forward=config_forward)
                attention_output = None
        else:
            assert input is not None
            hidden_states, attention_mask, attention_output = input
        
        # if detach_idx is not None and detach_idx < 0:
        #     detach_idx = len(split_modules_list) + detach_idx
        
        for i in range(starting_idx, ending_idx):
            hidden_states, attention_mask, attention_output = split_modules_list[i](hidden_states, attention_mask, attention_output)
            # if detach_idx is not None and i == detach_idx:
            #     hidden_states, attention_mask, attention_output = hidden_states.detach(), attention_mask.detach(), attention_output.detach()
            #     hidden_states.requires_grad = True

        if return_loss_reduction == 'pzo':
            hidden_states, attention_mask, attention_output = hidden_states.detach(), attention_mask.detach(), attention_output.detach()
            hidden_states.requires_grad = True 
            
            if ZO_post_block_forward is not None:
                pred,pred_slot = ZO_post_block_forward(hidden_states)
                pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)
            
            loss_MLM =  criterion(pred_slot, slot_label)
            loss = criterion(pred,target_0)  + loss_MLM
            
            loss.backward()
            return hidden_states.detach(), hidden_states.grad.detach(), (pred,pred_slot), loss
            
        elif return_loss_reduction == 'no_loss':
            return (hidden_states, attention_mask, attention_output)
        else:
            ### ZO_post_block_forward when end at classifier head
            if ending_idx == len(split_modules_list):
                if ZO_post_block_forward is not None:
                    pred,pred_slot = ZO_post_block_forward(hidden_states)
                    pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)
                    
            if return_loss_reduction == 'mean':
                criterion.reduction = 'mean'
                loss_MLM =  criterion(pred_slot, slot_label)
                loss = criterion(pred,target_0)  + loss_MLM
                return (pred, pred_slot), loss
            elif return_loss_reduction == 'none':
                criterion.reduction = 'none'
                batch_size = pred.size(0)
                loss_MLM =  criterion(pred_slot, slot_label).view(batch_size, -1).mean(dim=1)
                loss = criterion(pred,target_0)  + loss_MLM
                criterion.reduction = 'mean'
                return (pred, pred_slot), loss
            else:
                raise NotImplementedError(f'Unknown {return_loss_reduction}')
    
    return _obj_fn

def build_obj_fn_MNLI(data, target, model, criterion):
    def _obj_fn(return_loss_reduction='mean'):
        # optimizer.step((w1,attn,seg),(target,slot_label))
        
        input_ids, attn_mask, seg, config_forward = data
        target_labels = target

        static_y_pred = model(input_ids,mask=attn_mask,seg=seg,config_forward=config_forward)
        
        if return_loss_reduction == 'mean':
            criterion.reduction = 'mean'
            static_loss = criterion(static_y_pred, target_labels)
        elif return_loss_reduction == 'none':
            criterion.reduction = 'none'
            static_loss = criterion(static_y_pred, target_labels)
            criterion.reduction = 'mean'
        
        return static_y_pred, static_loss
    return _obj_fn

def build_obj_fn_MNLI_fc_layerwise(data, target, model, criterion):
    input_ids, attn_mask, seg, config_forward = data
    target_labels = target
    
    ZO_iterable_block_name = getattr(model, 'ZO_iterable_block_name', None)
    ZO_pre_block_forward = getattr(model, 'ZO_pre_block_forward', None)
    ZO_post_block_forward = getattr(model, 'ZO_post_block_forward', None)
    
    split_modules_list = split_model(model, ZO_iterable_block_name)
    
    # if no attribute for _obj_fn: same as build_obj_fn_classifier
    def _obj_fn(starting_idx=0, ending_idx=None, input=None, return_loss_reduction='mean', detach_idx=None):
        if ending_idx == None:
            ending_idx = len(split_modules_list)

        if starting_idx == 0:
            y = data
            ### ZO_pre_block_forward when start from image input
            if ZO_pre_block_forward is not None:
                hidden_states, attention_mask = ZO_pre_block_forward(input_ids,mask=attn_mask,seg=seg, config_forward=config_forward)
        else:
            assert input is not None
            hidden_states, attention_mask = input
        
        if detach_idx is not None and detach_idx < 0:
            detach_idx = len(split_modules_list) + detach_idx
        
        for i in range(starting_idx, ending_idx):
            hidden_states = split_modules_list[i](hidden_states, attention_mask)
            if detach_idx is not None and i == detach_idx:
                detach_hidden_states, detach_attention_mask = hidden_states.detach(), attention_mask.detach()
                detach_hidden_states.requires_grad = True
                hidden_states, attention_mask = detach_hidden_states, detach_attention_mask

        if return_loss_reduction == 'pzo':
            if detach_idx is None:
                detach_hidden_states, detach_attention_mask = hidden_states.detach(), attention_mask.detach()
                detach_hidden_states.requires_grad = True
                hidden_states, attention_mask = detach_hidden_states, detach_attention_mask
            
            if ZO_post_block_forward is not None:
                static_y_pred = ZO_post_block_forward(hidden_states)
            
            static_loss = criterion(static_y_pred, target_labels)
            
            static_loss.backward()
            return detach_hidden_states, detach_hidden_states.grad.detach(), static_y_pred, static_loss
        
        elif return_loss_reduction == 'pzo_nograd':
            if detach_idx is None:
                detach_hidden_states, detach_attention_mask = hidden_states.detach(), attention_mask.detach()
            
            return detach_hidden_states
          
        elif return_loss_reduction == 'no_loss':
            return (hidden_states, attention_mask)
        else:
            ### ZO_post_block_forward when end at classifier head
            if ending_idx == len(split_modules_list):
                if ZO_post_block_forward is not None:
                    static_y_pred = ZO_post_block_forward(hidden_states)
                    
            if return_loss_reduction == 'mean':
                criterion.reduction = 'mean'
                static_loss = criterion(static_y_pred, target_labels)
                return static_y_pred, static_loss
            elif return_loss_reduction == 'none':
                criterion.reduction = 'none'
                static_loss = criterion(static_y_pred, target_labels)
                criterion.reduction = 'mean'
                return static_y_pred, static_loss
            else:
                raise NotImplementedError(f'Unknown {return_loss_reduction}')
    
    return _obj_fn

def build_obj_fn_MNLI_layerwise(data, target, model, criterion):
    input_ids, attn_mask, seg, config_forward = data
    target_labels = target
    
    ZO_iterable_block_name = getattr(model, 'ZO_iterable_block_name', None)
    ZO_pre_block_forward = getattr(model, 'ZO_pre_block_forward', None)
    ZO_post_block_forward = getattr(model, 'ZO_post_block_forward', None)
    
    split_modules_list = split_model(model, ZO_iterable_block_name)
    
    # if no attribute for _obj_fn: same as build_obj_fn_classifier
    def _obj_fn(starting_idx=0, ending_idx=None, input=None, return_loss_reduction='mean', detach_idx=None):
        if ending_idx == None:
            ending_idx = len(split_modules_list)

        if starting_idx == 0:
            y = data
            ### ZO_pre_block_forward when start from image input
            if ZO_pre_block_forward is not None:
                hidden_states, attention_mask = ZO_pre_block_forward(input_ids,mask=attn_mask,seg=seg, config_forward=config_forward)
                attention_output = None
        else:
            assert input is not None
            hidden_states, attention_mask, attention_output = input
        
        # if detach_idx is not None and detach_idx < 0:
        #     detach_idx = len(split_modules_list) + detach_idx
        
        for i in range(starting_idx, ending_idx):
            hidden_states, attention_mask, attention_output = split_modules_list[i](hidden_states, attention_mask, attention_output)
            # if detach_idx is not None and i == detach_idx:
            #     y = y.detach()
            #     y.requires_grad = True

        if return_loss_reduction == 'pzo':
            hidden_states, attention_mask, attention_output = hidden_states.detach(), attention_mask.detach(), attention_output.detach()
            hidden_states.requires_grad = True 
            
            if ZO_post_block_forward is not None:
                static_y_pred = ZO_post_block_forward(hidden_states)
            
            static_loss = criterion(static_y_pred, target_labels)
            
            static_loss.backward()
            return hidden_states.detach(), hidden_states.grad.detach(), static_y_pred, static_loss
        
        elif return_loss_reduction == 'no_loss':
            return (hidden_states, attention_mask, attention_output)
        else:
            ### ZO_post_block_forward when end at classifier head
            if ending_idx == len(split_modules_list):
                if ZO_post_block_forward is not None:
                    static_y_pred = ZO_post_block_forward(hidden_states)
                    
            if return_loss_reduction == 'mean':
                criterion.reduction = 'mean'
                static_loss = criterion(static_y_pred, target_labels)
                return static_y_pred, static_loss
            elif return_loss_reduction == 'none':
                criterion.reduction = 'none'
                static_loss = criterion(static_y_pred, target_labels)
                criterion.reduction = 'mean'
                return static_y_pred, static_loss
            else:
                raise NotImplementedError(f'Unknown {return_loss_reduction}')
    
    return _obj_fn

def build_obj_fn_LM(data, target, model, criterion):
    def _obj_fn(return_loss_reduction='mean'):
        
        input_ids = data
        target_ids = target
        
        output = model(input_ids)
        
        if return_loss_reduction == 'mean':
            criterion.reduction = 'mean'
            loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
        elif return_loss_reduction == 'none':
            criterion.reduction = 'none'
            # (bz, seq_len, vocab_size)
            bz, seq_len, vocab_size = output.size()
            loss = criterion(output.view(-1, vocab_size), target_ids.view(-1))
            loss = loss.view(bz, seq_len)
            loss = loss.mean(dim=1)
            criterion.reduction = 'mean'
        
        return output, loss
    return _obj_fn

def build_obj_fn_LM_fc_layerwise(data, target, model, criterion):
    input_ids = data
    target_ids = target
    
    ZO_iterable_block_name = getattr(model, 'ZO_iterable_block_name', None)
    ZO_pre_block_forward = getattr(model, 'ZO_pre_block_forward', None)
    ZO_post_block_forward = getattr(model, 'ZO_post_block_forward', None)
    
    split_modules_list = split_model(model, ZO_iterable_block_name)
    
    # if no attribute for _obj_fn: same as build_obj_fn_classifier
    def _obj_fn(starting_idx=0, ending_idx=None, input=None, return_loss_reduction='mean', detach_idx=None):
        if ending_idx == None:
            ending_idx = len(split_modules_list)

        if starting_idx == 0:
            y = data
            ### ZO_pre_block_forward when start from image input
            if ZO_pre_block_forward is not None:
                hidden_states, attention_mask = ZO_pre_block_forward(input_ids)
        else:
            assert input is not None
            hidden_states, attention_mask = input
        
        # if detach_idx is not None and detach_idx < 0:
        #     detach_idx = len(split_modules_list) + detach_idx

        for i in range(starting_idx, ending_idx):
            hidden_states = split_modules_list[i](hidden_states, attention_mask)
            # if detach_idx is not None and i == detach_idx:
            #     y = y.detach()
            #     y.requires_grad = True

        if return_loss_reduction == 'pzo':
            hidden_states, attention_mask = hidden_states.detach(), attention_mask.detach()
            hidden_states.requires_grad = True 
            
            if ZO_post_block_forward is not None:
                output = ZO_post_block_forward(hidden_states)
            
            loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
            
            loss.backward()
            return hidden_states.detach(), hidden_states.grad.detach(), output, loss
        
        elif return_loss_reduction == 'no_loss':
            return (hidden_states, attention_mask)
        else:
            ### ZO_post_block_forward when end at classifier head
            if ending_idx == len(split_modules_list):
                if ZO_post_block_forward is not None:
                    output = ZO_post_block_forward(hidden_states)
                    
            if return_loss_reduction == 'mean':
                criterion.reduction = 'mean'
                loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
                return output, loss
            elif return_loss_reduction == 'none':
                criterion.reduction = 'none'
                bz, seq_len, vocab_size = output.size()
                loss = criterion(output.view(-1, vocab_size), target_ids.view(-1))
                loss = loss.view(bz, seq_len)
                loss = loss.mean(dim=1)
                criterion.reduction = 'mean'
                return output, loss
            else:
                raise NotImplementedError(f'Unknown {return_loss_reduction}')
    
    return _obj_fn

def build_obj_fn_LM_layerwise(data, target, model, criterion):
    input_ids = data
    target_ids = target
    
    ZO_iterable_block_name = getattr(model, 'ZO_iterable_block_name', None)
    ZO_pre_block_forward = getattr(model, 'ZO_pre_block_forward', None)
    ZO_post_block_forward = getattr(model, 'ZO_post_block_forward', None)
    
    split_modules_list = split_model(model, ZO_iterable_block_name)
    
    # if no attribute for _obj_fn: same as build_obj_fn_classifier
    def _obj_fn(starting_idx=0, ending_idx=None, input=None, return_loss_reduction='mean', detach_idx=None):
        if ending_idx == None:
            ending_idx = len(split_modules_list)

        if starting_idx == 0:
            y = data
            ### ZO_pre_block_forward when start from image input
            if ZO_pre_block_forward is not None:
                hidden_states, attention_mask = ZO_pre_block_forward(input_ids)
                attention_output = None
        else:
            assert input is not None
            hidden_states, attention_mask, attention_output = input
        
        # if detach_idx is not None and detach_idx < 0:
        #     detach_idx = len(split_modules_list) + detach_idx
        
        for i in range(starting_idx, ending_idx):
            hidden_states, attention_mask, attention_output = split_modules_list[i](hidden_states, attention_mask, attention_output)
            # if detach_idx is not None and i == detach_idx:
            #     y = y.detach()
            #     y.requires_grad = True

        if return_loss_reduction == 'pzo':
            hidden_states, attention_mask, attention_output = hidden_states.detach(), attention_mask.detach(), attention_output.detach()
            hidden_states.requires_grad = True 
            
            if ZO_post_block_forward is not None:
                output = ZO_post_block_forward(hidden_states)
            
            loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
            
            loss.backward()
            return hidden_states.detach(), hidden_states.grad.detach(), output, loss
        elif return_loss_reduction == 'no_loss':
            return (hidden_states, attention_mask, attention_output)
        else:
            ### ZO_post_block_forward when end at classifier head
            if ending_idx == len(split_modules_list):
                if ZO_post_block_forward is not None:
                    output = ZO_post_block_forward(hidden_states)
                    
            if return_loss_reduction == 'mean':
                criterion.reduction = 'mean'
                loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
                return output, loss
            elif return_loss_reduction == 'none':
                criterion.reduction = 'none'
                bz, seq_len, vocab_size = output.size()
                loss = criterion(output.view(-1, vocab_size), target_ids.view(-1))
                loss = loss.view(bz, seq_len)
                loss = loss.mean(dim=1)
                criterion.reduction = 'mean'
                return output, loss
            else:
                raise NotImplementedError(f'Unknown {return_loss_reduction}')
    
    return _obj_fn
  
class MNLI_fc_layerwise:
    def __init__(self, data, target, model, criterion):
        self.input_ids, self.attn_mask, self.seg, self.config_forward = data
        self.target_labels = target
        self.model = model
        self.criterion = criterion

        self.ZO_iterable_block_name = getattr(model, 'ZO_iterable_block_name', None)
        self.ZO_pre_block_forward = getattr(model, 'ZO_pre_block_forward', None)
        self.ZO_post_block_forward = getattr(model, 'ZO_post_block_forward', None)

        self.split_modules_list = split_model(model, self.ZO_iterable_block_name)
    
    def get_mask(self):
        return self.attn_mask

    def __call__(self, starting_idx=0, ending_idx=None, input=None, return_loss_reduction='mean', detach_idx=None):
        if ending_idx is None:
            ending_idx = len(self.split_modules_list)

        if starting_idx == 0:
            y = self.input_ids
            # ZO_pre_block_forward when starting from image input
            if self.ZO_pre_block_forward is not None:
                hidden_states, attention_mask = self.ZO_pre_block_forward(
                    self.input_ids, mask=self.attn_mask, seg=self.seg, config_forward=self.config_forward
                )
        else:
            assert input is not None
            hidden_states, attention_mask = input

        if detach_idx is not None and detach_idx < 0:
            detach_idx = len(self.split_modules_list) + detach_idx

        for i in range(starting_idx, ending_idx):
            hidden_states = self.split_modules_list[i](hidden_states, attention_mask)
            if detach_idx is not None and i == detach_idx:
                detach_hidden_states, detach_attention_mask = hidden_states.detach(), attention_mask.detach()
                detach_hidden_states.requires_grad = True
                hidden_states, attention_mask = detach_hidden_states, detach_attention_mask

        if return_loss_reduction == 'pzo':
            if detach_idx is None:
                detach_hidden_states, detach_attention_mask = hidden_states.detach(), attention_mask.detach()
                detach_hidden_states.requires_grad = True
                hidden_states, attention_mask = detach_hidden_states, detach_attention_mask

            if self.ZO_post_block_forward is not None:
                static_y_pred = self.ZO_post_block_forward(hidden_states)

            static_loss = self.criterion(static_y_pred, self.target_labels)

            static_loss.backward()
            return detach_hidden_states, detach_hidden_states.grad.detach(), static_y_pred, static_loss

        elif return_loss_reduction == 'pzo_nograd':
            if detach_idx is None:
                detach_hidden_states, detach_attention_mask = hidden_states.detach(), attention_mask.detach()

            return detach_hidden_states

        elif return_loss_reduction == 'no_loss':
            return (hidden_states, attention_mask)

        else:
            # ZO_post_block_forward when ending at classifier head
            if ending_idx == len(self.split_modules_list):
                if self.ZO_post_block_forward is not None:
                    static_y_pred = self.ZO_post_block_forward(hidden_states)

            if return_loss_reduction == 'mean':
                self.criterion.reduction = 'mean'
                static_loss = self.criterion(static_y_pred, self.target_labels)
                return static_y_pred, static_loss
            elif return_loss_reduction == 'none':
                self.criterion.reduction = 'none'
                static_loss = self.criterion(static_y_pred, self.target_labels)
                self.criterion.reduction = 'mean'
                return static_y_pred, static_loss
            else:
                raise NotImplementedError(f'Unknown {return_loss_reduction}')