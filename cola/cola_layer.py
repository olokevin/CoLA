import torch
import torch.nn as nn
from transformers.activations import ACT2FN


class ColaLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank,
        bias=True,
        lr_act=True,
        lr_act_type="silu",
    ):
        super(ColaLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        if lr_act:
            self.lr_act = ACT2FN[lr_act_type]

        target_sdv = (in_features + out_features) ** (-1 / 2)
        self.cola_a = nn.Parameter(
            torch.randn(in_features, rank) / rank ** (1 / 4) * target_sdv ** (1 / 2)
        )
        self.cola_b = nn.Parameter(
            torch.randn(rank, out_features) / rank ** (1 / 4) * target_sdv ** (1 / 2)
        )

        if bias == False:
            self.register_parameter("bias", None)
        else:
            stdv = 1.0 / out_features ** (1 / 2)
            self.bias = torch.nn.Parameter(torch.randn(out_features))
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        return (
            f"cola_a: {self.cola_a.shape}, cola_b: {self.cola_b.shape}, "
            f"bias: {self.bias.shape if self.bias is not None else False}"
        )

    def forward(self, x):
        out = torch.matmul(x, self.cola_a)

        if hasattr(self, "lr_act"):
            out = self.lr_act(out)

        out = torch.matmul(out, self.cola_b)

        if self.bias is not None:
            out += self.bias

        return out
    
    def create_fwd_hook_add_perturbation(self, seed, sigma, rand_gen_fn, mask=None):
        def fwd_hook(module, input, output):
            # input is a tuple
            # module.in_value = input[0].detach().clone()
            # output is a tensor. inplace & return modifiled output both work
            
            # state = torch.get_rng_state()
            
            if seed is not None:
                torch.manual_seed(seed)
            u = rand_gen_fn(output.shape).to(output.dtype)
            
            # power law scaling, sigma = C j^{-1/2}, u = sigma * e
            # scale = torch.arange(1, u.size(1)).pow(-0.5)
            # scale = torch.cat([torch.tensor([1.0]), scale]).to(output.dtype).to(output.device)
            # u *= scale.reshape(1, -1, 1)
            
            # torch.set_rng_state(state)
            
            if mask is not None:
                u *= mask
                
            # module.perturbation = perturbation
            
            # output += sigma * perturbation
            return output + sigma * u
        return fwd_hook
      
    def create_fwd_hook_assign_grad(self, seed_list, scale_factor_list, rand_gen_fn, mask=None):
        """
        Create a forward hook function that uses an externally provided
        estimation of the layer's output gradient (grad_output_estimate)
        and computes gradients for cola_a, cola_b, and bias manually.
        """
        def forward_hook(module, inputs, output):
            ### retrieve ZO_grad_output
            for i, (seed, scale_factor) in enumerate(zip(seed_list, scale_factor_list)):
                torch.manual_seed(seed)
                u = rand_gen_fn(output.shape).to(output.dtype)
                
                # power law scaling, u / sigma^2 = e / sigma, sigma = C j^{-1/2}, C already in scale_factor
                # scale = torch.arange(1, u.size(1)).pow(0.5)
                # scale = torch.cat([torch.tensor([1.0]), scale]).to(output.dtype).to(output.device)
                # u *= scale.reshape(1, -1, 1)
                
                if mask is not None:
                    u *= mask
                    
                ### init
                if i == 0:
                    ZO_grad_output = torch.zeros_like(u)
                    
                ### accumulate
                ZO_grad_output += torch.einsum('bs,bsd->bsd', (scale_factor, u)).to(u.dtype) 

            module.ZO_grad_output = ZO_grad_output
            
            # Retrieve the input, shape: [B, T, in_features]
            x = inputs[0]
            
            # Recompute h = x @ cola_a, shape: [B, T, rank]
            h = torch.matmul(x, module.cola_a)

            # If an activation is applied, compute the activated output and its derivative.
            if hasattr(module, "lr_act"):
                h_act = module.lr_act(h)
                # For SiLU (also known as swish), the derivative is:
                # d/dh silu(h) = sigmoid(h) * (1 + h * (1 - sigmoid(h)))
                if hasattr(module, "lr_act_type") and module.lr_act_type == "silu":
                    sigmoid_h = torch.sigmoid(h)
                    act_deriv = sigmoid_h * (1 + h * (1 - sigmoid_h))
                else:
                    # Default derivative for identity if unknown activation.
                    act_deriv = torch.ones_like(h)
            else:
                h_act = h
                act_deriv = torch.ones_like(h)

            # Use the externally provided gradient for the output.
            grad_out = ZO_grad_output  # shape: [B, T, out_features]

            # === Compute Gradients via Einstein Summation (einsum) ===

            # For cola_b:
            # dL/dcola_b = sum_{b,t} [h_act[b,t].T * grad_out[b,t]]
            grad_cola_b = torch.einsum('btr,bto->ro', h_act, grad_out)

            # For bias:
            # dL/dbias = sum_{b,t} grad_out[b,t]
            grad_bias = grad_out.sum(dim=(0, 1)) if module.bias is not None else None

            # For cola_a:
            # First, backpropagate through the second matmul:
            # grad_h_act = grad_out @ cola_b^T, shape: [B, T, rank]
            grad_h_act = torch.matmul(grad_out, module.cola_b.transpose(0, 1))
            # Then apply the derivative of the activation: element-wise multiplication.
            grad_h = grad_h_act * act_deriv
            # Finally, dL/dcola_a = sum_{b,t} [x[b,t].T * grad_h[b,t]]
            grad_cola_a = torch.einsum('bti,btr->ir', x, grad_h)

            # Manually assign the computed gradients to the parameters.
            if module.cola_a.grad is None:
                module.cola_a.grad = grad_cola_a
            else:
                module.cola_a.grad += grad_cola_a
            if module.cola_b.grad is None:
                module.cola_b.grad = grad_cola_b
            else:
                module.cola_b.grad += grad_cola_b
            
            if module.bias is not None:
                if module.bias.grad is None:
                    module.bias.grad = grad_bias
                else:
                    module.bias.grad += grad_bias

            # Return the unchanged output.
            return output

        return forward_hook
    
    def zo_np_create_forward_hook(self, ZO_grad_output):
        """
        Create a forward hook function that uses an externally provided
        estimation of the layer's output gradient (grad_output_estimate)
        and computes gradients for cola_a, cola_b, and bias manually.
        """
        def forward_hook(module, inputs, output):
            # Retrieve the input, shape: [B, T, in_features]
            x = inputs[0]
            
            # Recompute h = x @ cola_a, shape: [B, T, rank]
            h = torch.matmul(x, module.cola_a)

            # If an activation is applied, compute the activated output and its derivative.
            if hasattr(module, "lr_act"):
                h_act = module.lr_act(h)
                # For SiLU (also known as swish), the derivative is:
                # d/dh silu(h) = sigmoid(h) * (1 + h * (1 - sigmoid(h)))
                if hasattr(module, "lr_act_type") and module.lr_act_type == "silu":
                    sigmoid_h = torch.sigmoid(h)
                    act_deriv = sigmoid_h * (1 + h * (1 - sigmoid_h))
                else:
                    # Default derivative for identity if unknown activation.
                    act_deriv = torch.ones_like(h)
            else:
                h_act = h
                act_deriv = torch.ones_like(h)

            # Use the externally provided gradient for the output.
            grad_out = ZO_grad_output  # shape: [B, T, out_features]

            # === Compute Gradients via Einstein Summation (einsum) ===

            # For cola_b:
            # dL/dcola_b = sum_{b,t} [h_act[b,t].T * grad_out[b,t]]
            grad_cola_b = torch.einsum('btr,bto->ro', h_act, grad_out)

            # For bias:
            # dL/dbias = sum_{b,t} grad_out[b,t]
            grad_bias = grad_out.sum(dim=(0, 1)) if module.bias is not None else None

            # For cola_a:
            # First, backpropagate through the second matmul:
            # grad_h_act = grad_out @ cola_b^T, shape: [B, T, rank]
            grad_h_act = torch.matmul(grad_out, module.cola_b.transpose(0, 1))
            # Then apply the derivative of the activation: element-wise multiplication.
            grad_h = grad_h_act * act_deriv
            # Finally, dL/dcola_a = sum_{b,t} [x[b,t].T * grad_h[b,t]]
            grad_cola_a = torch.einsum('bti,btr->ir', x, grad_h)

            # Manually assign the computed gradients to the parameters.
            module.cola_a.grad = grad_cola_a
            module.cola_b.grad = grad_cola_b
            if module.bias is not None:
                module.bias.grad = grad_bias

            # Return the unchanged output.
            return output

        return forward_hook


class ColaMDownProjLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank,
        lr_act=True,
        lr_act_type="silu",
    ):
        super(ColaMDownProjLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        if lr_act:
            self.lr_act = ACT2FN[lr_act_type]

        target_sdv = (in_features + out_features) ** (-1 / 2)
        self.cola_a = nn.Parameter(
            torch.randn(in_features, rank) / rank ** (1 / 4) * target_sdv ** (1 / 2)
        )

    def extra_repr(self):
        return f"cola_a: {self.cola_a.shape}"

    def forward(self, x):
        out = torch.matmul(x, self.cola_a)

        if hasattr(self, "lr_act"):
            out = self.lr_act(out)

        return out


class ColaMUpProjLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank,
        bias=True,
    ):
        super(ColaMUpProjLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        target_sdv = (in_features + out_features) ** (-1 / 2)
        self.cola_b = nn.Parameter(
            torch.randn(rank, out_features) / rank ** (1 / 4) * target_sdv ** (1 / 2)
        )
        if bias == False:
            self.register_parameter("bias", None)
        else:
            stdv = 1.0 / out_features ** (1 / 2)
            self.bias = torch.nn.Parameter(torch.randn(out_features))
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        return f"cola_b: {self.cola_b.shape}, bias: {self.bias.shape if self.bias is not None else False}"

    def forward(self, x):
        out = torch.matmul(x, self.cola_b)

        if self.bias is not None:
            out += self.bias

        return out
