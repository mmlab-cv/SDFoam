
#Implementation taken form NeuS https://github.com/Totoro97/NeuS

"""
Modified SDFNetwork that supports aligning geometric initialization to
the centre of a supplied point cloud.  This allows the network to be
initialised around the mean of the training data rather than
implicitly assuming the world origin at (0, 0, 0).  If a point cloud
is provided to the constructor via the ``point_cloud`` parameter, its
mean is computed and subtracted from all inputs during the forward
pass.  The existing geometric initialisation is preserved.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Positional encoding
def get_embedder(multires: int, input_dims: int = 3):
    if multires <= 0:
        return None, input_dims

    max_freq = multires - 1
    freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=multires)

    def embed(x):
        # allocate on x.device once
        fb = (2.0 ** torch.linspace(0, multires-1, multires, device=x.device))
        TWO_PI = 2.0 * math.pi
        outs = [x]
        for f in fb:
            w = TWO_PI * f
            outs += [torch.sin(x*w), torch.cos(x*w)]
        return torch.cat(outs, dim=-1)

    # One raw + 2*multires sincos for each dim
    input_ch = input_dims + 2 * multires * input_dims
    return embed, input_ch


class SDFNetwork(nn.Module):
    """
    Multi-layer perceptron that predicts a signed distance field (SDF).
    This variant accepts an optional ``point_cloud`` argument at
    construction time.  If provided, the centre of the point cloud is
    subtracted from all inputs before scaling, effectively aligning
    the geometric initialisation to the point cloud's centre rather
    than the origin.
    """

    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 point_cloud=None):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.embed_fn_fine = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        # Compute point cloud centre if provided
        if point_cloud is not None:
            # Accept both numpy arrays and torch tensors
            if isinstance(point_cloud, torch.Tensor):
                # Flatten to Nx3
                pc = point_cloud.reshape(-1, d_in)
            else:
                pc = torch.tensor(point_cloud, dtype=torch.float32).reshape(-1, d_in)
            self.register_buffer('pc_center', pc.mean(dim=0, keepdim=True))
        else:
            self.register_buffer('pc_center', None)

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Align inputs by subtracting the point cloud centre if provided
        if self.pc_center is not None:
            inputs = inputs - self.pc_center
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        # The first channel encodes the SDF; scale back to world units
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        y = self.sdf(x)
        (grad,) = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True, allow_unused=False)
        return grad

    def eikonal_loss(self, grads: torch.Tensor) -> torch.Tensor:
        loss = ((grads.norm(2, dim=-1) - 1) ** 2).mean()
        return loss

#Learnable sharpness paramter of the sigmoid activation used in NeuS-style rendering.  This is a single scalar that is shared across the entire image, and is applied to the SDF before the sigmoid to control the "sharpness" of the surface boundary.  The original NeuS paper uses a fixed value of 100, but making it learnable allows the model to adapt this parameter during training for potentially better convergence and results.
class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val, device):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val, device=device)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 1.0) #10.0
