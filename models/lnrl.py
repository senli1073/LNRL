import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque
from .loss import BCELoss
from ._factory import register_model,create_model


class Reparameter(nn.Module):
    """Reparameter Trick"""
    def __init__(self,feature_dim,hidden_dim):
        super().__init__()
        
        self.proj = nn.Linear(feature_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, hidden_dim)
        self.log_var = nn.Linear(hidden_dim, hidden_dim)
        
        
    def forward(self,x):
        
        x = x.flatten(1)
        x = self.proj(x)
        mu = self.mu(x)
        log_var = self.log_var(x)

        rand = torch.normal(mean=0.0, std=1.0, size=x.shape, device = x.device)
        x = mu + rand * torch.exp(log_var) ** 0.5
        
        return x,(mu,log_var)


class RFF(nn.Module):
    """
    Random Fourier Feature

    :math: `\phi_{arccos}(\mathbf{x})=\sqrt{\frac{1}{D}}[\text{ReLU}(\mathbf{w_{1}\cdot x}),\dots,
            \text{ReLU}(\mathbf{w_{D}\cdot x})]^T`
    :math: `\phi_{gaussian}(\mathbf{x})=\sqrt{\frac{1}{D}}[\text{sin}(\mathbf{w_{1}\cdot x}),\dots,
            \text{sin}(\mathbf{w_{D}\cdot x}),\text{cos}(\mathbf{w_{1}\cdot x}),\dots,\text{cos}
            (\mathbf{w_{D}\cdot x})]^T`

    Reference:
        [1] Alber, M., Kindermans, P. J., Schütt, K., Müller, K. R., & Sha, F.
            An empirical study on the properties of random bases for kernel methods.
            Advances in Neural Information Processing Systems, 30 (2017).
        [2] Rahimi, A., & Recht, B.
            Random features for large-scale kernel machines.
            Advances in neural information processing systems, 20 (2007).

    Shape:
        - Input: [N,*]
        - Output: [N,*]
    """

    def __init__(self, feature_dim: int, rff_dim: int, kernel: str = "gaussian"):
        super().__init__()

        assert kernel in ("gaussian", "arccos")

        if kernel == "gaussian":
            assert rff_dim % 2 == 0
            rff_dim = rff_dim // 2

        self.feature_dim = feature_dim
        self.rff_dim = rff_dim
        self.kernel = (
            self._kernel_gaussian if kernel == "gaussian" else self._kernel_arccos
        )

    def _kernel_arccos(self, x):
        return F.relu(x)

    def _kernel_gaussian(self, x):
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

    def forward(self, x,y):
        x = x.flatten(1)
        y = y.flatten(1)
        w = torch.randn((self.feature_dim, self.rff_dim),dtype=torch.float32,device=x.device)
        scale = torch.tensor(self.rff_dim**-0.5,dtype=torch.float32,device=x.device)
        x = torch.matmul(x, w)
        x = self.kernel(x) * scale
        y = torch.matmul(y, w)
        y = self.kernel(y) * scale

        return x,y


class ErrorLearner(nn.Module):
    """Error Learner"""

    def __init__(
        self,
        io_dim,
        head_dim,
        qkv_bias,
        attn_drop_rate,
        key_drop_rate,
        proj_drop_rate,
        pool_ker,
        norm_layer,
    ):
        super().__init__()

        self.num_heads = io_dim // head_dim

        self.pool = (
            nn.AvgPool1d(pool_ker)
            if pool_ker > 1
            else nn.Identity()
        )
        self.norm = norm_layer(io_dim) if pool_ker > 1 else nn.Identity()

        self.q_proj = nn.Conv1d(
            in_channels=io_dim, out_channels=io_dim, kernel_size=1, bias=qkv_bias
        )
        self.k_proj = nn.Conv1d(
            in_channels=io_dim, out_channels=io_dim, kernel_size=1, bias=qkv_bias
        )
        self.v_proj = nn.Conv1d(
            in_channels=io_dim, out_channels=io_dim, kernel_size=1, bias=qkv_bias
        )
        self.k_dropout = nn.Dropout(key_drop_rate)
        self.attn_dropout = nn.Dropout(attn_drop_rate)

        self.out_proj = nn.Conv1d(
            in_channels=io_dim, out_channels=io_dim, kernel_size=1, bias=qkv_bias
        )
        self.proj_dropout = nn.Dropout(proj_drop_rate)

    def forward(self, q,kv):
        N, C, L = q.size()

        q = self.q_proj(q).view(N, self.num_heads, C // self.num_heads, L)

        kv = self.pool(kv)
        kv = self.norm(kv)

        k = self.k_proj(kv).view(N, self.num_heads, C // self.num_heads, -1)
        v = self.v_proj(kv).view(N, self.num_heads, C // self.num_heads, -1)

        k = self.k_dropout(k)

        N, Nh, E, L = q.size()

        q_scaled = q / math.sqrt(E)

        attn = (q_scaled.transpose(-1, -2) @ k).softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v.transpose(-1, -2)).transpose(-1, -2).reshape(N, C, L)

        x = self.out_proj(x)
        x = self.proj_dropout(x)

        return x



class LNRL(nn.Module):
    """Lable Noise-Robust Learning"""
    def __init__(self,model_x,model_y,feature_channels:int,feature_dim=int,z_dim:int=100,grad_clip_value=5.0):
        """
        Args:
            model_x (nn.Module): model for input `x_sigma`.
            model_y (nn.Module): model for input `q`.
            feature_channels (int): Number of feature channels.
            feature_dim (int): Number of features.
            grad_clip_value (float): Gradient clipping value.
        """
        super().__init__()
        
        self.model_x = model_x
        self.model_y = model_y   
        
        self.el = ErrorLearner(io_dim=feature_channels,head_dim=int(feature_channels/2),qkv_bias=True,
                            attn_drop_rate=0.1,key_drop_rate=0.1,proj_drop_rate=0.1,
                            pool_ker=4,norm_layer= nn.BatchNorm1d)
        
        self.reparameter_x = Reparameter(feature_dim=feature_dim,hidden_dim=z_dim)
        self.reparameter_y = Reparameter(feature_dim=feature_dim,hidden_dim=z_dim)
        
        self.bce = BCELoss()
        
        self.grad_clip_value, self.grad_norm_queue = grad_clip_value, deque([torch.inf], maxlen=5)

        self.rff=RFF(feature_dim=feature_dim,rff_dim=z_dim,kernel="gaussian")

    
    def clip_grad(self):
        if self.grad_clip_value is not None:
            max_norm = max(self.grad_norm_queue)
            total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm * self.grad_clip_value)
            self.grad_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))
    
    def forward(self,x,y):
        """Get model output and loss.
        """
        # Encoders 
        feature_x = self.model_x.forward_features(x)
        feature_y = self.model_y.forward_features(y)

        # EL
        feature_y = self.el(feature_x.detach(),feature_y)
        
        # Reparameterization
        reparam_x,(mu_x,log_var_x) = self.reparameter_x(feature_x)
        reparam_y,(mu_y,log_var_y) = self.reparameter_y(feature_y)
        
        # KL
        loss_f = -0.5 * torch.mean(torch.sum(log_var_x - log_var_y - torch.exp(log_var_x) / torch.exp(log_var_y) - (mu_x - mu_y) ** 2 / torch.exp(log_var_y) + 1, dim=1)).reshape(1,1)
        
        sim_x = self.compute_cosine_similarity(reparam_x,reparam_x)
        sim_y = self.compute_cosine_similarity(reparam_y,reparam_y)
        
        # Sim
        loss_s = F.mse_loss(sim_x,sim_y).reshape(1,1)

        # Sim RFF
        rff_x,rff_y = self.rff(feature_x,feature_y)
        rff_sim_x = self.compute_cosine_similarity(rff_x,rff_x)
        rff_sim_y = self.compute_cosine_similarity(rff_y,rff_y)
        loss_rff = F.mse_loss(rff_sim_x,rff_sim_y).reshape(1,1)

        # Decoders
        out_x = self.model_x.forward_decoder(feature_x,in_samples=x.shape[-1])
        out_y = self.model_y.forward_decoder(feature_y,in_samples=x.shape[-1])
        
        loss_p = self.bce(out_x,y).reshape(1,1)
        loss_r = self.bce(out_y,y).reshape(1,1)
        
        loss = 0.4 * loss_f + 0.1 * loss_s + 0.1 * loss_rff + 0.4 * loss_p + 0.1 * loss_r

        return out_x,loss
        
        
    
    def compute_cosine_similarity(self, x0, x1)->torch.Tensor:
        """
        Computes cosine similarity between x0 and x1.
        Args:
            x0 (Tensor): shape (m,d)
            x1 (Tensor): shape (n,d)
        Returns:
            Tensor: Similarity matrix: shape (m,n)
        """
        assert len(x0.shape) == len(x1.shape) == 2
        assert x0.size(1) == x1.size(1)
        
        norm_x0 = F.normalize(x0, dim=1)
        norm_x1 = F.normalize(x1, dim=1)
        
        sim = norm_x0 @ norm_x1.transpose(0, 1)
        
        return sim

    
    
@register_model
def lnrl(**kwargs):
    
    model_x = create_model("seist",in_channels=1)
    model_y = create_model("seist",in_channels=2)
    lnrl = LNRL(model_x=model_x,model_y=model_y,feature_channels=96,feature_dim=9024)
    
    return lnrl
    
    
