import math
import torch
from Metis.lowrank_eig import svd_lowrank_eig_graph_pipelined


class NewMuon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                    
                with torch.no_grad():
                    device = p.device
                    orig_dtype = p.dtype
                    G = g.detach().to(device=device, dtype=torch.float32).clone()
                    m, n = G.shape
                    k = int(max(1, round(0.015 * min(m, n))))
                    
                    U, S, V = torch.svd_lowrank(G, q=k)
                    # U, S, V = svd_lowrank_eig_graph_pipelined(G, q=k)
                    
                    head = (U * S.unsqueeze(0)) @ V.t()
                    tail = G - head
                    
                    n_tail = min(m, n) - k
                    tail_fro_norm_sq = torch.norm(tail, p='fro') ** 2 # 尾部奇异值平方和
                    tail_rms = torch.sqrt(tail_fro_norm_sq / n_tail)
                    S_new = torch.full_like(S, fill_value=tail_rms)
                    # target_mean = tail_rms

                    # tail_var = tail_fro_norm_sq / n_tail - tail_rms ** 2
                    # tail_std = torch.sqrt(torch.clamp(tail_var, min=0.0))
                    # target_std = tail_std
                    
                    # if target_std < 1e-8:  # 尾部标准差太小，使用平滑值
                    #     target_std = tail_rms * 0.1
                    
                    # # 标准化头部奇异值
                    # S_mean = S.mean()
                    # S_std = S.std()
                    
                    # if S_std < 1e-8:  # 避免除零
                    #     S_std = S_mean * 0.1 + 1e-8
                    
                    # # 重新缩放以匹配目标分布
                    # S_normalized = (S - S_mean) / S_std
                    # S_new = S_normalized * target_std + target_mean
                    
                    # # 保持单调递减性（SVD性质）
                    # S_new = torch.sort(S_new, descending=True)[0]
                    
                    # # 确保最小值不低于尾部RMS的80%
                    # S_new = torch.clamp(S_new, min=tail_rms * 0.8, max=S.max())

                    new_G = tail + (U * S_new.unsqueeze(0)) @ V.t()
                    u = new_G.to(p.dtype, copy=True)

                # scale update
                norm = torch.linalg.matrix_norm(u).item()
                rms = norm / math.sqrt(u.numel())
                adjusted_lr = lr * 0.2 / rms

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
    
# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
    

class SGDMuon(torch.optim.Optimizer):
    """
    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # # calc update
                # state = self.state[p]
                # if "momentum_buffer" not in state:
                #     state["momentum_buffer"] = torch.zeros_like(g)
                # buf = state["momentum_buffer"]
                # buf.mul_(momentum).add_(g)
                # if group["nesterov"]:
                #     g = g.add(buf, alpha=momentum)
                # else:
                #     g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


class MomentumMuon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        orth_Momentum=True,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False
        self._orth_Momentum = orth_Momentum

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                if self._orth_Momentum:
                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                    p.data.mul_(1 - lr * wd)
                    p.data.add_(u, alpha=-adjusted_lr)
                else:
                    # scale update
                    norm = torch.linalg.matrix_norm(u).item()
                    rms = norm / math.sqrt(u.numel())
                    adjusted_lr = lr * 0.2 / rms
                    p.data.mul_(1 - lr * wd)
                    p.data.add_(u, alpha=-adjusted_lr)                    

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


class NewMuon_SM(torch.optim.Optimizer):
    
    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,         
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        orth_Momentum=True,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        
        for p in muon_params:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False
        self._orth_Momentum = orth_Momentum

    @staticmethod
    def _spectral_shaping(G: torch.Tensor, k_ratio: float = 0.015) -> torch.Tensor:
        m, n = G.shape
        k = int(max(1, round(k_ratio * min(m, n))))

        # fp32 计算更稳定
        G32 = G.to(dtype=torch.float32, copy=True)

        # 低秩 SVD
        U, S, V = torch.svd_lowrank(G32, q=k)   # U:[m,k], S:[k], V:[n,k]

        head = (U * S.unsqueeze(0)) @ V.t()     # [m,n]
        tail = G32 - head

        n_tail = max(1, min(m, n) - k)
        
        tail_fro_norm_sq = torch.norm(tail, p='fro') ** 2
        tail_rms = torch.sqrt(tail_fro_norm_sq / n_tail)
        
        S_new = torch.full_like(S, fill_value=tail_rms)

        new_G32 = tail + (U * S_new.unsqueeze(0)) @ V.t()
        return new_G32.to(dtype=G.dtype, device=G.device, copy=True)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            muon_params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]

            for p in muon_params:
                g = p.grad
                if g is None:
                    continue

                # 展平到 2D
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                M = state["momentum_buffer"]

                with torch.no_grad():
                    # -------- Step 1: 先对原始梯度做谱处理，再累积到动量 --------
                    device = p.device
                    orig_dtype = p.dtype

                    G_raw = g.detach().to(device=device, dtype=torch.float32)

                    G_proc = self._spectral_shaping(G_raw)

                    # 保持累积到动量上的 Frobenius 范数与原梯度一致
                    fro_raw = torch.linalg.matrix_norm(G_raw)
                    fro_proc = torch.linalg.matrix_norm(G_proc)
                    if fro_proc > 0:
                        G_proc = G_proc * (fro_raw / fro_proc)

                    # 动量累积：M_t = μ M_{t-1} + G_proc
                    M.mul_(momentum).add_(G_proc)

                    # Nesterov: 构造用来生成更新的“有效动量” M_eff
                    if nesterov:
                        # 经典 Nesterov 是 μ M + g，这里用处理后的梯度
                        M_eff = G_proc.add(M, alpha=momentum)
                    else:
                        M_eff = M

                    # # -------- Step 2: 再对动量/有效动量做一次谱处理，得到真正更新 --------
                    if self._orth_Momentum:
                        new_G = self._spectral_shaping(M_eff)
                        u = new_G.to(dtype=orig_dtype, device=device, copy=True)
                    else:
                        u = M_eff.to(dtype=orig_dtype, device=device, copy=True)

                    # 下面这块和你原来的 NewMuon 一样：根据 u 的 RMS 调整 lr
                    norm = torch.linalg.matrix_norm(u).item()
                    rms = norm / math.sqrt(u.numel())
                    # 目标 RMS ~ 0.2 * lr
                    adjusted_lr = lr * 0.2 / (rms + 1e-12)

                    # weight decay
                    p.data.mul_(1 - lr * wd)
                    # 参数更新
                    p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            adamw_params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in adamw_params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]

                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g_hat = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                scale = bias_correction1 / (bias_correction2 ** 0.5)

                # decoupled weight decay
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g_hat, alpha=-lr / scale)

        return loss


class Oron(torch.optim.Optimizer):

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False
            
    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr
    

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update - 修改为基于前一个正交化更新的动量
                state = self.state[p]
                
                # 新增：存储前一个正交化更新 O_{t-1}
                if "prev_orthogonal_update" not in state:
                    state["prev_orthogonal_update"] = torch.zeros_like(g)
                
                # 关键修改：B_t = μ * O_{t-1} + G_t
                prev_u = state["prev_orthogonal_update"]
                buf = momentum * prev_u + g  # 这里直接计算新的动量缓冲区
                
                # 如果使用Nesterov，需要特殊处理
                if group["nesterov"]:
                    g_for_orth = g.add(buf, alpha=momentum)
                else:
                    g_for_orth = buf
                
                # 计算当前正交化更新 O_t = NewtonSchulz(B_t)
                current_u = zeropower_via_newtonschulz5(g_for_orth, steps=group["ns_steps"])
                

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(current_u, alpha=-adjusted_lr)
                
                n_u = current_u.numel()
                n_g = g.numel()
                f_norm_u = torch.norm(current_u)
                f_norm_g = torch.norm(g)
                # RMS = F_norm / sqrt(N)
                rms_u = f_norm_u / (n_u ** 0.5)
                rms_g = f_norm_g / (n_g ** 0.5)
                state["prev_orthogonal_update"] = rms_g / rms_u * current_u

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
    

import math
import torch


class Newmuon_SM_iter(torch.optim.Optimizer):
    
    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,             # 先保留接口，暂时未使用
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        orth_Momentum=True,
        k_ratio: float = 0.015,   # 用于近似 SVD 的 rank 比例
        power_iters: int = 1,     # 幂迭代轮数
    ):
        if muon_params is None:
            muon_params = []
        if adamw_params is None:
            adamw_params = []

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params) + list(adamw_params)
        super().__init__(params, defaults)

        # 标记哪些参数走 Muon，哪些走 AdamW
        for p in muon_params:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

        self._orth_Momentum = orth_Momentum
        self._k_ratio = float(k_ratio)
        self._power_iters = max(1, int(power_iters))

    # ------------------------------------------------------------------
    #  幂迭代近似 SVD: G ≈ U S V^T，并在 state 中缓存右子空间 V
    # ------------------------------------------------------------------
    def _power_iter_svd(
        self,
        G32: torch.Tensor,   # [m, n], float32
        state: dict,
        mode: str,           # "grad" 或 "mom"
        k_ratio: float,
        eps: float = 1e-8,
    ):
        """
        用 Dion 风格幂迭代近似 G32 的 rank-k SVD:
            G ≈ U S V^T

        首次调用（或 rank 变化）时，用 torch.svd_lowrank 做初始化：
            U, S, V = svd_lowrank(G32, q=k)
        并将 V 缓存到 state 里作为后续 step 的初始子空间。

        返回:
            U : [m, k]
            S : [k]
            V : [n, k]
        """
        m, n = G32.shape
        k = int(max(1, round(k_ratio * min(m, n))))

        key = f"V_cache_{mode}"
        V_prev = state.get(key, None)

        # ---- 情况 1：没有历史 V，或者形状不匹配 —— 回退到精度较高的 svd_lowrank ----
        if V_prev is None or V_prev.shape != (n, k):
            # torch.svd_lowrank 直接给出近似 U,S,V
            U_svd, S_svd, V_svd = torch.svd_lowrank(G32, q=k)  # U:[m,k], S:[k], V:[n,k]
            # 缓存右子空间作为后续 step 的初始迭代子空间
            state[key] = V_svd.detach()
            return U_svd, S_svd, V_svd

        # ---- 情况 2：有历史 V，用 Dion 风格幂迭代 refine ----
        V_prev = V_prev.to(device=G32.device, dtype=G32.dtype)
        V = V_prev
        U = None
        W = None

        for it in range(self._power_iters):
            # P = G V,  shape [m,k]
            P = G32 @ V

            # U = Orthonormalize(P)，这里直接用 QR
            U, _ = torch.linalg.qr(P, mode="reduced")  # [m,k]

            # W = G^T U, shape [n,k]
            W = G32.mT @ U

            if it < self._power_iters - 1:
                # 为下一轮迭代更新 V：列归一化 W
                col_norm = torch.linalg.norm(W, dim=0, keepdim=True)  # [1,k]
                V = W / (col_norm + eps)

        # 最后一次迭代后，从 W 中提取 S, V
        S = torch.linalg.norm(W, dim=0)           # [k]
        V_new = W / (S.unsqueeze(0) + eps)        # [n,k]，列单位范数

        # 缓存右子空间，供下一步使用
        state[key] = V_new.detach()

        return U, S, V_new

    # ------------------------------------------------------------------
    #  谱整形：用幂迭代近似 SVD + 抹平 head 奇异值
    # ------------------------------------------------------------------
    def _spectral_shaping(
        self,
        G: torch.Tensor,       # 2D [m,n]
        state: dict,
        mode: str = "grad",    # "grad" 或 "mom"
        k_ratio: float = None,
    ) -> torch.Tensor:
        """
        用幂迭代近似 SVD，再做“tail + flatten(head singular values)”的谱整形。
        """
        if k_ratio is None:
            k_ratio = self._k_ratio

        m, n = G.shape
        G32 = G.to(dtype=torch.float32, copy=True)

        # 1) 幂迭代近似 SVD: G32 ≈ U S V^T
        U, S, V = self._power_iter_svd(
            G32,
            state=state,
            mode=mode,
            k_ratio=k_ratio,
        )  # U:[m,k], S:[k], V:[n,k]

        # 2) head / tail 分解
        head = (U * S.unsqueeze(0)) @ V.mT        # [m,n]
        tail = G32 - head

        n_tail = max(1, min(m, n) - S.numel())

        tail_fro_norm_sq = torch.norm(tail, p="fro") ** 2
        tail_rms = torch.sqrt(tail_fro_norm_sq / n_tail)

        # 3) 把 head 的奇异值抹平成 tail_rms
        S_new = torch.full_like(S, fill_value=tail_rms)
        new_G32 = tail + (U * S_new.unsqueeze(0)) @ V.mT

        return new_G32.to(dtype=G.dtype, device=G.device, copy=True)

    # ------------------------------------------------------------------
    #  step: Muon + AdamW backup
    # ------------------------------------------------------------------
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            # =========================
            #          Muon
            # =========================
            muon_params = [p for p in group["params"] if self.state[p]["use_muon"]]

            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]

            for p in muon_params:
                g = p.grad
                if g is None:
                    continue

                # 假设 muon_params 全是 2D 权重矩阵
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                M = state["momentum_buffer"]

                with torch.no_grad():
                    device = p.device
                    orig_dtype = p.dtype

                    # -------- Step 1: 对原始梯度做谱整形，再累积到动量 --------
                    G_raw = g.detach().to(device=device, dtype=torch.float32)

                    # 使用 "grad" 通道的谱子空间
                    G_proc = self._spectral_shaping(G_raw, state, mode="grad")

                    # 保持累积到动量上的 Frobenius 范数与原梯度一致
                    fro_raw = torch.linalg.matrix_norm(G_raw)
                    fro_proc = torch.linalg.matrix_norm(G_proc)
                    if fro_proc > 0:
                        G_proc = G_proc * (fro_raw / fro_proc)

                    # 动量累积：M_t = μ M_{t-1} + G_proc
                    M.mul_(momentum).add_(G_proc)

                    # Nesterov: 构造有效动量 M_eff
                    if nesterov:
                        # 经典 Nesterov 是 μ M + g，这里使用整形后的 G_proc
                        M_eff = G_proc.add(M, alpha=momentum)
                    else:
                        M_eff = M

                    # -------- Step 2: （可选）对 M_eff 再做一次谱整形 --------
                    if self._orth_Momentum:
                        # 使用 "mom" 通道的谱子空间
                        new_G = self._spectral_shaping(M_eff, state, mode="mom")
                        u = new_G.to(dtype=orig_dtype, device=device, copy=True)
                    else:
                        u = M_eff.to(dtype=orig_dtype, device=device, copy=True)

                    # -------- Step 3: RMS 归一化调节 lr + 更新参数 --------
                    norm = torch.linalg.matrix_norm(u).item()
                    rms = norm / math.sqrt(u.numel())
                    adjusted_lr = lr * 0.2 / (rms + 1e-12)

                    # decoupled weight decay
                    p.data.mul_(1 - lr * wd)
                    p.data.add_(u, alpha=-adjusted_lr)

            # =========================
            #       AdamW backup
            # =========================
            adamw_params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in adamw_params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)

                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]

                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g_hat = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                scale = bias_correction1 / (bias_correction2 ** 0.5)

                # decoupled weight decay
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g_hat, alpha=-lr / scale)

        return loss
