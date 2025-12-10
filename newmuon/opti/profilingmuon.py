import math
import torch
from Metis.lowrank_eig import svd_lowrank_eig_graph_pipelined


class ProfilingNewMuon(torch.optim.Optimizer):
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
        profiling_step=-1
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
        self._profiling_step = profiling_step

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
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1
                step = state["step"]
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                
                ### profiliing ###
                if step % self._profiling_step == 0:
                    U1, S1, V1h = torch.linalg.svd(g, full_matrices=True)
                    # S1 = S1[:32]
                    energy = S1 ** 2
                    total_energy = torch.sum(energy)
                    explained_variance_ratio = energy / total_energy
                    cumulative_energy = torch.cumsum(explained_variance_ratio, dim=0)
                    for i in range(32):
                        print(f"{i}: {cumulative_energy[i].item() * 100:.2f}%", end="\t")
                    print()
                    print(S1.tolist())
                ### profiliing ###
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


class ProfilingMuon(torch.optim.Optimizer):
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
        profiling_step=-1,
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
        self._profiling_step = profiling_step

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
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1
                step = state["step"]
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                ### profiliing ###
                if step % self._profiling_step == 0:
                    U1, S1, V1h = torch.linalg.svd(g, full_matrices=True)
                    # S1 = S1[:32]
                    energy = S1 ** 2
                    total_energy = torch.sum(energy)
                    explained_variance_ratio = energy / total_energy
                    cumulative_energy = torch.cumsum(explained_variance_ratio, dim=0)
                    for i in range(32):
                        print(f"{i}: {cumulative_energy[i].item() * 100:.2f}%", end="\t")
                    print()
                    print(S1.tolist())
                ### profiliing ###
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

class ProfilingAdam(torch.optim.Optimizer):
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
        profiling_step=-1,
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
            assert p.ndim == 2, "Muon expects 2D param matrices"
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False
        self._profiling_step = profiling_step

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None, target_f_norms=None):
        """
        target_f_norms: list aligned with muon_params; if >0, scale u to have ||u||_F == target
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            muon_params = [p for p in group["params"] if self.state[p]["use_muon"]]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]
            lr = group["lr"]

            for p in muon_params:
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
                ### profiliing ###
                if step % self._profiling_step == 0:
                    print("first order")
                    U1, S1, V1h = torch.linalg.svd(buf1, full_matrices=True)
                    # S1 = S1[:32]
                    energy = S1 ** 2
                    total_energy = torch.sum(energy)
                    explained_variance_ratio = energy / total_energy
                    cumulative_energy = torch.cumsum(explained_variance_ratio, dim=0)
                    for i in range(32):
                        print(f"{i}: {cumulative_energy[i].item() * 100:.2f}%", end="\t")
                    print()
                    print(S1.tolist())
                    print("second order")
                    U1, S1, V1h = torch.linalg.svd(buf2, full_matrices=True)
                    # S1 = S1[:32]
                    energy = S1 ** 2
                    total_energy = torch.sum(energy)
                    explained_variance_ratio = energy / total_energy
                    cumulative_energy = torch.cumsum(explained_variance_ratio, dim=0)
                    for i in range(32):
                        print(f"{i}: {cumulative_energy[i].item() * 100:.2f}%", end="\t")
                    print()
                    print(S1.tolist())
                ### profiliing ###

                ghat = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / (bias_correction2**0.5 + 1e-12)
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(ghat, alpha=-lr / scale)
            
            # muon_params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # lr = group["lr"]
            # wd = group["wd"]
            # momentum = group["momentum"]

            # if target_f_norms is not None:
            #     if len(target_f_norms) != len(muon_params):
            #         raise ValueError("target_f_norms length mismatch with muon params")

            # for idx, p in enumerate(muon_params):
            #     g = p.grad
            #     if g is None:
            #         continue
            #     if g.ndim > 2:
            #         g = g.view(g.size(0), -1)
            #     state = self.state[p]
            #     if "momentum_buffer" not in state:
            #         state["momentum_buffer"] = torch.zeros_like(g)
            #     buf = state["momentum_buffer"]
            #     buf.mul_(momentum).add_(g)
            #     if group["nesterov"]:
            #         g_eff = g.add(buf, alpha=momentum)
            #     else:
            #         g_eff = buf

            #     u = zeropower_via_newtonschulz5(g_eff, steps=group["ns_steps"])

            #     adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
            #     u *= -adjusted_lr

            #     if target_f_norms is not None:
            #         target = target_f_norms[idx]
            #         if target is not None and target > 0:
            #             cur_f = float(torch.linalg.norm(u, ord='fro'))
            #             if cur_f > 1e-12:
            #                 scale = target / (cur_f + 1e-12)
            #                 u = u * scale

            #     p.data.mul_(1 - lr * wd)
            #     p.data.add_(u)

            adamw_params = [p for p in group["params"] if not self.state[p]["use_muon"]]
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

                ghat = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / (bias_correction2**0.5 + 1e-12)
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(ghat, alpha=-lr / scale)
