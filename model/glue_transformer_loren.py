import math
import time
from typing import Optional

import numpy as np
import torch

from .glue_transformer_lozo import GLUETransformerLoZo
from common.alg_utils import sgd_with_wdecay


class GLUETransformerLOREN(GLUETransformerLoZo):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        zero_order_eps: float = 1e-3,
        learning_rate_aux: float = 1e-3,
        minibatch: int = 64,
        max_norm: float = 18000.0,
        z_std: float = 1.0,
        lr_anneal: float = 1.0,
        full_parameter: bool = True,
        batchsize_limit: int = 16,
        eval_splits: Optional[list] = None,
        soft_prompt: bool = False,
        use_SGD: bool = True,
        logger_type: str = 'wandb',
        hf_token: Optional[str] = None,
        lora: bool = False,
        model_init_seed: Optional[int] = None,
        weight_decay: float = 0.01,
        damping: float = 0.1,
        lr_cov: float = 2e-3,
        beta1: float = 0.9,
        num_queries: int = 10,
        **kwargs,
    ):
        super().__init__(model_name_or_path,
                         num_labels,
                         zero_order_eps,
                         learning_rate_aux,
                         minibatch,
                         max_norm,
                         z_std,
                         lr_anneal,
                         full_parameter,
                         batchsize_limit,
                         eval_splits,
                         soft_prompt,
                         use_SGD,
                         logger_type,
                         hf_token,
                         lora,
                         model_init_seed,
                         weight_decay)

        self.damping = damping
        self.lr_cov = lr_cov
        self.d_sqrt = math.sqrt(damping)
        self.evec_a = {}
        self.momentum = {}
        self.num_queries = num_queries
        self.beta1 = beta1
        self.param_info = {}

        assert num_queries >= 2, 'At least 2 samples are required for RLOO estimation'

    def configure_params(self):
        super().configure_params()
        for name, param in self.params:
            param_shape = param.shape
            param_flat_size = param.numel()
            evec_size = param.size(1) if param.ndim >= 2 else param_flat_size
            a = torch.randn(evec_size, device=param.device, dtype=param.dtype)
            self.evec_a[name] = a
            if param.ndim >= 2:
                self.momentum[name] = torch.zeros_like(param)
            self.param_info[name] = {
                'shape': param_shape,
                'flat_size': param_flat_size,
                'evec_size': evec_size,
                'mfactor': param_flat_size // evec_size
            }

    @torch.no_grad()
    def perturb_parameters(self, random_seed: int, scaling_factor=1):
        torch.manual_seed(random_seed)
        for name, param in self.params:
            info = self.param_info[name]
            shape = info['shape']
            a = self.evec_a[name]
            a_flat = a.view(1, -1).expand(info['mfactor'], -1).reshape(-1)
            u_vec = torch.randn_like(param).view(-1)
            sq_norm_a = torch.dot(a_flat, a_flat)
            dot_au = torch.dot(a_flat, u_vec)
            damping_sq_norm_a = math.sqrt(self.damping + sq_norm_a.item())
            alpha = (self.d_sqrt + damping_sq_norm_a) * dot_au / (sq_norm_a * damping_sq_norm_a)
            sqrt_cov_u = u_vec.sub(alpha * a_flat).view(shape)
            param.add_(sqrt_cov_u, alpha=scaling_factor * self.zero_order_eps)

    @torch.no_grad()
    def lnes_zo_update(self, random_seeds, fvals, fmean):
        fweight = torch.tensor(fvals, device='cuda') - fmean
        num_queries_inv = 1.0 / (self.num_queries - 1)
        eps_inv = 1.0 / (self.zero_order_eps * (self.num_queries - 1))

        for k in range(self.num_queries):
            torch.manual_seed(random_seeds[k])
            for name, param in self.params:
                info = self.param_info[name]
                shape = info['shape']
                a = self.evec_a[name]
                a_flat = a.view(1, -1).expand(info['mfactor'], -1).reshape(-1)
                u_vec = torch.randn_like(param).view(-1)
                sq_norm_a = torch.dot(a_flat, a_flat)
                dot_au = torch.dot(a_flat, u_vec)
                damping_sq_norm_a = math.sqrt(self.damping + sq_norm_a.item())

                alpha = (self.d_sqrt + damping_sq_norm_a) * dot_au / (sq_norm_a * damping_sq_norm_a)
                sqrt_cov_u = u_vec.sub(alpha * a_flat).view(shape)

                gx = sqrt_cov_u.mul(fweight[k] * eps_inv)
                if param.ndim >= 2:
                    self.momentum[name].mul_(self.beta1).add_(gx)
                    gx = self.momentum[name]
                sgd_with_wdecay(param, gx, self.state.learning_rate, self.weight_decay)

                c1 = (dot_au * u_vec - a_flat)
                c2 = (self.d_sqrt + damping_sq_norm_a) * (dot_au ** 2 - sq_norm_a) * a_flat / (sq_norm_a * damping_sq_norm_a)
                ak_grad = fweight[k] * (self.zero_order_eps ** 2) * (c1 - c2) / damping_sq_norm_a
                ak_grad = ak_grad.view(shape).sum(0) * num_queries_inv
                sgd_with_wdecay(self.evec_a[name], ak_grad, self.lr_cov, self.weight_decay)

    def training_step(self, model, batch, tb_writer=None, **kwargs):
        n = batch['input_ids'].size(0)
        start_time = time.perf_counter()
        random_seeds = [np.random.randint(1000000000) for _ in range(self.num_queries)]
        log_dict = kwargs['log']

        fvals = []
        with torch.inference_mode():
            for k in range(self.num_queries):
                self.perturb_parameters(random_seeds[k], scaling_factor=1)
                loss = self.zo_forward_memory_eff(model, batch)
                fvals.append(loss.item())
                self.perturb_parameters(random_seeds[k], scaling_factor=-1)

        fmean = np.mean(fvals)
        self.lnes_zo_update(random_seeds, fvals, fmean)
        self.state.global_training_steps += 1

        total_time = time.perf_counter() - start_time

        self.state.time.append(total_time)
        self.state.query.append(self.num_queries * n)

        log_dict['train_loss'] = fmean.item()
        log_dict['time'] = total_time
