import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import transformer_engine.pytorch as te

from utils import Tokenized_data
from models import GPT
from torch.utils.data import DataLoader
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from Metis import BitLinear
from utils import parse
from transformer_engine.common.recipe import Format, DelayedScaling
import torch.distributed as dist

from opti import NewMuon, Muon, Dion, ProfilingDion, ProfilingMuon, ProfilingNewMuon, ProfilingAdam, ProfilingMyDion, Myopt, MomentumMuon, NewMuon_SM, SGDMuon, Oron, Newmuon_SM_iter

def load_model(args):
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl') 
        args.device = torch.device("cuda", args.local_rank)
    
    model = GPT(args)
    
    if args.load_from:
        state_dict = torch.load(args.load_from)

        try:
            model.load_state_dict(state_dict)
        except:
            for m in model.modules():
                if isinstance(m, BitLinear):
                    m.split()
            model.load_state_dict(state_dict)
            
            
        if args.local_rank == 0:
            print(f"model loaded from {args.load_from}")

    if args.local_rank == 0:
        print(f'Model ok on device {args.device}. params: {sum(p.numel() for p in model.parameters())}')
        print("******** Model Parameters *******")
        for name, p in model.named_parameters():
            print(name, p.shape)
        print("*********************************")
    return model

def load_dataset(args):
    dataset = Tokenized_data(args)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=args.shuffle, 
        num_workers=args.dataset_workers
    )
    print(f'Data ok on device {args.device}.')

    return dataloader


def print_model_structure(model, show_parameters=True):
    """
    打印 model 的层级结构、module 类型、参数数量（总/训练）以及 module 名称（带缩进体现层级）
    """
    def num_params(module):
        total = sum(p.numel() for p in module.parameters(recurse=False))
        trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        return total, trainable

    for name, module in model.named_modules():
        # 计算缩进等级（模块名按 '.' 分层）
        indent = "  " * (0 if name == "" else name.count(".") + 1)
        t_total, t_train = num_params(module)
        if show_parameters:
            print(f"{indent}{name or '<root>'}: {module.__class__.__name__}  | params (this module only): total={t_total}, trainable={t_train}")
        else:
            print(f"{indent}{name or '<root>'}: {module.__class__.__name__}")

def get_optimizer(args, model):
    if args.optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            betas=(args.adam_beta1, args.adam_beta2), 
            eps=1e-8, 
            weight_decay=args.weight_decay
        ) 
    elif args.optimizer_name == "newmuon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "emb" not in name and "fc" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "emb" not in name and "fc" not in name
            )
        ]

        return NewMuon(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=(args.adam_beta1, args.adam_beta2)
        )
    elif args.optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "emb" not in name and "fc" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "emb" not in name and "fc" not in name
            )
        ]

        return Muon(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=(args.adam_beta1, args.adam_beta2)
        )
    elif args.optimizer_name == "momentummuon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "emb" not in name and "fc" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "emb" not in name and "fc" not in name
            )
        ]

        return MomentumMuon(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=(args.adam_beta1, args.adam_beta2),
            orth_Momentum=args.orthM
        )
    elif args.optimizer_name == "sgdmuon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "emb" not in name and "fc" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "emb" not in name and "fc" not in name
            )
        ]

        return SGDMuon(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=(args.adam_beta1, args.adam_beta2),
        )
    elif args.optimizer_name == "newmuon_sm":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "emb" not in name and "fc" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "emb" not in name and "fc" not in name
            )
        ]

        return NewMuon_SM(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=(args.adam_beta1, args.adam_beta2),
            orth_Momentum=args.orthM
        )
    elif args.optimizer_name == "newmuon_sm_iter":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "emb" not in name and "fc" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "emb" not in name and "fc" not in name
            )
        ]

        return Newmuon_SM_iter(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=(args.adam_beta1, args.adam_beta2),
            orth_Momentum=args.orthM,
            power_iters=args.power_iters
        )
        
    elif args.optimizer_name == "dion":
        dion_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "emb" not in name and "fc" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "emb" not in name and "fc" not in name
            )
        ]
        param_groups = [
            dict(params=dion_params, algorithm="dion"),
            dict(params=adamw_params, algorithm="adamw")
        ]

        return Dion(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            rank_fraction=args.rf,
            prob=args.prob,
            iters=args.dion_iters
        )
    elif args.optimizer_name == "myopt":
        dion_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "emb" not in name and "fc" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "emb" not in name and "fc" not in name
            )
        ]
        param_groups = [
            dict(params=dion_params, algorithm="dion"),
            dict(params=adamw_params, algorithm="adamw")
        ]

        return Myopt(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            rank_fraction=args.rf,
            prob=args.prob
        )
    elif args.optimizer_name == "oron":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "emb" not in name and "fc" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "emb" not in name and "fc" not in name
            )
        ]

        return Oron(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=(args.adam_beta1, args.adam_beta2)
        )
    elif args.optimizer_name == "profilingdion":
        dion_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "emb" not in name and "fc" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "emb" not in name and "fc" not in name
            )
        ]
        param_groups = [
            dict(params=dion_params, algorithm="dion"),
            dict(params=adamw_params, algorithm="adamw")
        ]

        return ProfilingDion(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            rank_fraction=args.rf,
            profiling_step=1000,
            iters=args.dion_iters
        )
    elif args.optimizer_name == "profilingmuon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "emb" not in name and "fc" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "emb" not in name and "fc" not in name
            )
        ]

        return ProfilingMuon(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=(args.adam_beta1, args.adam_beta2),
            profiling_step=1000,
        )
    elif args.optimizer_name == "profilingnewmuon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "emb" not in name and "fc" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "emb" not in name and "fc" not in name
            )
        ]

        return ProfilingNewMuon(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=(args.adam_beta1, args.adam_beta2),
            profiling_step=1000,
        )
    elif args.optimizer_name == "profilingadam":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "emb" not in name and "fc" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "emb" not in name and "fc" not in name
            )
        ]

        return ProfilingAdam(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=(args.adam_beta1, args.adam_beta2),
            profiling_step=1000,
        )
    elif args.optimizer_name == "profilingmydion":
        dion_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "emb" not in name and "fc" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "emb" not in name and "fc" not in name
            )
        ]
        param_groups = [
            dict(params=dion_params, algorithm="dion"),
            dict(params=adamw_params, algorithm="adamw")
        ]

        return ProfilingMyDion(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            rank_fraction=args.rf,
            prob=args.prob,
            profiling_step=1000,
            iters=args.dion_iters
        )
    else:
        assert 0, "optimizer not supported"
        

def train(args):
    if args.local_rank <= 0:
        writer = SummaryWriter(f"{args.log_dir}/{args.tag}")

    dataloader = load_dataset(args)
    model = load_model(args)

    model.train()
    
    if args.orth_initialization:
        for m in model.modules():
            if isinstance(m, BitLinear):
                m.orth_initialization()

    loss_fn = nn.CrossEntropyLoss(ignore_index = args.vocab_size - 1)
    optimizer = get_optimizer(args, model) 
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.lr_warmup_steps, args.train_steps)

    train_steps = 0
    acc_steps = 1
    acc_loss = 0
    
    for epoch in range(args.max_epochs):   
          
        for batch, (source, target, _) in enumerate(dataloader):

            if args.enable_forward_svd and batch >= args.forward_svd_warmup_steps and acc_steps == 1:
                if batch == args.forward_svd_warmup_steps or \
                   (args.forward_svd_merge_steps > 0 and (batch - args.forward_svd_warmup_steps) % args.forward_svd_merge_steps == 0):
                    print("split")
                    for m in model.modules():
                        if isinstance(m, BitLinear):
                            m.split()
                    
                    optimizer = get_optimizer(args, model) 
                    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.lr_warmup_steps, args.train_steps)
            
            source, target = source.to(args.device), target.to(args.device)
            
            if acc_steps == 1:
                optimizer.zero_grad()

            if args.model == "gpt":
                output = model(source)
                loss = loss_fn(output.view(-1, args.vocab_size), target.view(-1)) / args.grad_acc
            else:
                output, loss = model(source, targets=target)
                loss /= args.grad_acc
            acc_loss += loss.item()
        
            if acc_steps == args.grad_acc:
                if args.local_rank <= 0:
                    writer.add_scalar("train_loss", acc_loss, train_steps)
                
                # regularization dp only
                rloss = torch.zeros_like(loss)
                if args.reg_lambda > 0:
                    for name, p in model.decoders.named_parameters():
                        if ("ulinear" in name or "vlinear" in name or (not ("ln" in name))) and "weight" in name:
                            rloss += (torch.sum(p ** 2) * args.reg_alpha1 + \
                                    torch.sum(((p + 1e-6) ** -2) * args.reg_alpha2)) * \
                                    (1 / p.shape[0] / p.shape[1] * args.reg_lambda) 
                loss += rloss
                loss.backward()
                
            else:
                loss.backward()
                acc_steps += 1
                continue
            

            if args.local_rank <= 0:
                print(f"rank: {args.local_rank}, "
                    f"epoch: {epoch}, "
                    f"batch: {train_steps}, "
                    f"loss: {acc_loss:.3f}, "
                    f"r-loss: {rloss.item() + acc_loss:.3f}"
                    )
            # f"grad_norm: {g}"
            
            g = 0
            for name, p in model.named_parameters():
                if not (p.grad is None):
                    g += p.grad.norm().item()
            clip_thres = 1 if args.grad_clipping > g else args.grad_clipping / g
            for name, p in model.named_parameters():
                if not (p.grad is None):
                    p.grad *= clip_thres

            optimizer.step()
            if args.enable_scheduler:
                lr_scheduler.step()
            
            
            torch.cuda.synchronize()  

            if train_steps % args.save_steps == 0 and train_steps > 0 and args.local_rank <= 0:
                torch.save(model.state_dict(), f"{args.chkpt_dir}/{args.tag}/model_{epoch}_{train_steps}.pth")
                torch.save(optimizer.state_dict(), f"{args.chkpt_dir}/{args.tag}/optimizer_{epoch}_{train_steps}.pth")
                torch.save(lr_scheduler.state_dict(), f"{args.chkpt_dir}/{args.tag}/lr_scheduler_{epoch}_{train_steps}.pth")
                print(f"model saved at {args.chkpt_dir}/{args.tag}/{epoch}_{train_steps}.pth")
            
            acc_loss = 0
            acc_steps = 1
            # if args.train_steps == batch + 1:
            if args.train_steps == train_steps + 1:
                break
            train_steps += 1


if __name__ == "__main__":
    args = parse()
    train(args)
