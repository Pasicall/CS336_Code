import argparse
import os 
import torch
import numpy as np
import wandb
from cs336_basics.nn import TransformerLM
from cs336_basics.optimizer import AdamW,clip_gradient_norm
from cs336_basics.checkpointing import save_checkpoint,load_checkpoint
from cs336_basics.scheduler import get_lr_cosine_scheduler
from cs336_basics.data import get_batch
from cs336_basics.losses import cross_entropy



def load_dateset(path):
    return np.memmap(
        path, 
        dtype=np.uint16,
        mode='r'
    )

def main():
# ===================================================================
# configuration
# ===================================================================
    parser = argparse.ArgumentParser()
#  Data
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)


# Model
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)


# Ablation
    parser.add_argument("--no_rms_norm", action="store_true", help="Disable RMSNorm")
    parser.add_argument("--norm_mode", type=str, default="pre", choices=["pre","post"], help="Normalization placement")
    parser.add_argument("--no_rope", action="store_true", help="Disable RoPE")
    parser.add_argument("--ffn_type", type=str, default="swiglu", choices=["swiglu","silu"], help="Type of Feedforward Network")

# Optimizer
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)

# Scheduler
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--cosine_cycle_iters", type=int, default=10000)

# Training
    parser.add_argument("--max_iters", type=int, default=10000)

# Gradient clipping
    parser.add_argument("--max_norm", type=float, default=1.0)

# Evaluation
    parser.add_argument("--eval_interval", type=int, default=100)

# Logging
    parser.add_argument("--log_interval", type=int, default=100)

# Path and system
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

# WandB
    parser.add_argument("--run_name", type=str, default=None, help="WandB实验名称")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exit_ok=True)

    # load dataset
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training dataset not found at {args.train_data_path}")
    if not os.path.exists(args.val_data_path):
        raise FileNotFoundError(f"Training dataset not found at {args.val_data_path}")
    train_data = load_dateset(args.train_data_path)
    val_data = load_dateset(args.val_data_path)
    print(f"训练集大小{len(train_data)}tokens")
    print(f"验证集大小{len(val_data)}tokens")

    # create model
    actual_rope_theta = None if args.no_rope else args.rope_theta
    use_rms_norm = not args.no_rms_norm

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        rope_theta=actual_rope_theta,
        device=args.device,
        use_rms_norm=use_rms_norm,
        norm_mode=args.norm_mode,
        ffn_type=args.ffn_type
    ).to(args.device)

    print(f"Model Config: Norm={args.norm_mode}, Use_norm={use_rms_norm}, FFN={args.ffn_type}, RoPE={not args.no_rope}")

    # create optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train Loop
    # load checkpoint 
    start_iter = 0
    ckpt_path = os.path.join(args.out_dir,"ckpt.pt")
    if os.path.exists(ckpt_path):
        start_iter = load_checkpoint(ckpt_path,model,optimizer)
        print(f"Resuming from iteration {start_iter}")

    # WandB init
    wandb.init(
        project="cs336_assignment1",
        name=args.run_name,
        config=args
    )

    for it in range(start_iter,args.max_iters):
        optimizer.zero_grad()

        # 1. update learning rate and  get batch
        lr = get_lr_cosine_scheduler(it,args.lr,args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        x, y = get_batch(
            dataset=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device
        )

        # 2. forward
        logits = model(x)

        # 3. compute loss
        loss = cross_entropy(logits=logits,targets=y)

        # 4. backward
        loss.backward()

        # 5. gradient clipping
        clip_gradient_norm(parameters=model.parameters(), max_norm=args.max_norm)

        # 6. optimizer step
        optimizer.step()

        # 8. evaluation and logging
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(dataset=val_data, batch_size=args.batch_size, context_length=args.context_length, device=args.device)
                v_logits = model(vx)
                v_loss = cross_entropy(v_logits, vy)
                print(f"Iter{it}: train_loss {loss.item():.4f}, val_loss {v_loss.item():.4f}, lr {lr:.2e}")
                wandb.log({
                    "train/loss": loss.item(),
                    "val/loss": v_loss.item(),
                    "lr": lr,
                    "iter": it + 1
                })
            model.train()

        # 9. save checkpoint
        if it % 1000 == 0 and it > 0:
            save_checkpoint(model=model,optimizer=optimizer,iteration=it,out=ckpt_path)

# 训练结束保存最终模型
    save_checkpoint(model=model,optimizer=optimizer,iteration=args.max_iters,out=os.path.join(args.out_dir,"ckpt_final.pt"))
    wandb.finish()


if __name__ == "__main__":
    main()