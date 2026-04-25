"""
training/train.py

Training script for both static (letters) and dynamic (words) models.
Supports mixed precision, learning rate scheduling, checkpointing, and metrics.

Usage:
    python -m training.train --task letters --epochs 50
    python -m training.train --task words --epochs 100 --lr 3e-4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from rich.console import Console
from rich.progress import track
from rich.table import Table

from data.dataset import LettersDataset, WordsDataset
from models.architectures import StaticGestureNet, DynamicGestureNet


console = Console()


def get_dataset_and_model(task: str, data_root: str | None):
    if task == "letters":
        ds = LettersDataset(data_root=data_root, augment=True)
        model = StaticGestureNet(num_classes=26)
        return ds, model, 26
    elif task == "words":
        ds = WordsDataset(data_root=data_root, augment=True)
        model = DynamicGestureNet(num_classes=len(ds.vocab))
        return ds, model, len(ds.vocab)
    else:
        raise ValueError(f"Unknown task: {task}")


def evaluate(model, loader, device, criterion):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    per_class_correct = {}
    per_class_total = {}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)

            for label, pred in zip(y.tolist(), preds.tolist()):
                per_class_total[label] = per_class_total.get(label, 0) + 1
                if label == pred:
                    per_class_correct[label] = per_class_correct.get(label, 0) + 1

    acc = correct / total if total else 0
    avg_loss = total_loss / total if total else 0
    return acc, avg_loss, per_class_correct, per_class_total


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type=device.type):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["letters", "words"], required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    console.print(f"[bold]Training {args.task} model on {device}[/bold]")

    ds, model, num_classes = get_dataset_and_model(args.task, args.data_root)
    val_size = int(len(ds) * args.val_split)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2, pin_memory=True)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"Model: {model.__class__.__name__}  |  Params: {n_params:,}  |  Classes: {num_classes}")
    console.print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    table = Table(title=f"Training: {args.task}", show_header=True, header_style="bold cyan")
    for col in ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "LR", "Time (s)"]:
        table.add_column(col, justify="right")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_acc, val_loss, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        elapsed = time.time() - start

        table.add_row(
            str(epoch),
            f"{train_loss:.4f}",
            f"{train_acc:.3f}",
            f"{val_loss:.4f}",
            f"{val_acc:.3f}",
            f"{scheduler.get_last_lr()[0]:.2e}",
            f"{elapsed:.1f}",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = Path(args.checkpoint_dir) / f"{args.task}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
                "num_classes": num_classes,
            }, ckpt_path)

        # Print incrementally
        console.clear()
        console.print(table)
        console.print(f"[dim]Best val acc: {best_val_acc:.4f}[/dim]")

    console.print(f"\n[bold green]Done. Best val acc: {best_val_acc:.4f}[/bold green]")
    console.print(f"Saved to {args.checkpoint_dir}/{args.task}_best.pt")


if __name__ == "__main__":
    main()
