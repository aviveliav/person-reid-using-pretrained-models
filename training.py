
import os, time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.wait = 0
        self.stop = False

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop = True


@torch.no_grad()
def evaluate(
    extractor: nn.Module,
    comparator: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Tuple[float, float]:
    """
    Evaluate extractor+comparator on a dataset loader.
    """
    extractor.eval()
    comparator.eval()
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    total_loss, total_correct, total = 0.0, 0, 0

    for a, b, y in loader:
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)

        if a.dim() == 4: # a is an image
            f1 = extractor(a)
            f2 = extractor(b)
        else:
            f1, f2 = a, b
        # Normalize the features
        f1 = F.normalize(f1, dim=1).float()
        f2 = F.normalize(f2, dim=1).float()

        logits = comparator(f1, f2)
        loss = criterion(logits, y)

        total_loss   += loss.item() * y.size(0)
        total_correct += (torch.sigmoid(logits).round() == y).sum().item()
        total        += y.size(0)

    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    return avg_loss, acc

def train(
    extractor: nn.Module,
    comparator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    # training params
    epochs: int = 5,
    lr: float = 1e-4,
    early_stopping_flag: bool = False,
    # naming/saving
    dataset_tag: str = "cuhk03",
    extractor_name: str = "resnet50",
    comparator_name: str = "lowrank_bilinear",
):
    """
    Train comparator on train_loader, validate on val_loader.
    """

    # Ensure models on device
    extractor.to(device)
    comparator.to(device)

    # Freeze extractor
    extractor.train(False)
    for p in extractor.parameters():
        p.requires_grad_(False)

    # Saving paths and names
    save_dir = f"{extractor_name}_{comparator_name}"
    os.makedirs(save_dir, exist_ok=True)
    run_tag   = f"{dataset_tag}_{extractor_name}_{comparator_name}"
    ckpt_path = os.path.join(save_dir, f"{run_tag}.pt")
    plot_path = os.path.join(save_dir, f"loss_curve_{run_tag}.png")

    # Optimizer & loss
    optimizer = torch.optim.AdamW(list(comparator.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=1e-3)
    
    if early_stopping_flag:
        early_stopping = EarlyStopping()

    criterion = nn.BCEWithLogitsLoss()

    history = {
        "train_losses": [],
        "val_losses": [],
        "val_accs": []
    }
    best_val_loss = float("inf")
    best_epoch    = -1

    start = time.time()
    for epoch in range(1, epochs + 1):
        comparator.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0

        for a, b, y in train_loader:
            a = a.to(device, non_blocking=True)
            b = b.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if a.dim() == 4: # a is an image
                f1 = extractor(a)
                f2 = extractor(b)
            else:
                f1, f2 = a, b
            # Normalize the features
            f1 = F.normalize(f1, dim=1).float()
            f2 = F.normalize(f2, dim=1).float()

            logits = comparator(f1, f2)
            loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(comparator.parameters(), 1.0)
            optimizer.step()

            ep_loss    += loss.item() * y.size(0)
            ep_correct += (torch.sigmoid(logits).round() == y).sum().item()
            ep_total   += y.size(0)

        train_loss = ep_loss / max(1, ep_total)
        val_loss, val_acc = evaluate(extractor, comparator, val_loader, device, criterion)
        scheduler.step(val_loss) # LR auto-adjusts if no improvement

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["val_accs"].append(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} (acc {val_acc:.3f}) | lr {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_state    = {k: v.detach().cpu().clone() for k, v in comparator.state_dict().items()}

        if early_stopping_flag:
            early_stopping.step(val_loss)
            if early_stopping.stop:
                print(f"Early stopping at epoch {epoch}")
                break

    # restore best weights before saving
    if best_state:
        comparator.load_state_dict(best_state, strict=True)

    # Save best model and history
    save_obj = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "dataset": dataset_tag,
        "extractor_name": extractor_name,
        "comparator_name": comparator_name,
        "comparator_state": best_state,
        "history": history
    }
    torch.save(save_obj, ckpt_path)

    # Plot losses
    fig, ax = plt.subplots(figsize=(6,4))
    xs = list(range(1, len(history["train_losses"])+1))
    ax.plot(xs, history["train_losses"], label="Train Loss")
    ax.plot(xs, history["val_losses"], label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss\n{run_tag}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s | Best val loss = {best_val_loss:.4f} (epoch {best_epoch})")
    print("Saved best model & history to: ", ckpt_path)
    print("Saved plot to: ", plot_path)

    return best_val_loss, comparator, fig
