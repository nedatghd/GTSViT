import os
import numpy as np
from tqdm import tqdm
import torch
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.utils import grouper, sliding_window, count_sliding_window


def train(network, optimizer, criterion, train_loader, val_loader, epoch, saving_path, device,
          scheduler=None, wandb_run=None):

    best_acc = -0.1

    # ── history buffers ────────────────────────────────────────────────────────
    history = {
        "train_loss": [],
        "train_acc":  [],
        "val_loss":   [],
        "val_acc":    [],
    }

    for e in tqdm(range(1, epoch + 1), desc="Training"):
        network.train()

        epoch_losses = []
        num_correct_train = 0
        total_train = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = network(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            # train accuracy
            _, preds = torch.max(outputs, dim=1)
            num_correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        # ── per-epoch metrics ──────────────────────────────────────────────────
        mean_train_loss = float(np.mean(epoch_losses))
        train_acc = num_correct_train / total_train

        val_acc, mean_val_loss = validation(network, val_loader, criterion, device)

        history["train_loss"].append(mean_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(mean_val_loss)
        history["val_acc"].append(val_acc)

        # ── console log every 10 epochs ───────────────────────────────────────
        if e % 10 == 0 or e == 1:
            tqdm.write(
                f"Epoch [{e}/{epoch}]  "
                f"train_loss={mean_train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={mean_val_loss:.4f}  val_acc={val_acc:.4f}"
            )

        # ── W&B logging ───────────────────────────────────────────────────────
        if wandb_run is not None:
            wandb_run.log({
                "epoch":      e,
                "train/loss": mean_train_loss,
                "train/acc":  train_acc,
                "val/loss":   mean_val_loss,
                "val/acc":    val_acc,
            })

        if scheduler is not None:
            scheduler.step()

        # ── checkpoint ────────────────────────────────────────────────────────
        is_best = val_acc >= best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint(network, is_best, saving_path, epoch=e, acc=best_acc)

    # ── plot & save curves after training ─────────────────────────────────────
    fig = plot_history(history, saving_path)

    if wandb_run is not None:
        wandb_run.log({"training_curves": wandb.Image(fig)})

    plt.close(fig)


# ── helpers ────────────────────────────────────────────────────────────────────

def plot_history(history, saving_path):
    """Plot train/val loss and accuracy; save PNG to saving_path."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="tomato", linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="steelblue")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   color="tomato", linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()

    os.makedirs(saving_path, exist_ok=True)
    fig_path = os.path.join(saving_path, "train_val_curves.png")
    fig.savefig(fig_path, dpi=150)
    tqdm.write(f"Training curves saved → {fig_path}")

    return fig


def validation(network, val_loader, criterion, device):
    """Return (overall_acc, mean_val_loss)."""
    num_correct = 0.
    total_num = 0.
    val_losses = []

    network.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device)
            outputs = network(images)
            loss = criterion(outputs, targets)
            val_losses.append(loss.item())

            _, preds = torch.max(outputs, dim=1)
            num_correct += (preds == targets).sum().item()
            total_num += targets.size(0)

    overall_acc = num_correct / total_num
    mean_val_loss = float(np.mean(val_losses))
    return overall_acc, mean_val_loss


def test(network, model_dir, image, patch_size, n_classes, device):
    network.load_state_dict(torch.load(model_dir + "/model_best.pth"))
    network.eval()

    patch_size = patch_size
    batch_size = 64
    window_size = (patch_size, patch_size)
    image_w, image_h = image.shape[:2]
    pad_size = patch_size // 2

    # pad the image
    image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')

    probs = np.zeros(image.shape[:2] + (n_classes,))

    iterations = count_sliding_window(image, window_size=window_size) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(image, window_size=window_size)),
                      total=iterations,
                      desc="Inference on HSI"):
        with torch.no_grad():
            data = [b[0] for b in batch]
            data = np.copy(data)
            data = data.transpose((0, 3, 1, 2))
            data = torch.from_numpy(data)
            data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = network(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu').numpy()

            for (x, y, w, h), out in zip(indices, output):
                probs[x + w // 2, y + h // 2] += out

    return probs[pad_size:image_w + pad_size, pad_size:image_h + pad_size, :]


def save_checkpoint(network, is_best, saving_path, **kwargs):
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path, exist_ok=True)

    if is_best:
        tqdm.write("epoch = {epoch}: best OA = {acc:.4f}".format(**kwargs))
        torch.save(network.state_dict(), os.path.join(saving_path, 'model_best.pth'))
    else:
        if kwargs['epoch'] % 10 == 0:
            torch.save(network.state_dict(), os.path.join(saving_path, 'model.pth'))