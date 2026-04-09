import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
import wandb
from torchsummaryX import summary
from utils.dataset import load_mat_hsi, sample_gt, HSIDataset
from utils.utils import split_info_print, metrics, show_results, plot_confusion_matrix
from models.gtsvit import gtsvit
from train import train, test
from utils.utils import Draw
import torch.optim as optim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run patch-based HSI classification")
    parser.add_argument("--model", type=str, default='gtsvit')
    parser.add_argument("--dataset_name", type=str, default="ip")
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--num_run", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--wandb_project", type=str, default="GSCViT-HSI",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity",  type=str, default=None,
                        help="W&B entity (team/user). Leave blank to use default.")
    parser.add_argument("--no_wandb",      action="store_true",
                        help="Disable W&B logging entirely")

    opts = parser.parse_args()

    device = torch.device("cuda:{}".format(opts.device))

    # print parameters
    print("experiments will run on GPU device {}".format(opts.device))
    print("model = {}".format(opts.model))
    print("dataset = {}".format(opts.dataset_name))
    print("dataset folder = {}".format(opts.dataset_dir))
    print("patch size = {}".format(opts.patch_size))
    print("batch size = {}".format(opts.bs))
    print("total epoch = {}".format(opts.epoch))
    print("{} for training, {} for validation and {} testing".format(
        opts.ratio / 2, opts.ratio / 2, 1 - opts.ratio))

    # load data
    image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir)

    num_classes = len(labels)
    print("number of classes: {}".format(num_classes))

    num_bands = image.shape[-1]
    print("number of bands: {}".format(num_bands))

    # random seeds
    seeds = [202401, 202402, 202403, 202404, 202405,
             202406, 202407, 202408, 202409, 202410]

    results = []

    for run in range(opts.num_run):
        np.random.seed(seeds[run])
        print("running an experiment with the {} model".format(opts.model))
        print("run {} / {}".format(run + 1, opts.num_run))

        # ── W&B: one run per training run ─────────────────────────────────────
        wandb_run = None
        if not opts.no_wandb:
            wandb_run = wandb.init(
                project=opts.wandb_project,
                entity=opts.wandb_entity,
                name=f"{opts.model}_{opts.dataset_name}_run{run + 1}",
                config={
                    "model":       opts.model,
                    "dataset":     opts.dataset_name,
                    "patch_size":  opts.patch_size,
                    "batch_size":  opts.bs,
                    "epochs":      opts.epoch,
                    "ratio":       opts.ratio,
                    "seed":        seeds[run],
                    "run":         run + 1,
                },
                reinit=True,   # allow multiple wandb.init calls in same process
            )

        # get train_gt, val_gt and test_gt
        trainval_gt, test_gt = sample_gt(gt, opts.ratio, seeds[run])
        train_gt, val_gt     = sample_gt(trainval_gt, 0.5, seeds[run])
        del trainval_gt

        train_set = HSIDataset(image, train_gt, patch_size=opts.patch_size, data_aug=True)
        print("the number of training samples: {}".format(len(train_set)))
        val_set = HSIDataset(image, val_gt, patch_size=opts.patch_size, data_aug=False)
        print("the number of validation samples: {}".format(len(val_set)))

        train_loader = torch.utils.data.DataLoader(train_set, opts.bs, drop_last=False, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_set,   opts.bs, drop_last=False, shuffle=False)

        model = gtsvit(opts.dataset_name)
        if run == 0:
            split_info_print(train_gt, val_gt, test_gt, labels)


    
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
        optimizer =  optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
        scheduler = None    # load_scheduler(opts.model, optimizer, opts.epoch) --- IGNORE ---      
        model_dir = "./checkpoints/" + opts.model + '/' + opts.dataset_name + '/' + str(run)

        try:
            train(model, optimizer, criterion, train_loader, val_loader,
                  opts.epoch, model_dir, device,
                  scheduler=scheduler,
                  wandb_run=wandb_run)   # ← pass wandb run
        except KeyboardInterrupt:
            print('"ctrl+c" is pushed, the training is over')

        # test the model
        probabilities = test(model, model_dir, image, opts.patch_size, num_classes, device)
        prediction    = np.argmax(probabilities, axis=-1)

        # computing metrics
        run_results = metrics(prediction, test_gt, n_classes=num_classes)
        results.append(run_results)
        show_results(run_results, label_values=labels, agregated=False) 



        # ── log final test metrics to W&B ─────────────────────────────────────
        if wandb_run is not None:
            wandb_run.log({
                "test/OA":    run_results["Accuracy"],
                "test/class acc":    run_results["class acc"],
                "test/AA":    run_results["AA"],
                "test/kappa": run_results["Kappa"],
            })
            wandb_run.finish()

        # draw the classification map
        Draw(model, image, gt, opts.patch_size, opts.dataset_name, opts.model, num_classes)

        del model, train_set, train_loader, val_set, val_loader

    if opts.num_run > 1:
        show_results(results, label_values=labels, agregated=True)
                # Plot and save
        plot_confusion_matrix(
            results,
            class_names=labels,
            normalize=True,
            figsize=(15,15),
            save_path="./checkpoints/" + opts.model + '/' + opts.dataset_name + '/' + str(run) + "/confusion_matrix.png"   # change to .pdf if needed
        )