import argparse
import os
import pickle
from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.resnet import ResNet20
from src.dataset import CIFAR10
from src.transforms import transforms_train
from train_with_validate import (
    train,
    seed_everything,
)
from globals import (
    SEED,
    BATCH_SIZE,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch', default=BATCH_SIZE, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float)
    parser.add_argument('--momentum', '-m', default=0.9, type=float)
    parser.add_argument('--weight-decay', '-w', dest='weight_decay', default=1e-4, type=float)
    parser.add_argument("--use-cuda", dest='use_cuda', action="store_true")
    parser.add_argument('--train-data-path', dest='train_data_path', help='Path to train data file',
                        default='data/data_train', type=str)
    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', help='Directory for checkpoint file',
                        default='checkpoints', type=str)

    args = parser.parse_args()
    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    seed_everything(SEED)

    model = torch.nn.DataParallel(ResNet20())
    model.to(device)

    train_data_path = Path(args.train_data_path)

    train_data = CIFAR10(
        root=str(train_data_path.parent.absolute()),
        filename=train_data_path.name,
        transform=transforms_train,
    )

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    def lr_schedule(ep):
        factor = 1
        if ep >= 81:
            factor /= 10
        if ep >= 122:
            factor /= 10
        return factor


    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    best_acc1 = 0.0

    train_loss_list = []
    train_acc1_list = []
    for epoch in tqdm(range(args.epochs)):
        # train for one epoch
        print(f"current lr {optimizer.param_groups[0]['lr']:.5e}")
        acc1_train, loss_train = train(train_loader, model, criterion, optimizer, epoch)
        train_acc1_list.append(acc1_train)
        train_loss_list.append(loss_train)
        lr_scheduler.step()

    torch.save(
        {
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
        },
        os.path.join(args.checkpoint_dir, 'checkpoint.th')
    )

    with open(Path(args.checkpoint_dir).joinpath('statistics.pickle'), 'wb') as f:
        pickle.dump({
            "train_loss": train_loss_list,
            "train_acc1": train_acc1_list,
        }, f, pickle.HIGHEST_PROTOCOL)
