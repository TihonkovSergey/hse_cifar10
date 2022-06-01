import argparse
import random
import os
import time
import pickle
from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from src.resnet import ResNet20
from src.dataset import CIFAR10
from src.transforms import transforms_test, transforms_train
from src.average_meter import AverageMeter
from globals import (
    SEED,
    BATCH_SIZE,
    PRINT_FREQ,
)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        acc1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print('\rEpoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1), end='')
    print('\nEpoch took {:.2f} s.'.format(end - start))
    return top1.avg, losses.avg


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            acc1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % PRINT_FREQ == 0:
                print('\rTest: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1), end='')

    print('\n * Acc@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, losses.avg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch', default=BATCH_SIZE, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float)
    parser.add_argument('--momentum', '-m', default=0.9, type=float)
    parser.add_argument('--weight-decay', '-w', dest='weight_decay', default=1e-4, type=float)
    parser.add_argument("--use-cuda", dest='use_cuda', action="store_true")
    parser.add_argument('--train-data-path', dest='train_data_path', help='Path to train data file',
                        default='data/train', type=str)
    parser.add_argument('--validate-data-path', dest='validate_data_path', help='Path to validate data file',
                        default='data/valid', type=str)
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
    validate_data_path = Path(args.validate_data_path)

    train_data = CIFAR10(
        root=str(train_data_path.parent.absolute()),
        filename=train_data_path.name,
        transform=transforms_train,
    )
    val_data = CIFAR10(
        root=str(validate_data_path.parent.absolute()),
        filename=validate_data_path.name,
        transform=transforms_test,
    )

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

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
    valid_loss_list = []
    train_acc1_list = []
    valid_acc1_list = []
    for epoch in tqdm(range(args.epochs)):
        # train for one epoch
        print(f"current lr {optimizer.param_groups[0]['lr']:.5e}")
        acc1_train, loss_train = train(train_loader, model, criterion, optimizer, epoch)
        train_acc1_list.append(acc1_train)
        train_loss_list.append(loss_train)
        lr_scheduler.step()

        # validation
        acc1_val, loss_val = validate(val_loader, model, criterion)
        valid_acc1_list.append(acc1_val)
        valid_loss_list.append(loss_val)

        # save best checkpoint
        if acc1_val > best_acc1 and epoch > 0:
            best_acc1 = acc1_val
            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                },
                os.path.join(args.checkpoint_dir, 'best_checkpoint.th')
            )

    with open(Path(args.checkpoint_dir).joinpath('statistics.pickle'), 'wb') as f:
        pickle.dump({
            "train_loss": train_loss_list,
            "valid_loss": valid_loss_list,
            "train_acc1": train_acc1_list,
            "valid_acc1": valid_acc1_list
        }, f, pickle.HIGHEST_PROTOCOL)
