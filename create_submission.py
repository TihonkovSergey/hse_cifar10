import argparse
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.resnet import ResNet20
from src.transforms import transforms_test
from src.dataset import CIFAR10
from globals import BATCH_SIZE


def write_solution(filename, labels):
    with open(filename, 'w') as solution:
        print('Id,Category', file=solution)
        for i, label in enumerate(labels):
            print(f'{i},{label}', file=solution)


def main(args):
    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    data_path = Path(args.data_path)

    test_data = CIFAR10(
        root=str(data_path.parent.absolute()),
        filename=data_path.name,
        transform=transforms_test,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
    )

    predictions = []

    model = torch.nn.DataParallel(ResNet20())
    model.load_state_dict(torch.load(args.checkpoint_path)["state_dict"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        for X, _ in tqdm(test_dataloader):
            X = X.to(device)
            pred = model(X).argmax(1).cpu().numpy()
            predictions.extend(list(pred))

    write_solution(args.name, predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", dest='use_cuda', action="store_true")
    parser.add_argument('--name', help='The solution filename', default='solution.csv', type=str)
    parser.add_argument('--data-path', dest='data_path', help='Path to data file', default='data/data_test', type=str)
    parser.add_argument('--checkpoint-path', dest='checkpoint_path', help='Path to checkpoint file',
                        default='checkpoints/checkpoint.th', type=str)

    args = parser.parse_args()
    main(args)
