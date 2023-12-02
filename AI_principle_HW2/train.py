import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import utils


parser = argparse.ArgumentParser(description="cifar10_training")

parser.add_argument("--batch_size", type=int, default=1024,
                    help="training batch size")
parser.add_argument("--save_dir", type=str, default="./part1_cifar10",
                    help="saving checkpoint and pictures")
parser.add_argument("--resume", default="", type=str,
                    help="path to checkpoint") 
parser.add_argument("--n_epoch", default=50, type=int,
                    help="number of epochs to train")
parser.add_argument("--lr", default=0.001, type=float,
                    help="initial learning rate") 
parser.add_argument("--download", default=False, type=bool,
                    help="download dataset")
parser.add_argument("--device", default='cuda', type=str,
                    help="device to run the model")

args = parser.parse_args()

save_dir = args.save_dir

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_data = datasets.CIFAR10(
    root='./cifar10',
    train=True,
    transform = train_transform,
    download = args.download
)

test_data = datasets.CIFAR10(
    root='./cifar10',
    train=False,
    transform = test_transform,
    download = args.download
)


train_dataloader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)

test_dataloader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']

class Net(nn.Module):
    def __init__(self, num_classes = 10):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(8 * 8 * 64, num_classes),
            nn.ReLU(inplace=True),
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        out = self.softmax(out)
        return out
 
  
model = Net().to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_function = nn.CrossEntropyLoss()
best_acc = 0

history = []


def Test(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            test_loss += loss_function(out, labels).item()
            pred = out.argmax(dim=1, keepdim=True)
            test_acc += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_acc = (100. * test_acc) / len(test_loader.dataset)
    print('\nTest Set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_acc))
    return test_loss, test_acc

if __name__ == "__main__":

    for epoch in range(args.n_epoch):
        # train
        model.train()
        epoch_loss = 0
        result = {'train_loss': [], 'test_loss': [], 'test_acc': [], 'lrs': []}
        for batch_idx, (images, labels) in enumerate(tqdm(train_dataloader)):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            out = model(images)
            loss = loss_function(out, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('Train Epoch: {} Loss: {:.6f}'.format(epoch, epoch_loss / len(train_dataloader)))
        
        test_loss, test_acc = Test(model, args.device, test_dataloader)
        
        result['train_loss'].append(epoch_loss / len(train_dataloader))
        result['test_loss'].append(test_loss)
        result['test_acc'].append(test_acc)
        result['lrs'].append(optimizer.param_groups[0]['lr'])
        history.append(result)
        utils.plot_test(history)
        utils.plot_lrs(history)
        utils.plot_losses(history)
        
        # Save Checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            print('Saving checkpoint...')
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc }
            if not os.path.isdir(save_dir): os.mkdir(save_dir)
            torch.save(state, os.path.join(save_dir , 'checkpoint.pth'))

        print(f'Best Accuracy in this training process : {best_acc}%')

