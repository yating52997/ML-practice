import matplotlib.pyplot as plt
import numpy as np
import os
from train import save_dir

def plot_lrs(history):
  plt.cla()
  lrs = ([x.get('lrs') for x in history])
  plt.plot(lrs, '-bx')
  plt.xlabel('Batch no.')
  plt.ylabel('Learning rate')
  plt.title('Learning Rate vs. Batch no.')
  plt.savefig(os.path.join(save_dir, 'lrs.png'))



def plot_losses(history):
  plt.cla()
  train_losses = ([x.get('train_loss') for x in history])
  test_losses = ([x.get('test_loss') for x in history])
  plt.plot(train_losses, '-bx', label='train')
  plt.plot(test_losses, '-rx', label='test')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend()
  plt.title('Loss vs. No. of epochs')
  plt.savefig(os.path.join(save_dir, 'losses.png'))


def plot_test(history):
  plt.cla()
  test_accs = ([x.get('test_acc') for x in history])
  plt.plot(test_accs, '-bx')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend(['testing'])
  plt.title('accuracy vs. No. of epochs')
  plt.savefig(os.path.join(save_dir, 'acc.png'))


if __name__ == "__main__":
  history = [{'train_loss': [0.1], 'test_loss': [0.2], 'test_acc': [0.3], 'lrs': [0.4]}]
  plot_test(history)
  plot_lrs(history)
  plot_losses(history)
