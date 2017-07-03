from __future__ import print_function

import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--log_file', default='log.txt', help='Log file', type=str)
parser.add_argument('--loss_lim', default='10', help='Upper limit for the loss plot', type=float)
parser.add_argument('--acc_lim', default='0', help='Lower limit for the accuracy plot', type=float)
parser.add_argument('--window_size', default='15', help='The size of the smoothing window', type=int)
args = parser.parse_args()

file_name = args.log_file
ws = args.window_size
loss_lim = args.loss_lim
acc_lim = args.acc_lim

def smoothen(arr, window):
    to_return = []
    s = 0.
    for i in range(min(len(arr), window)):
        s += arr[i]
        to_return.append(s/(i+1))
    for i in range(window, len(arr)):
        s += arr[i]
        s -= arr[i-window]
        to_return.append(s/window)
    return to_return

with open(file_name, 'r') as log:
    lines = log.readlines()
    train_iters = [float(line.split()[1]) for line in lines if "Train Loss" in line]
    train_losses = [float(line.split()[4]) for line in lines if "Train Loss" in line]
    train_acc = [float(line.split()[7]) for line in lines if "Train Acc" in line]

    val_iters = [float(line.split()[1]) for line in lines if "Val Loss" in line]
    val_losses = [float(line.split()[4]) for line in lines if  "Val Loss" in line]
    val_acc = [float(line.split()[7]) for line in lines if "Val Acc" in line]


plt.plot(train_iters, smoothen(train_losses, window=ws))
plt.plot(np.array(val_iters), val_losses)
plt.ylim([0, loss_lim])
#plt.show()
plt.savefig('loss.png')
plt.plot(train_iters, smoothen(train_acc, window=ws))
plt.plot(np.array(val_iters), val_acc)
plt.ylim([0, acc_lim])
#plt.show()
plt.savefig('acc.png')
