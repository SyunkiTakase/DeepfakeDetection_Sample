import numpy as np
import matplotlib.pyplot as plt

def draw_loss_graph(train_losses, val_losses):
    # total loss
    fig = plt.figure()
    plt.plot(train_losses,label='train')
    plt.plot(val_losses,label='val')
    plt.title('loss')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    fig.savefig("loss.png")

def draw_acc_graph(train_accs, val_accs):
    # class accuracy
    fig = plt.figure()
    plt.plot(train_accs,label='train')
    plt.plot(val_accs,label='val')
    plt.title('accuracy')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    fig.savefig("accuracy.png")

