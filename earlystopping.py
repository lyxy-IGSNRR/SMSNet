import numpy as np
import torch
import os


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    """ 如果在给定的耐心之后，验证损失没有改善，则提前停止训练。（暂时不用管）"""
    def __init__(self, patience=15, verbose=False, delta=1e-6):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.最后一次验证损失改善后要等多长时间。
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        
    def __call__(self, val_loss, model, epoch, save_path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path)
        elif score - self.delta < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            #self.save_checkpoint(val_loss, model, epoch, save_path)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        # torch.save(
        #     model, save_path + "/" +
        #     "checkpoint_{}_{:.6f}.pth.tar".format(epoch, val_loss))
        torch.save(model, save_path + "/" + "checkpoint_best.pth.tar")
            
        self.val_loss_min = val_loss
