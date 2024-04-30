import numpy as np
import torch
import os
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, cfg ,patience=7, verbose=True, delta=0,trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = cfg["patience"]
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_min = np.Inf
        self.delta = delta
        self.path = None
        self.path2 = None
        self.trace_func = trace_func
        self.cfg = cfg
        
    def path_replace(self, args,epoch):
        save_path = str(args.arch) + "_" + str(self.cfg["lr"]) + "_" + str(self.cfg["batch_size"]) + "_" + str(self.cfg["imgsize"]) + "_" + str(self.cfg["num_class"]) + "_" + str(self.cfg["train_loss_pra"])
        best_path = os.path.join(args.save_path_best, 'One_GPU_Version_{}_best_{}.pt'.format(save_path,epoch))
        best_path2 = os.path.join(args.save_path_best2, 'Two_GPU_Version{}_best_{}.pt'.format(save_path,epoch))
        self.path = best_path
        self.path2 = best_path2

    def __call__(self, val_acc, model,args,epoch):
        score = val_acc
        if self.best_score is None:
            self.path_replace(args,epoch)
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.path_replace(args,epoch)
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation acc decrease.'''
        if self.verbose:
            self.trace_func(f'Validation acc decreased ({self.val_acc_min:.6f}% --> {val_acc:.6f}%).  Saving model ...')
        torch.save(model.module.state_dict(), self.path)
        torch.save(model.state_dict(),self.path2)
        self.val_acc_min = val_acc