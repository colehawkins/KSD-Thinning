"""Utilities for BNN metrics computation"""

import numpy as np
import torch
import utils
           

def get_reference_metrics(bma_pred,reference_predictions):
    """Wrapper function for agreement/TV metrics"""
    bma_agreement = torch.eq(bma_pred.argmax(dim=1),
                         reference_predictions.argmax(dim=1)).float().mean()

    bma_tv = 0.5*(bma_pred-reference_predictions).abs().mean(dim=0).sum()

    return {'bma_agreement':bma_agreement.item(),'bma_tv':bma_tv.item()}

def get_metrics(bma_pred,stacked_labels,pruning_container):
    """Wrapper function for clean metrics"""

    bma_ece = utils.bnn_metrics.ece_score(bma_pred.cpu().numpy(),
        stacked_labels.cpu().numpy())
    bma_test_acc = torch.eq(bma_pred.argmax(dim=1),
                             stacked_labels).float().mean().item() 
   
    num_samples = pruning_container.points.shape[0]
    ksd = pruning_container.get_ksd_squared().sqrt().item()
    
    return {'bma_ece':bma_ece,'bma_accuracy':bma_test_acc,'num_samples':num_samples,'ksd':ksd}


def get_all_labels(test_loader):
    """Utility function to stack labels from dataloader"""
    return torch.cat([ y for _,y in test_loader])



def ece_score(py, y_test, n_bins=10):
    """
    Computes ECE given preds and labels
    Taken from https://raw.githubusercontent.com/sirius8050/Expected-Calibration-Error/master/ECE.py
    """
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)


class BMAContainer:
    """Stores running BMA for predictions and maintains add/drop"""

    def __init__(self,add_only):
        self.add_only = add_only
        self.results_dict = {}
        self.num_preds = 0

    def make_key(self,sample):
        return tuple([x for x in np.reshape(sample.cpu().numpy(),-1)])

    def add_pred(self,pred,sample=None):
        
        if not hasattr(self,'bma_pred'):
            self.bma_pred = pred 

        #if add-only, don't need to store preds to drop later
        
        coeff = 1.0/(self.num_preds+1.0)
        self.bma_pred = (1.0-coeff)*self.bma_pred + coeff*pred
     
        self.num_preds += 1
        #may need to drop preds later
        if not self.add_only:
            key = self.make_key(sample)
            self.results_dict[key]=pred

    def drop_sample(self,sample):

        key = self.make_key(sample) 
        #drop pred from dict
        pred = self.results_dict.pop(key)

        #remove from running average
        self.bma_pred -= (1.0/self.num_preds)*pred

        #renormalize
        self.bma_pred *= (self.num_preds/(self.num_preds-1.0))

        #adjust number of results
        self.num_preds -= 1

    def compute_from_scratch(self):
        """Useful to test equivalence after pruning"""
        vals = [x for x in self.results_dict.values()]

        return torch.mean(torch.stack(vals),dim=0)
