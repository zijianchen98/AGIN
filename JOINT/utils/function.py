import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import torch
import torchsort
import torch.nn as nn
import torch.nn.functional as F

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
                        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)

    return y_output_logistic


def fit_function_regression_values(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
                        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)

    return y_output_logistic, popt


def performance_fit(y_label, y_output):
    y_output_logistic, popt = fit_function_regression_values(y_label, y_output)
    PLCC = stats.pearsonr(y_output_logistic, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt(((y_output_logistic - y_label) ** 2).mean())
    MAE = np.absolute((y_output_logistic - y_label)).mean()

    return PLCC, SRCC, KRCC, RMSE, MAE, popt


EPS = 1e-2
esp = 1e-8


class Fidelity_Loss(torch.nn.Module):

    def __init__(self):
        super(Fidelity_Loss, self).__init__()

    def forward(self, p, g):
        g = g.view(-1, 1)
        p = p.view(-1, 1)
        loss = 1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))

        return torch.mean(loss)


def loss_accuracy(y_pred, y):
    """prediction accuracy related loss"""
    assert y_pred.size(0) > 1
    return (1 - torch.cosine_similarity(y_pred.t() - torch.mean(y_pred), y.t() - torch.mean(y))[0]) / 2


class Monotonicity_Loss(nn.Module):
    def __init__(self):
        super(Monotonicity_Loss, self).__init__()
        
    def forward(self,p,v,**kw):
        pred = torch.t(p)
        pred = torchsort.soft_rank(pred, **kw)
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = torch.t(v)
        target = torchsort.soft_rank(target, **kw)
        target = target - target.mean()
        target = target / target.norm()
        return 1 - (pred * target).sum()
                
                
                
                

def loss_monotonicity(y_pred, y, **kw):
    """prediction monotonicity related loss"""
    assert y_pred.size(0) > 1
    pred = torch.t(y_pred)
    pred = torchsort.soft_rank(pred, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = torch.t(y)
    target = torchsort.soft_rank(target, **kw)
    target = target - target.mean()
    target = target / target.norm()
    return 1 - (pred * target).sum()