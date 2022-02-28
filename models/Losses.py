import torch

def gram(x):
    c = x.shape[1]
    n = torch.numel(x) / x.shape[1]
    y = x.reshape((c, int(n)))
    return (y.mm(y.t())) / n

def style_loss(yhat, y):
    return torch.abs(gram(yhat) - gram(y)).mean()

def tv_loss(yhat):
    return 0.5*((yhat[:,:,1:,:] - yhat[:,:,:-1,:]).abs_().mean() +
                (yhat[:,:,:,1:] - yhat[:,:,:,:-1]).abs_().mean())