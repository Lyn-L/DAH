import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import Function

class DoublySN_Function(Function):
    
    @staticmethod
    def forward(ctx, logits, epsilon):
        ctx.save_for_backward(logits, epsilon)
        prob = 1.0 / (1 + torch.exp(-1 * logits))
        yout = (torch.sign(prob - epsilon) + 1.0) / 2.0
        return yout, prob
        
    @staticmethod
    def backward(ctx, dprev, dpout):
        logits, epsilon = ctx.saved_variables
        prob = 1.0 / (1 + torch.exp(-1 * logits))
        yout = (torch.sign(prob - epsilon) + 1.0) / 2.0
        dlogits = prob * (1 - prob) * (dprev + dpout)
        depsilon = dprev
        return dlogits, depsilon

class DoublySN(nn.Module):
	def __init__(self):
		super(DoublySN, self).__init__()

	def forward(self, input, epsilon):
		return DoublySN_Function.apply(input, epsilon)