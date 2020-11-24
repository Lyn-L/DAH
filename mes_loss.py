from torch.autograd import Function

class mes_loss(Function):
    @staticmethod
    def forward(ctx, i, j):
        result = (i - j).pow(2).sum().mul(i.size(0))
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, *grad_outputs):
        result, = ctx.saved_variables
        return result