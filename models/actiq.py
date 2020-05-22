import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


# Inherit from Function
class Fakerelu(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, a, k=4):
        ctx.save_for_backward(input, a)

        # output = input.sign()
        output = 0.5 * (torch.abs(input) - torch.abs(input - a) + a)
        output = torch.round(output * float(2 ** k - 1) / a) * (a / float(2 ** k - 1))
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, a = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
            grad_input[input.ge(a)] = 0.0
            grad_input[input.le(0)] = 0.0
        if ctx.needs_input_grad[1]:
            grad_a = grad_output.clone()
            grad_a[input.le(a)] = 0

        return grad_input, grad_a


class FR(nn.Module):
    def __init__(self):
        super(FR, self).__init__()
        self.a = Parameter(torch.tensor(3.0))

    def forward(self, input):
        return Fakerelu.apply(input, self.a)


# ts = torch.tensor([[[[-2, -1, -0.5], [0, 0.2, 0.6], [1, 1.5, 7]], [[-2, -1, -0.5], [0, 0.2, 0.6], [1, 1.5, 7]],
#                     [[-2, -1, -0.5], [0, 0.2, 0.6], [1, 1.5, 7]]]])
# op = FR()(ts)
# print(op)
# op = nn.BatchNorm2d(3)(op)
# print(op)
# op = FR()(op)
# print(op)
