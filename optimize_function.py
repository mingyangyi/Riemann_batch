#from .optimizer import Optimizer, required
import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np

import gutils

import pdb

import gutils

def apply_dense_on_grasssmann(grad_clip, grad_on_grassmann, grad_on_oblique, var,learning_rate,times,delta):
    a = torch.max(delta, 1 / (torch.log(times+2)))
    #a=0.5
    n = gutils.unit(gutils.grassmann_project(var, grad_on_oblique)) * gutils.norm(grad_on_grassmann)
    b_1 = 2 * (1 - a) * gutils.xTy(grad_on_grassmann, n)
    b_2 = gutils.norm(grad_on_grassmann)
    b = b_1 / (b_2+1e-5)

    if grad_clip != None:
        h = learning_rate * (a * grad_on_grassmann + b * n)
        h = -1*gutils.clip_by_norm(h, grad_clip)
    else:
        h = -1* learning_rate * (a * grad_on_grassmann + b * n)

    var_update = gutils.grassmann_retrction(var, h)
    return var_update

def _apply_dense_on_oblique(grad_clip, grad_on_grassmann, grad_on_oblique, var,learning_rate,times,delta):
    a = torch.max(delta, 1 / torch.log(times + 2))
    # a=0.5
    n = gutils.unit(gutils.oblique_project(var, grad_on_grassmann)) * gutils.norm(grad_on_oblique)
    b_1 = 2 * (1 - a) * gutils.xTy(grad_on_oblique, n)
    b_2 = gutils.norm(grad_on_oblique)
    b = b_1 / (b_2 + 1e-5)

    if grad_clip !=None:
        h=-1*learning_rate*(a * grad_on_oblique + b * n)
        h=gutils.clip_by_norm(h,grad_clip)
    else:
        h = -1*learning_rate * (a * grad_on_oblique + b * n)

    var_update=gutils.oblique_retrction(var,h)
    return var_update

def _apply_dense_on_grassmann_with_noise(grad_clip,grad, var, learning_rate, times, variance):
    g=gutils.grassmann_project(var,grad)
    #g_norm=gutils.norm(g)

    #a=tf.minimum(1-1/(tf.square(times+1)*tf.square(g_norm)+1e-5),1/tf.square(times+1))
    a=1.0

    b=1/torch.square(times+1)

    noise = variance * gutils.grassmann_project(var,torch.randn(var.size()[0]))

    if grad_clip==None:
        h=-learning_rate*(a*g+b*gutils.noise)
    else:
        h = -learning_rate * (a * g + b * noise)
        h=gutils.clip_by_norm(h,grad_clip)
    var_new=gutils.grassmann_retrction(var,h)
    return var_new

def _apply_dense_on_oblique_with_noise(grad_clip,grad, var,learning_rate,times, variance):
    g = gutils.oblique_project(var, grad)
    #g_norm = gutils.norm(g)
    #a = tf.minimum(1 - 1 / (tf.square(times + 1) * tf.square(g_norm) + 1e-5), 1 / tf.square(times + 1))

    a=1.0
    b = 1 / torch.square(times+1)

    noise = variance * gutils.oblique_project(var, torch.randn(var.size()[0]))

    if grad_clip==None:
        h = -1*learning_rate * (a * g + b * noise)
    else:
        h = -1*learning_rate * (a * g + b * noise)
        h = gutils.clip_by_norm(h, grad_clip)

    var_new = gutils.grassmann_retrction(var, h)

    return var_new


class SGDM(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'grassmann'.

        If grassmann is True, the variables will be updated by SGD-G proposed
        in 'Riemannian approach to batch normalization'.

        If grassmann is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

        References:
           - Minhyung Cho and Jaehyung Lee, Riemannian approach to batch normalization
             (https://arxiv.org/abs/1709.09603)

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        grassmann (bool, optional): whether to use SGD-G (default: False)

        -- parameters in case grassmann is False
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case grassmann is True
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(self, params, lr=required, weight_decay=0, manifold="None", grad_clip=None, label='grassmann'):
        defaults = dict(lr=lr, weight_decay=weight_decay, manifold=manifold, grad_clip=grad_clip, label=label)
        #if nesterov and (momentum <= 0 or dampening != 0):
        #    raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDM, self).__init__(params, defaults)

    #def __setstate__(self, state):
    #    super(SGDG, self).__setstate__(state)
        #for group in self.param_groups:
        #    group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            #momentum = group['momentum']
            manifold = group['manifold']

            if manifold != "None":
                grad_clip = group['grad_clip']

                length = len(group['params'])

                for i in range(length):

                    p_grassmann = group['params'][i]
                    p_oblique = group['params'][i + length / 2]

                    if p_grassmann.grad and p_oblique is None:
                        continue

                    unity_grassmann, _ = gutils.unit(p_grassmann.data.view(p_grassmann.size()[0], -1))
                    unity_oblique, _ = gutils.unit(p_oblique.data.view(p_grassmann.size()[0], -1))

                    grad_grassmann = p_grassmann.grad.data.view(p_grassmann.size()[0], -1)
                    grad_oblique = p_grassmann.grad.data.view(p_oblique.size()[0], -1)

                    # if omega != 0:
                    # L=|Y'Y-I|^2/2=|YY'-I|^2/2+c
                    # dL/dY=2(YY'Y-Y)
                    # g.add_(2*omega, torch.mm(torch.mm(unity, unity.t()), unity) - unity)

                    h_grassmann = gutils.grassmann_project(unity_grassmann, grad_grassmann)
                    h_oblique = gutils.oblique_project(unity_oblique, grad_oblique)

                    if grad_clip is not None:
                        h_hat_grassmann = gutils.clip_by_norm(h_grassmann, grad_clip)
                        h_hat_oblique = gutils.clip_by_norm(h_oblique, grad_clip)
                    else:
                        h_hat_grassmann = h_grassmann
                        h_hat_oblique = h_oblique

                        # param_state = self.state[p]
                        # if 'momentum_buffer' not in param_state:
                        #    param_state['momentum_buffer'] = torch.zeros(h_hat.size())
                        #    if p.is_cuda:
                        #      param_state['momentum_buffer'] = param_state['momentum_buffer'].cuda()

                        # mom = param_state['momentum_buffer']
                        # mom_new = momentum*mom - group['lr']*h_hat

                    p_grassmann.data.copy_(
                        gutils.grassmann_retrction(unity_grassmann, group['lr'] * h_hat_grassmann).view(
                            p_grassmann.size()))
                    p_oblique.data.copy_(
                        gutils.oblique_retrction(unity_oblique, group['lr'] * h_hat_oblique).view(
                            p_oblique.size()))

            elif manifold =="None":
                # This routine is from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
                weight_decay = group['weight_decay']
                #dampening = group['dampening']
                #nesterov = group['nesterov']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    #if momentum != 0:
                    #    param_state = self.state[p]
                    #    if 'momentum_buffer' not in param_state:
                    #        buf = param_state['momentum_buffer'] = d_p.clone()
                    #    else:
                    #        buf = param_state['momentum_buffer']
                    #        buf.mul_(momentum).add_(1 - dampening, d_p)
                    #    if nesterov:
                    #        d_p = d_p.add(momentum, buf)
                    #    else:
                    #        d_p = buf

                    p.data.add_(-group['lr'], d_p)
                else:
                    raise ValueError("There is no such a manifold")

        return loss

class SGDE(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'grassmann'.

        If grassmann is True, the variables will be updated by SGD-G proposed
        in 'Riemannian approach to batch normalization'.

        If grassmann is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

        References:
           - Minhyung Cho and Jaehyung Lee, Riemannian approach to batch normalization
             (https://arxiv.org/abs/1709.09603)

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        grassmann (bool, optional): whether to use SGD-G (default: False)

        -- parameters in case grassmann is False
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case grassmann is True
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(self, params, times=0, lr=required, delta=0.001, grad_clip=None, manifold="None", weight_decay=0,
                 label='grassmann'):
        defaults = dict(times=times, lr=lr, grad_clip=grad_clip, delta=delta, manifold=manifold,
                        weight_decay=weight_decay, label=label)
        #if nesterov and (momentum <= 0 or dampening != 0):
        #    raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDE, self).__init__(params,defaults)

    #def __setstate__(self, state):
    #    super(SGDG, self).__setstate__(state)
        #for group in self.param_groups:
        #    group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            #momentum = group['momentum']
            manifold = group['manifold']

            if manifold == "True":
                grad_clip = group['grad_clip']

                length = len(group['params'])

                for i in range(length):

                    p_grassmann = group['params'][i]
                    p_oblique = group['params'][i + length/2]

                    if p_grassmann.grad and p_oblique is None:
                        continue

                    unity_grassmann,_ = gutils.unit(p_grassmann.data.view(p_grassmann.size()[0],-1))
                    unity_oblique,_ = gutils.unit(p_oblique.data.view(p_grassmann.size()[0],-1))

                    grad_grassmann = p_grassmann.grad.data.view(p_grassmann.size()[0],-1)
                    grad_oblique = p_grassmann.grad.data.view(p_oblique.size()[0],-1)

                        #if omega != 0:
                          # L=|Y'Y-I|^2/2=|YY'-I|^2/2+c
                          # dL/dY=2(YY'Y-Y)
                          #g.add_(2*omega, torch.mm(torch.mm(unity, unity.t()), unity) - unity)

                    h_grassmann = gutils.grassmann_project(unity_grassmann, grad_grassmann)
                    h_oblique = gutils.oblique_project(unity_oblique, grad_oblique)

                        # param_state = self.state[p]
                        # if 'momentum_buffer' not in param_state:
                        #    param_state['momentum_buffer'] = torch.zeros(h_hat.size())
                        #    if p.is_cuda:
                        #      param_state['momentum_buffer'] = param_state['momentum_buffer'].cuda()

                        # mom = param_state['momentum_buffer']
                        # mom_new = momentum*mom - group['lr']*h_hat

                    p_grassmann.data.copy_(
                        apply_dense_on_grasssmann(grad_clip, h_grassmann, h_oblique, unity_grassmann,
                                                    group['lr'], group['times'], group['delta']).view(p_grassmann.size()))

                    p_oblique.data.copy_(
                        _apply_dense_on_oblique(grad_clip, h_grassmann, h_oblique, unity_oblique, group['lr'],
                                                group['times'], group['delta']).view(p_oblique.size()))

                        # mom.copy_(gpt(unity, mom_new))

                    #if momentum != 0:
                    #    param_state = self.state[p]
                    #    if 'momentum_buffer' not in param_state:
                    #        buf = param_state['momentum_buffer'] = d_p.clone()
                    #    else:
                    #        buf = param_state['momentum_buffer']
                    #        buf.mul_(momentum).add_(1 - dampening, d_p)
                    #    if nesterov:
                    #        d_p = d_p.add(momentum, buf)
                    #    else:
                    #        d_p = buf
            else:
                # This routine is from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
                weight_decay = group['weight_decay']
                # dampening = group['dampening']
                # nesterov = group['nesterov']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    # if momentum != 0:
                    #    param_state = self.state[p]
                    #    if 'momentum_buffer' not in param_state:
                    #        buf = param_state['momentum_buffer'] = d_p.clone()
                    #    else:
                    #        buf = param_state['momentum_buffer']
                    #        buf.mul_(momentum).add_(1 - dampening, d_p)
                    #    if nesterov:
                    #        d_p = d_p.add(momentum, buf)
                    #    else:
                    #        d_p = buf

                    p.data.add_(-group['lr'], d_p)

        return loss

class SGDN(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'grassmann'.

        If grassmann is True, the variables will be updated by SGD-G proposed
        in 'Riemannian approach to batch normalization'.

        If grassmann is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

        References:
           - Minhyung Cho and Jaehyung Lee, Riemannian approach to batch normalization
             (https://arxiv.org/abs/1709.09603)

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        grassmann (bool, optional): whether to use SGD-G (default: False)

        -- parameters in case grassmann is False
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case grassmann is True
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(self, params, times, lr=required,variance = 0.0001 , grad_clip=None, manifold="None",weight_decay=0, label='grassmann'):
        defaults = dict(times=times, lr=lr, variance = variance, manifold=manifold, grad_clip=grad_clip,weight_decay=weight_decay,label=label)
        #if nesterov and (momentum <= 0 or dampening != 0):
        #    raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDN, self).__init__(params, defaults)

    #def __setstate__(self, state):
    #    super(SGDG, self).__setstate__(state)
        #for group in self.param_groups:
        #    group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            #momentum = group['momentum']
            manifold = group['manifold']
            learning_rate = group['lr']
            variance = group['variance']
            times = group['times']

            if manifold != 'None':
                grad_clip = group['grad_clip']

                length = len(group['params'])

                for i in range(length):

                    p_grassmann = group['params'][i]
                    p_oblique = group['params'][i + length / 2]

                    if p_grassmann.grad and p_oblique is None:
                        continue

                    unity_grassmann, _ = gutils.unit(p_grassmann.data.view(p_grassmann.size()[0], -1))
                    unity_oblique, _ = gutils.unit(p_oblique.data.view(p_grassmann.size()[0], -1))

                    grad_grassmann = p_grassmann.grad.data.view(p_grassmann.size()[0], -1)
                    grad_oblique = p_grassmann.grad.data.view(p_oblique.size()[0], -1)

                    # if omega != 0:
                    # L=|Y'Y-I|^2/2=|YY'-I|^2/2+c
                    # dL/dY=2(YY'Y-Y)
                    # g.add_(2*omega, torch.mm(torch.mm(unity, unity.t()), unity) - unity)

                    h_grassmann = gutils.grassmann_project(unity_grassmann, grad_grassmann)
                    h_oblique = gutils.oblique_project(unity_oblique, grad_oblique)

                        # param_state = self.state[p]
                        # if 'momentum_buffer' not in param_state:
                        #    param_state['momentum_buffer'] = torch.zeros(h_hat.size())
                        #    if p.is_cuda:
                        #      param_state['momentum_buffer'] = param_state['momentum_buffer'].cuda()

                        # mom = param_state['momentum_buffer']
                        # mom_new = momentum*mom - group['lr']*h_hat

                    p_grassmann.data.copy_(
                        _apply_dense_on_grassmann_with_noise(grad_clip, h_grassmann, unity_grassmann, learning_rate,
                                                             times, variance).view(p_grassmann.size()))

                    p_oblique.data.copy_(
                        _apply_dense_on_oblique_with_noise(grad_clip, h_oblique, unity_oblique, learning_rate,
                                                             times, variance).view(p_oblique.size()))

            elif manifold == "None":
                # This routine is from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
                weight_decay = group['weight_decay']
                # dampening = group['dampening']
                # nesterov = group['nesterov']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    # if momentum != 0:
                    #    param_state = self.state[p]
                    #    if 'momentum_buffer' not in param_state:
                    #        buf = param_state['momentum_buffer'] = d_p.clone()
                    #    else:
                    #        buf = param_state['momentum_buffer']
                    #        buf.mul_(momentum).add_(1 - dampening, d_p)
                    #    if nesterov:
                    #        d_p = d_p.add(momentum, buf)
                    #    else:
                    #        d_p = buf

                    p.data.add_(-group['lr'], d_p)

            else:
                raise ValueError("There is no such a manifold")

        return loss
