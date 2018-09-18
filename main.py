"""
    PyTorch training code for Wide Residual Networks:
    http://arxiv.org/abs/1605.07146

    The code reproduces *exactly* it's lua version:
    https://github.com/szagoruyko/wide-residual-networks

    2016 Sergey Zagoruyko
"""

import argparse
import os
import json
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.optim
import torch.utils.data
import cvtransforms as T
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from utils import cast, data_parallel
import torch.backends.cudnn as cudnn
import resnet

import optimize_function
from gutils import unit

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--model', default='resnet', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--dataroot', default='.', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--groups', default=1, type=int)
parser.add_argument('--nthread', default=4, type=int)

# Training options
parser.add_argument('--batchSize', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--lrm', default=0.1, type=float)
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weightDecay', default=0.0005, type=float)
parser.add_argument('--bnDecay', default=0, type=float)
parser.add_argument('--omega', default=0.1, type=float)
parser.add_argument('--grad_clip', default=0.1, type=float)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--optim_method', default='SGD', type=str)
parser.add_argument('--randomcrop_pad', default=4, type=float)

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='output/original', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--save_grassmann', default='output/grassmann', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--save_oblique', default='output/oblique', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

def create_dataset(opt, mode):
    if opt.dataset == 'CIFAR10':
      mean = [125.3, 123.0, 113.9]
      std = [63.0, 62.1, 66.7]
    elif opt.dataset =='CIFAR100':
      mean = [129.3, 124.1, 112.4]
      std = [68.2, 65.4, 70.4]
    else:
      mean = [0, 0, 0]
      std = [1.0, 1.0, 1.0]


    convert = tnt.transform.compose([
        lambda x: x.astype(np.float32),
        T.Normalize(mean, std),
        lambda x: x.transpose(2,0,1).astype(np.float32),
        torch.from_numpy,
    ])

    train_transform = tnt.transform.compose([
        T.RandomHorizontalFlip(),
        T.Pad(opt.randomcrop_pad, cv2.BORDER_REFLECT),
        T.RandomCrop(32),
        convert,
    ])

    ds = getattr(datasets, opt.dataset)(opt.dataroot, train=mode, download=True)
    smode = 'train' if mode else 'test'
    ds = tnt.dataset.TensorDataset([getattr(ds, smode + '_data'),
                                    getattr(ds, smode + '_labels')])
    return ds.transform({0: train_transform if mode else convert})


def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    epoch_step = json.loads(opt.epoch_step)
    num_classes = 10 if opt.dataset == 'CIFAR10' else 100

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    # to prevent opencv from initializing CUDA in workers
    torch.randn(8).cuda()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    def create_iterator(mode):
        ds = create_dataset(opt, mode)
        return ds.parallel(batch_size=opt.batchSize, shuffle=mode,
                           num_workers=opt.nthread, pin_memory=True)

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)

    if opt.optim_method == 'SGDM' or opt.optim_method == 'SGDE' or opt.optim_method == 'SGDN':

        f_grassmann, params_grassmann, stats_grassmann = resnet.resnet_grassmann(opt.depth, opt.width, num_classes)
        f_oblique, params_oblique, stats_oblique = resnet.resnet_oblique(opt.depth, opt.width, num_classes)

        key_g = []
        key_o = []

        param_g = []
        param_g_e0 = []
        param_g_e1 = []

        param_o = []
        param_o_e0 = []
        param_o_e1 = []

        params_total = []

        for key, value in params_grassmann.items():
            if 'conv' in key and value.size()[0] < np.prod(value.size()[1:]):
                params_total.append(value)
                key_g.append(key)
                # initlize to scale 1
                unitp = unit(value.data.view(value.size(0), -1))
                value.data.copy_(unitp.view(value.size()))
            elif 'bn' in key or 'bias' in key:
                param_g_e0.append(value)
            else:
                param_g_e1.append(value)

        for key, value in params_oblique.items():
            if 'conv' in key and value.size()[0] < np.prod(value.size()[1:]):
                params_total.append(value)
                key_o.append(key)
                # initlize to scale 1
                unitp = unit(value.data.view(value.size(0), -1))
                value.data.copy_(unitp.view(value.size()))
            elif 'bn' in key or 'bias' in key:
                param_o_e0.append(value)
            else:
                param_o_e1.append(value)

        def create_optimizer(opt, lr, lrm, times):
            print('creating optimizer with lr = ', lr)

            if opt.optim_method == 'SGDM':
                dict_total = {'params': params_total, 'lr': lrm, 'manifold' : 'True', 'grad_clip': opt.grad_clip}
                dict_g_e0 = {'params': param_g_e0, 'lr': lr, 'weight_decay': opt.bnDecay, 'manifold': 'None'}
                dict_g_e1 = {'params': param_g_e1, 'lr': lr, 'weight_decay': opt.bnDecay, 'manifold': 'None'}

                dict_o_e0 = {'params': param_o_e0, 'lr': lr, 'weight_decay': opt.bnDecay, 'manifold': 'None',
                             'label': 'oblique'}
                dict_o_e1 = {'params': param_o_e1, 'lr': lr, 'weight_decay': opt.bnDecay, 'manifold': 'None',
                             'label': 'oblique'}

                return optimize_function.SGDM([dict_total, dict_g_e0, dict_g_e1, dict_o_e0, dict_o_e1])

            elif opt.optim_method == 'SGDE':
                dict_total = {'params': params_total, 'times' : times, 'lr': lrm, 'manifold': 'True', 'grad_clip': opt.grad_clip}
                dict_g_e0 = {'params': param_g_e0, 'lr': lr, 'weight_decay': opt.bnDecay, 'manifold': 'None'}
                dict_g_e1 = {'params': param_g_e1, 'lr': lr, 'weight_decay': opt.bnDecay, 'manifold': 'None'}
                dict_o_e0 = {'params': param_o_e0, 'lr': lr, 'weight_decay': opt.bnDecay, 'manifold': 'None',
                             'label': 'oblique'}
                dict_o_e1 = {'params': param_o_e1, 'lr': lr, 'weight_decay': opt.bnDecay, 'manifold': 'None',
                             'label': 'oblique'}

                return optimize_function.SGDM([dict_total, dict_g_e0, dict_g_e1, dict_o_e0, dict_o_e1])

            elif opt.optim_method == 'SGDN':
                dict_total = {'params': params_total, 'lr': lrm, 'times': times, 'manifold': 'True', 'grad_clip': opt.grad_clip}
                dict_g_e0 = {'params': param_g_e0, 'lr': lr, 'weight_decay': opt.bnDecay, 'manifold': 'None'}
                dict_g_e1 = {'params': param_g_e1, 'lr': lr, 'weight_decay': opt.bnDecay, 'manifold': 'None'}

                dict_o_e0 = {'params': param_o_e0, 'lr': lr, 'weight_decay': opt.bnDecay, 'manifold': 'None',
                             'label': 'oblique'}
                dict_o_e1 = {'params': param_o_e1, 'lr': lr, 'weight_decay': opt.bnDecay, 'manifold': 'None',
                             'label': 'oblique'}

                return optimize_function.SGDM([dict_total, dict_g_e0, dict_g_e1, dict_o_e0, dict_o_e1])

        epoch = 0
        optimizer = create_optimizer(opt, opt.lr, opt.lrm, epoch)

        if opt.resume != '':
            state_dict = torch.load(opt.resume)
            epoch = state_dict['epoch']
            if opt.optim_method != 'SGD':
                params_tensor, stats, manifold, label = state_dict['params'], state_dict['stats'], state_dict['manifold'], \
                                                        state_dict['label']
                size = manifold.size()[0]

                for i in range(size):

                    if state_dict['manifold'][i] != 'None':
                        length = params_tensor[i].size()[0]/2

                        tmp_grassmann = list(params_grassmann.items())
                        tmp_grassmann[i].data.copy_(params_tensor[i][0:length])

                        tmp_oblique = list(params_oblique.items())
                        tmp_oblique[i].data.copy_(params_tensor[i][length:])


                    elif state_dict['label'][i] == 'grassmann':

                        tmp_grassmann = list(params_grassmann.items())
                        tmp_grassmann[i].data.copy_(params_tensor[i])

                    else:
                        tmp_oblique = list(params_grassmann.items())
                        tmp_oblique[i].data.copy_(params_tensor[i])

                optimizer.load_state_dict(state_dict['optimizer'])

        print('\nParameters:')
        kmax = max(len(key) for key in params_grassmann.keys())
        for i, (key, v) in enumerate(params_grassmann.items()):
            print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.size())).ljust(23), torch.typename(v.data), end='')
            print(' on G(1,n)' if key in key_g else '')

        meter_loss_ensemble = tnt.meter.AverageValueMeter()
        classacc_ensemble = tnt.meter.ClassErrorMeter(accuracy=True)

        timer_train = tnt.meter.TimeMeter('s')
        timer_test = tnt.meter.TimeMeter('s')

        #print('\nAdditional buffers:')
        #kmax = max(len(key) for key in stats.keys())
        #for i, (key, v) in enumerate(stats.items()):
        #    print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.size())).ljust(23), torch.typename(v))

    #    n_parameters = sum(p.numel() for p in params.values() + stats.values())
        #n_training_params = sum(p.numel() for p in params.values())
        #n_parameters = sum(p.numel() for p in params.values()) + sum(p.numel() for p in stats.values())
        #print('Total number of parameters:', n_parameters, '(%d)'%n_training_params)

        if not os.path.exists(opt.save):
            os.mkdir(opt.save)

        def h_ensemble(sample):
            inputs = Variable(cast(sample[0], opt.dtype))
            targets = Variable(cast(sample[1], 'long'))
            y_grassmann = data_parallel(f_grassmann, inputs, params_grassmann, stats_grassmann, sample[2], np.arange(opt.ngpu))
            y_oblique = data_parallel(f_oblique, inputs, params_oblique, stats_oblique, sample[2], np.arange(opt.ngpu))
            y_ensemble = y_grassmann + y_oblique
            return F.cross_entropy(y_ensemble, targets), y_ensemble

        def log_grassmann(t, state):
            #        torch.save(dict(params={k: v.data for k, v in params.iteritems()},
            torch.save(dict(params_grassmann={k: v.data for k, v in list(params_grassmann.items())},
                            stats_grassmann=stats_grassmann,
                            optimizer=state['optimizer'].state_dict(),
                            epoch=t['epoch']),
                       open(os.path.join(opt.save_grassmann, 'model.pt7'), 'wb'))
            z = vars(opt).copy();
            z.update(t)
            logname = os.path.join(opt.save_grassmann, 'log.txt')
            with open(logname, 'a') as f:
                f.write('json_stats: ' + json.dumps(z) + '\n')
            print(z)

        def log_oblique(t, state):
            #        torch.save(dict(params={k: v.data for k, v in params.iteritems()},
            torch.save(dict(params_oblique={k: v.data for k, v in list(params_oblique.items())},
                            stats_oblique=stats_oblique,
                            optimizer=state['optimizer'].state_dict(),
                            epoch=t['epoch']),
                       open(os.path.join(opt.save_oblique, 'model.pt7'), 'wb'))
            z = vars(opt).copy();
            z.update(t)
            logname = os.path.join(opt.save_oblique, 'log.txt')
            with open(logname, 'a') as f:
                f.write('json_stats: ' + json.dumps(z) + '\n')
            print(z)

        def on_sample(state):
            state['sample'].append(state['train'])

        def on_forward(state):
            classacc_ensemble.add(state['output'].data, torch.LongTensor(state['sample'][1]))
            meter_loss_ensemble.add(state['loss'].data[0])

        def on_start(state):
            state['epoch'] = epoch

        def on_start_epoch(state):
            classacc_ensemble.reset()
            meter_loss_ensemble.reset()

            timer_train.reset()
            state['iterator'] = tqdm(train_loader)

            if epoch in epoch_step:
                power = sum(epoch >= i for i in epoch_step)
                lr = opt.lr * pow(opt.lr_decay_ratio, power)
                lrm = opt.lrm * pow(opt.lr_decay_ratio, power)
                times = opt.times + 1
                state['optimizer'] = create_optimizer(opt, lr, lrg, times)

        def on_end_epoch(state):

            train_loss_ensemble= meter_loss_ensemble.value()
            train_acc_ensemble = classacc_ensemble.value()


            train_time = timer_train.value()

            meter_loss_ensemble.reset()
            classacc_ensemble.reset()


            timer_test.reset()

            engine.test(h_ensemble, test_loader)

            test_acc_ensemble = classacc_ensemble.value()[0]

            print(log_grassmann({
                "train_loss_total": train_loss_ensemble[0],
                "train_acc_ensemble": train_acc_ensemble[0],
                "test_loss_ensemble": meter_loss_ensemble.value()[0],
                "test_acc_ensemble": test_acc_ensemble,
                "epoch": state['epoch'],
                "num_classes": num_classes,
                # "n_parameters": n_parameters,
                "train_time": train_time,
                "test_time": timer_test.value(),
            }, state))

            print(log_oblique({
                "train_loss_ensemble": train_loss_ensemble[0],
                "train_acc_ensemble": train_acc_ensemble[0],
                "test_loss_ensemble": meter_loss_ensemble.value()[0],
                "test_acc_ensemble": test_acc_ensemble,
                "epoch": state['epoch'],
                "num_classes": num_classes,
                # "n_parameters": n_parameters,
                "train_time": train_time,
                "test_time": timer_test.value(),
            }, state))
            print(
                '==> id: %s (%d/%d), test_acc_ensemble: \33[91m%.2f\033[0m' % \
                (opt.save, state['epoch'], opt.epochs, test_acc_ensemble))

        engine = Engine()
        engine.hooks['on_sample'] = on_sample
        engine.hooks['on_forward'] = on_forward
        engine.hooks['on_start_epoch'] = on_start_epoch
        engine.hooks['on_end_epoch'] = on_end_epoch
        engine.hooks['on_start'] = on_start
        engine.train(h_ensemble, train_loader, opt.epochs, optimizer)

    else:
        f, params, stats = resnet.resnet(opt.depth, opt.width, num_classes)

        def create_optimizer(opt, lr):
            print('creating optimizer with lr = ', lr)
            if opt.optim_method == 'SGD':
                return torch.optim.SGD(params.values(), lr, weight_decay=opt.weightDecay)


        epoch = 0
        optimizer = create_optimizer(opt, opt.lr)

        if opt.resume != '':
            state_dict = torch.load(opt.resume)
            epoch = state_dict['epoch']
            params_tensors, stats = state_dict['params'], state_dict['stats']
            #        for k, v in params.iteritems():
            for k, v in list(params.items()):
                v.data.copy_(params_tensors[k])
            optimizer.load_state_dict(state_dict['optimizer'])

        meter_loss = tnt.meter.AverageValueMeter()
        classacc = tnt.meter.ClassErrorMeter(accuracy=True)

        timer_train = tnt.meter.TimeMeter('s')
        timer_test = tnt.meter.TimeMeter('s')

        if not os.path.exists(opt.save):
            os.mkdir(opt.save)

        def h(sample):
            inputs = Variable(cast(sample[0], opt.dtype))
            targets = Variable(cast(sample[1], 'long'))
            y = data_parallel(f, inputs, params, stats, sample[2], np.arange(opt.ngpu))
            return F.cross_entropy(y, targets), y

        def log(t, state):
            #        torch.save(dict(params={k: v.data for k, v in params.iteritems()},
            torch.save(dict(params={k: v.data for k, v in list(params.items())},
                            stats=stats,
                            optimizer=state['optimizer'].state_dict(),
                            epoch=t['epoch']),
                        open(os.path.join(opt.save, 'model.pt7'), 'wb'))
            z = vars(opt).copy();
            z.update(t)
            logname = os.path.join(opt.save, 'log.txt')
            with open(logname, 'a') as f:
                f.write('json_stats: ' + json.dumps(z) + '\n')
            print(z)

        def on_sample(state):
            state['sample'].append(state['train'])

        def on_forward(state):
            classacc.add(state['output'].data, torch.LongTensor(state['sample'][1]))
            meter_loss.add(state['loss'].data[0])

        def on_start(state):
            state['epoch'] = epoch

        def on_start_epoch(state):

            classacc.reset()
            meter_loss.reset()
            timer_train.reset()
            state['iterator'] = tqdm(train_loader)

            epoch = state['epoch'] + 1

            if epoch in epoch_step:
                power = sum(epoch >= i for i in epoch_step)
                lr = opt.lr * pow(opt.lr_decay_ratio, power)
                #lrg = opt.lrg * pow(opt.lr_decay_ratio, power)
                state['optimizer'] = create_optimizer(opt, lr)

#            lr = state['optimizer'].param_groups[0]['lr']
#            lrm = state['optimizer'].param_groups[0]['lrm']
#            state['optimizer'] = create_optimizer(opt, 
#                                          lr * opt.lr_decay_ratio, 
#                                          lrm * opt.lr_decay_ratio)

        def on_end_epoch(state):

            train_loss = meter_loss.value()
            train_acc = classacc.value()
            train_time = timer_train.value()
            meter_loss.reset()
            classacc.reset()
            timer_test.reset()

            engine.test(h, test_loader)
            test_acc = classacc.value()[0]
            print(log({
                "train_loss": train_loss[0],
                "train_acc": train_acc[0],
                "test_loss": meter_loss.value()[0],
                "test_acc": test_acc,
                "epoch": state['epoch'],
                "num_classes": num_classes,
                #"n_parameters": n_parameters,
                "train_time": train_time,
                "test_time": timer_test.value(),
            }, state))
            print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % \
                  (opt.save, state['epoch'], opt.epochs, test_acc))

        engine = Engine()
        engine.hooks['on_sample'] = on_sample
        engine.hooks['on_forward'] = on_forward
        engine.hooks['on_start_epoch'] = on_start_epoch
        engine.hooks['on_end_epoch'] = on_end_epoch
        engine.hooks['on_start'] = on_start
        engine.train(h, train_loader, opt.epochs, optimizer)




if __name__ == '__main__':
    main()
