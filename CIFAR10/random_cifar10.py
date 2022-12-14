'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import pytorchfi
# from pytorchfi.core import fault_injection as pfi_core
from pytorchfi.weight_error_models import random_weight_inj
# from pytorchfi.neuron_error_models import random_weight_inj
from pytorchfi.core import FaultInjection as pfi_core
# from pytorchfi.neuron_error_models import single_bit_flip_func, random_neuron_single_bit_inj_batched
#from Opt import opt
#from diffGrad import diffGrad
#from diffRGrad import diffRGrad, SdiffRGrad, BetaDiffRGrad, Beta12DiffRGrad, BetaDFCDiffRGrad
#from RADAM import Radam, BetaRadam
#from BetaAdam import BetaAdam, BetaAdam1, BetaAdam2, BetaAdam3, BetaAdam4, BetaAdam5, BetaAdam6, BetaAdam7, BetaAdam4A
#from AdamRM import AdamRM, AdamRM1, AdamRM2, AdamRM3, AdamRM4, AdamRM5
#from sadam import sadam
#from SdiffGrad import SdiffGrad
#from SRADAM import SRADAM

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate'); lr = '001'
parser.add_argument('--af', default = 'ReLU', help = 'activation_function')
parser.add_argument('--net', default = 'ResNet50', help = 'network')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Data
cust_min=0
cust_max=0.00001
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

bs = 100 #set batch size
reset_layer_range = 100
tb=8
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
#net = Elliott_VGG('VGG16'); net1 = 'Elliott_VGG16'
#net = GELU_MobileNet(); net1 = 'GELU_MobileNet'
#net = GELU_SENet18(); net1 = 'GELU_SENet18'
#net = PDELU_ResNet50(); net1 = 'PDELU_ResNet50'
# net = Sigmoid_GoogLeNet(); net1 = 'Sigmoid_GoogLeNet'
#net = GELU_DenseNet121(); net1 = 'GELU_DenseNet121'
network_name = args.af + "_"+args.net+"()"
net1 = args.af + "_"+args.net
print(network_name)
net = eval(network_name)

net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9); optimizer1 = 'SGDM5'
#optimizer = optim.Adagrad(net.parameters()); optimizer1 = 'AdaGrad'
#optimizer = optim.Adadelta(net.parameters()); optimizer1 = 'AdaDelta'
#optimizer = optim.RMSprop(net.parameters()); optimizer1 = 'RMSprop'
optimizer_net = optim.Adam(net.parameters(), lr=args.lr); optimizer1 = 'Adam'
#optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=True); optimizer1 = 'amsgrad'
#optimizer = diffGrad(net.parameters(), lr=args.lr); optimizer1 = 'diffGrad'
#optimizer = Radam(net.parameters(), lr=args.lr); optimizer1 = 'Radam'
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1, last_epoch=-1)
# [print(i.shape) for i in net.parameters()]
# exit(0)
# num = 0
# for i in net.modules():
#   if isinstance(i, torch.nn.Conv2d):
#   # if i.__class__.__name__ == "Conv2D":
#     num += 1

# layer_ranges =  [24.375, 26.375, 13.179688, 3.367188, 3.314453]
# layer_ranges = [13.179688 for i in range(num)]
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_layer_ranges(net, trainloader):
    model = net
    model_children = list(model.modules())
    model_children[0].register_forward_hook(get_activation('cnn_0'))
    conv_layers = [i for i in model_children if isinstance(i, torch.nn.Conv2d)]
    act_hook = [i.register_forward_hook(get_activation( "cnn_" + str(idx))) for idx, i in enumerate(conv_layers)]
    input1 = next(iter(trainloader))
    result = model(input1[0])
    [x.remove() for x in act_hook]
    return [1.5 * torch.max(torch.abs(activation["cnn_" + str(i)])).item() for i in range(len(conv_layers))]


# layer_ranges = get_layer_ranges(net, trainloader)
# inj_net_obj = pfi_core(net, bs, input_shape=[3, 32, 32], use_cuda=True,)
# inj_net = random_weight_inj(inj_net_obj, min_val=cust_min, max_val=cust_max)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/CIFAR10_B'+str(bs)+'_LR'+lr+'_'+net1+'_'+optimizer1+'.t7')
    net.load_state_dict(checkpoint['inj_net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

f = open('./Results/CIFAR10_B'+str(bs)+'_LR'+lr+'_'+net1+'_'+'.txt', 'w')

net_acc = []
pfi_acc = []
net_top2diff = []
pfi_top2diff = []
# Training
def train(epoch):
    print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    top2diff_sum , td_num = 0,0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_net.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_net.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # print(outputs[1])
        # exit(0)
        # print(predicted.eq(targets))
        for idx, out in enumerate(predicted.eq(targets)):
            if out == False:
                sort_out,_ = outputs[idx].sort(descending = True)
                # print(sort_out)
                # exit(0)
                top2diff_sum += (sort_out[0].item() - sort_out[1].item())
                td_num +=1
        # exit()
        if td_num == 0:
            Top2Diff = 0
        else:
            Top2Diff = top2diff_sum/td_num
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Top2Diff: %.3f'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total, Top2Diff))
        if (batch_idx + 1) == len(trainloader):
            f.write('Train | Epoch: %d | Loss: %.3f | Acc: %.3f\n'
                % (epoch, train_loss / (batch_idx + 1), 100. * correct / total))
    net_acc.append(correct / total)
    net_top2diff.append(Top2Diff)
    return train_loss / (batch_idx + 1)

def pfi_train(epoch, inj_net, optimizer):
    print('\nPFI Epoch: %d' % epoch)
    # layer_ranges = get_layer_ranges(net, trainloader)
    # inj_net_obj = pfi_core(net, bs, input_shape=[3, 32, 32], use_cuda=True,)
#     inj_net = random_weight_inj(inj_net_obj, min_val=cust_min, max_val=cust_max)
#     opt_st = optimizer.state_dict()
#     optimizer = optim.Adam(inj_net.parameters(), lr=args.lr)
#     optimizer.load_state_dict(opt_st)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1, last_epoch=-1)
#     inj_net_obj.reset_fault_injection()
    # net_modules = [m for m in net.modules() if m.__class__.__name__.__contains__("onv2d")]
    # inj_net_modules = [m for m in inj_net.modules() if m.__class__.__name__.__contains__("onv2d")]
    # z = []
    # for idx, m in enumerate(net_modules):
    #     z.append((net_modules[idx].weight.data.clone().detach() - inj_net_modules[idx].weight.data.clone().detach()).abs().sum())

    # print(z)
    # exit(0)
  
    inj_net.train()
    for _ in range(epoch):
        scheduler.step()
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    correct = 0
    total = 0
    t2d = 0
    top2diff_sum , td_num = 0,0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # if batch_idx%reset_layer_range == 0:
        #     layer_ranges = get_layer_ranges(net, trainloader)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = inj_net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        for idx, out in enumerate(predicted.eq(targets)):
            if out == False:
                sort_out,_ = outputs[idx].sort(descending = True)
                top2diff_sum += (sort_out[0].item() - sort_out[1].item())
                td_num +=1

        if td_num == 0:
            Top2Diff = 0
        else:
            Top2Diff = top2diff_sum/td_num
        t2d += Top2Diff
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Top2Diff: %.3f'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total, Top2Diff))

        if (batch_idx + 1) == len(trainloader):
            f.write('Train | Epoch: %d | Loss: %.3f | Acc: %.3f\n'
                % (epoch, train_loss / (batch_idx + 1), 100. * correct / total))
    pfi_acc.append(correct / total)
    pfi_top2diff.append(Top2Diff)
    return train_loss / (batch_idx + 1)

test_acc = []
def test(epoch, inj_net):
    global best_acc
    # layer_ranges = get_layer_ranges(net, trainloader)
    # inj_net_obj = pfi_core(net, bs, input_shape=[3, 32, 32], use_cuda=True,)
#     inj_net = random_weight_inj(inj_net_obj, min_val=cust_min, max_val=cust_max)
    inj_net.eval()
    # net_modules = [m for m in net.modules() if m.__class__.__name__.__contains__("onv2d")]
    # inj_net_modules = [m for m in inj_net.modules() if m.__class__.__name__.__contains__("onv2d")]
    # z = []
    # for idx, m in enumerate(net_modules):
    #     if (net_modules[idx].weight.data.clone().detach() - inj_net_modules[idx].weight.data.clone().detach()).abs().sum() > 0 :
    #         ab = (net_modules[idx].weight.data.clone().detach() - inj_net_modules[idx].weight.data.clone().detach()).abs()
    #         ab[ab > 0] = 1
    #         z.append(ab.sum())

    # print(z) 
    # exit(0)
    test_loss = 0
    correct = 0
    total = 0
    top2diff_sum , td_num = 0,0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # if batch_idx%10 == 0:
            #     layer_ranges = get_layer_ranges(net, trainloader)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = inj_net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            for idx, out in enumerate(predicted.eq(targets)):
                if out == False:
                    sort_out,_ = outputs[idx].sort(descending = True)
                    top2diff_sum += (sort_out[0].item() - sort_out[1].item())
                    td_num +=1

            if td_num == 0:
                Top2Diff = 0
            else:
                Top2Diff = top2diff_sum/td_num
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Top2Diff: %.3f'
                        % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total, Top2Diff))
            if (batch_idx + 1) == len(testloader):
                f.write('Test | Epoch: %d | Loss: %.3f | Acc: %.3f\n'
                    % (epoch, test_loss / (batch_idx + 1), 100. * correct / total))
    # Save checkpoint.
    acc = 100. * correct / total
    test_acc.append(acc)
    if acc > best_acc:
        print('Saving..')
        state = {
            'inj_net': inj_net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/CIFAR10_B'+str(bs)+'_LR'+lr+'_'+net1+'_'+optimizer1+'.t7')
        best_acc = acc

#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150,180], gamma=0.1, last_epoch=-1)
error_inj_freq = 1
del_loss = []
for epoch in range(100):
#for epoch in range(start_epoch, 200):
#     scheduler.step()
    if epoch ==2:
      inj_net_obj = pfi_core(net, bs, input_shape=[3, 32, 32], use_cuda=True,)
      inj_net = random_weight_inj(inj_net_obj, min_val=cust_min, max_val=cust_max)
      optimizer = optim.Adam(inj_net.parameters(), lr=args.lr) 
    if epoch <2:
      train(epoch)
    else:
      if epoch%error_inj_freq==0:
          inj_net_obj = pfi_core(inj_net, bs, input_shape=[3, 32, 32], use_cuda=True,)
          inj_net = random_weight_inj(inj_net_obj, min_val=cust_min, max_val=cust_max)
          optimizer = optim.Adam(inj_net.parameters(), lr=args.lr) 
      inj_net_loss = pfi_train(epoch, inj_net, optimizer)
      test(epoch, inj_net)
      norm_net_loss = train(epoch)
      print(norm_net_loss - inj_net_loss)
      del_loss.append(norm_net_loss - inj_net_loss)
      

f.write('Best Accuracy:  %.3f\n'
    % (best_acc))
f.close()

print("Best Accuracy: ", best_acc)

print(net_acc,"\n", net_top2diff, "\n", pfi_acc, "\n", pfi_top2diff, "\n", test_acc, "\n", del_loss)
