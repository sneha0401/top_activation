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
from resnet import ResNet18
# from pytorchfi.core import fault_injection as pfi_core
# from pytorchfi.weight_error_models import random_neuron_inj
from pytorchfi.neuron_error_models import random_neuron_inj
from pytorchfi.neuron_error_models import random_inj_per_layer

from pytorchfi.core import FaultInjection as pfi_core
# from pytorchfi.neuron_error_models import single_bit_flip_func, random_neuron_single_bit_inj_batched

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
cust_max=0.0001
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=1)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=1)

# Model
print('==> Building model..')
network_name = args.af + "_"+args.net+"()"
net1 = args.af + "_"+args.net
print(network_name)
# net = eval(network_name)
net = ResNet18()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

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


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/CIFAR10_B'+str(bs)+'_LR'+lr+'_'+net1+'_'+optimizer1+'.t7')
    net.load_state_dict(checkpoint['inj_net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

f = open('./Results/CIFAR10_B'+str(bs)+'_LR'+lr+'_'+net1+'_'+'.txt', 'w')

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    top2diff_sum , td_num = 0,0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
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
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Top2Diff: %.3f'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total, top2diff_sum/td_num))
        if (batch_idx + 1) == len(trainloader):
            f.write('Train | Epoch: %d | Loss: %.3f | Acc: %.3f\n'
                % (epoch, train_loss / (batch_idx + 1), 100. * correct / total))

#         #### COMMENT FOR ERROR 4
# def pfi_train(epoch, inj_net, optimizer):
#         ####
        #### REMOVE COMMENT FOR ERROR 4
def pfi_train(epoch, inj_net_obj, optimizer):
        ####
    print('\nEpoch: %d' % epoch)
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    correct = 0
    total = 0
    top2diff_sum , td_num = 0,0
    #     #### COMMENT FOR ERROR 4
    # inj_net.train()
    #     ####
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #### REMOVE COMMENT FOR ERROR 4
        inj_net = random_inj_per_layer(inj_net_obj, min_val=cust_min, max_val=cust_max)
        optimizer = optim.Adam(inj_net.parameters(), lr=args.lr) 
        inj_net.train()
        ####
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Top2Diff: %.3f'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total, top2diff_sum/td_num))
        if (batch_idx + 1) == len(trainloader):
            f.write('Train | Epoch: %d | Loss: %.3f | Acc: %.3f\n'
                % (epoch, train_loss / (batch_idx + 1), 100. * correct / total))

#         #### COMMENT FOR ERROR 4
# def test(epoch, inj_net):
#         ####
        #### REMOVE COMMENT FOR ERROR 4
def test(epoch, inj_net_obj):
#         ####
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
# #         #### COMMENT FOR ERROR 4
#     inj_net.eval()
# #         ####
    top2diff_sum , td_num = 0,0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            #### REMOVE COMMENT FOR ERROR 4
            inj_net = random_inj_per_layer(inj_net_obj, min_val=cust_min, max_val=cust_max)
            optimizer = optim.Adam(inj_net.parameters(), lr=args.lr) 
            inj_net.eval()
            ####
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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Top2Diff: %.3f'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total, top2diff_sum/td_num))
            if (batch_idx + 1) == len(testloader):
                f.write('Test | Epoch: %d | Loss: %.3f | Acc: %.3f\n'
                    % (epoch, test_loss / (batch_idx + 1), 100. * correct / total))
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'inj_net': inj_net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/CIFAR10_B'+str(bs)+'_LR'+lr+'_'+net1+'_'+'adam'+'.t7')
        best_acc = acc
'''
Summary of problems
Error 1 : inj_net cannot be used for more than one inference
Error 2 : inj_net needs to be re-initialized from inj_net_obj for every inference
Error 3 : Due to error 1, if we take the method from (https://github.com/jose-sv/pytorchfi_train/blob/master/main.py#L151)
          code gives error, as inj_net is not re-initialized before every iteration
Error 4 : If we re-initialize the inj_net before every inference rrom inj_net_obj
          training does not work (gradients do not flow correctly.)
'''
error_inj_freq = 1
for epoch in range(100):
    ##################################################################################
    #### Error 1 --> Inj_Net cannot be reused for inference
    # inj_net_obj = pfi_core(net, bs, input_shape=[3, 32, 32], use_cuda=True,)
    # inj_net = random_inj_per_layer(inj_net_obj, min_val=cust_min, max_val=cust_max)
    # print(inj_net(torch.randn(100, 3, 32, 32).cuda()).shape)
    # print(inj_net(torch.randn(100, 3, 32, 32).cuda()).shape)
    # print(inj_net(torch.randn(100, 3, 32, 32).cuda()).shape)
    # print(inj_net(torch.randn(100, 3, 32, 32).cuda()).shape)
    # exit(0)
    ##################################################################################
    #### Error 2 --> Reinitializing inj_net after every inference works.
    # inj_net_obj = pfi_core(net, bs, input_shape=[3, 32, 32], use_cuda=True,)
    # inj_net = random_inj_per_layer(inj_net_obj, min_val=cust_min, max_val=cust_max)
    # print(inj_net(torch.randn(100, 3, 32, 32).cuda()).shape)
    # inj_net = random_inj_per_layer(inj_net_obj, min_val=cust_min, max_val=cust_max)
    # print(inj_net(torch.randn(100, 3, 32, 32).cuda()).shape)
    # inj_net = random_inj_per_layer(inj_net_obj, min_val=cust_min, max_val=cust_max)
    # print(inj_net(torch.randn(100, 3, 32, 32).cuda()).shape)
    # inj_net = random_inj_per_layer(inj_net_obj, min_val=cust_min, max_val=cust_max)
    # print(inj_net(torch.randn(100, 3, 32, 32).cuda()).shape)
    # exit(0)
    ##################################################################################
    #### Error 3 --> As done in (https://github.com/jose-sv/pytorchfi_train/blob/master/main.py#L151)
    #### We initialize the inj_net once before every epoch training
    #### code does not run, because inj_net needs to be reinitialized before every iteration
    # inj_net_obj = pfi_core(net, bs, input_shape=[3, 32, 32], use_cuda=True,)
    # inj_net = random_inj_per_layer(inj_net_obj, min_val=cust_min, max_val=cust_max)
    # optimizer = optim.Adam(inj_net.parameters(), lr=args.lr) 
    # if epoch < 0:
    #   train(epoch)
    # else:
    #   pfi_train(epoch, inj_net, optimizer)
    #   test(epoch, inj_net)
    ##################################################################################
    #### Error 4 --> As done in (https://github.com/jose-sv/pytorchfi_train/blob/master/main.py#L151)
    #### We initialize the inj_net_obj once before every epoch training
    #### BUT we initialize inj_net EVERY ITERATION
    #### THIS does not train
    inj_net_obj = pfi_core(net, bs, input_shape=[3, 32, 32], use_cuda=True,)
    if epoch < 0:
      train(epoch)
    else:
      pfi_train(epoch, inj_net_obj, optimizer)
      test(epoch, inj_net_obj)
    ##################################################################################

# error_inj_freq = 1
# for epoch in range(100):
#     if epoch ==0:
#       inj_net_obj = pfi_core(net, bs, input_shape=[3, 32, 32], use_cuda=True,)
#       inj_net = random_neuron_inj(inj_net_obj, min_val=cust_min, max_val=cust_max)
#       optimizer = optim.Adam(inj_net.parameters(), lr=args.lr) 
#     if epoch <0:
#       train(epoch)
#     else:
#       if epoch%error_inj_freq==0:
#           inj_net_obj = pfi_core(inj_net, bs, input_shape=[3, 32, 32], use_cuda=True,)
#           inj_net = random_neuron_inj(inj_net_obj, min_val=cust_min, max_val=cust_max)
#           print(inj_net(torch.randn(100,3,32,32).cuda()).shape)
#           optimizer = optim.Adam(inj_net.parameters(), lr=args.lr) 
#       pfi_train(epoch, inj_net, optimizer)
#       test(epoch, inj_net)

f.write('Best Accuracy:  %.3f\n'
    % (best_acc))
f.close()

print("Best Accuracy: ", best_acc)
