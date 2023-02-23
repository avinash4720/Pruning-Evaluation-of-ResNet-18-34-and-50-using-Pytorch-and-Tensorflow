import os
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

'''Train CIFAR10 with PyTorch.'''
import torchvision
import torchvision.transforms as transforms

import argparse


from utils import set_random_seeds, prepare_dataloader, train_model, save_model, load_model, evaluate_model, create_classification_report, measure_global_sparsity, measure_module_sparsity
from resnet import ResNet18, ResNet34, ResNet50



def train_loop(dataloader, device, model, loss_fn, optimizer, l1_regularization_strength):
    size = len(dataloader.dataset)
    running_loss = 0
    running_accurate = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X=X.to(device)
        y=y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()

        for module in model.modules():
            mask=None
            weight=None
            for name,buffer in module.named_buffers():
                if name == "weight_mask":
                    mask = buffer
            for name, param in module.named_parameters():
                if name == "weight_orig":
                    weight = param

            if mask and weight:
                loss += (l1_regularization_strength) * torch.norm(mask*weight, 1)
                

        loss.backward()
        optimizer.step()

        running_loss += loss.item()*len(X)
        running_accurate +=(pred.argmax(1) == y).type(torch.float).sum().item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")    # uncomment to see mini batch loss

    training_loss = running_loss/size
    training_accuracy = running_accurate/size

    return training_accuracy,  training_loss

def test_loop(dataloader, device, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X=X.to(device)
            y=y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    return (correct, test_loss)

def train(model,args, l1_regularization_strength = 0, l2_regularization_strength = 1e-4, learning_rate = 1e-1, num_epochs = 10):
    num_classes = 10

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=args.datapath, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(
        root=args.datapath, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    if args.cuda:
        if torch.cuda.is_available():
            device="cuda"
        else:
            device="cpu"
    else:
        device="cpu"


    if torch.cuda.device_count() > 1:
      print("Using", torch.cuda.device_count(), "GPUs!")

    net = model()
    net = net.to(device)

    if device=="cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer_map = {
        "SGD": optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=l2_regularization_strength),
        "ADAGRAD": optim.Adagrad(net.parameters(), lr=learning_rate, weight_decay=l2_regularization_strength),
        "NESTEROV": optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=l2_regularization_strength, nesterov=True),
        "ADADELTA":optim.Adadelta(net.parameters(), lr=learning_rate, weight_decay=l2_regularization_strength),
        "ADAM":optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2_regularization_strength),
    }

    criterion = nn.CrossEntropyLoss()

    optimizer = optimizer_map.get(args.optimizer,  optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=l2_regularization_strength))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,"min", patience=0, factor=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1, last_epoch=-1)
    for epoch in range(num_epochs):
        (train_accuracy, train_loss) = train_loop(trainloader, device, net, criterion, optimizer, l1_regularization_strength)
        (eval_accuracy, eval_loss) = test_loop(testloader, device,  net, criterion)
        scheduler.step()
        print("Epoch: {:03d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(epoch + 1, train_loss, train_accuracy, eval_loss, eval_accuracy))

    classification_report = create_classification_report(model=net, test_loader=testloader, device=device)

    print("Test Accuracy: {:.3f}".format(eval_accuracy))
    print("Classification Report:")
    print(classification_report)

    return net

def pretrain_model(model, args):
    model_dir = "saved_models"

    if model.__name__ == 'ResNet18':
        print("model trained is resnet 18")
        model_filename = "{}.pt".format("ResNet18")
    elif model.__name__ == 'ResNet34':
        print("model trained is resnet 34")
        model_filename = "{}.pt".format("ResNet34")
    elif model.__name__ == 'ResNet50':
        print("model trained is resnet 50")
        model_filename = "{}.pt".format("ResNet50")
    else:
        print("something wrong")
    model = train(model, args)
    save_model(model=model, model_dir=model_dir, model_filename=model_filename)
    return model