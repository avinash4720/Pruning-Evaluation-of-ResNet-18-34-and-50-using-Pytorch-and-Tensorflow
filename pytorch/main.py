from pretrain import pretrain_model
import argparse
from prune import prune_model
#from helpers import show_stats
from resnet import ResNet18, ResNet34, ResNet50

def main():
    models = [ResNet18]
    #ResNet50, ResNet34, 
    parser = argparse.ArgumentParser(description='CNN CIFAR10 with Pytorch')
    parser.add_argument('--cuda', help='--cuda to enable cuda, --no-cuda to disable cuda', action=argparse.BooleanOptionalAction)
    parser.add_argument('--datapath', help='provide data path for the dataset',default="/scratch/pp2603/")
    parser.add_argument('--workers',help='number of workes',default=2, type=int)
    parser.add_argument('--optimizer', help='optimizer for the cnn', default='sgd')
    parser.add_argument('--batchsize',help="batch size of the epoch",default=128, type=int)

    args = parser.parse_args()
    print(f"""Device is {"CUDA" if args.cuda else "CPU"}""")
    print(f"Number of workers: {args.workers}")
    print(f"Data Path of the dataset:{args.datapath}")
    print(f"Optimizer used:{args.optimizer}")
    print(f"Batch Size:{args.batchsize}")

    for model in models:
        pt_model = pretrain_model(model, args)
        # prn_model = prune_model(pt_model, args, model_name=model.__name__)
        #show_stats(pt_model, prn_model)


if __name__ == '__main__':
    main()
