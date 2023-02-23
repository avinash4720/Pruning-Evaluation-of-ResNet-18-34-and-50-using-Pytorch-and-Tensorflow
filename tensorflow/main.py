
from resnet import ResNet18, ResNet34, ResNet50


def main():
  models = [ResNet18, ResNet34, ResNet50]
  for model in models:
    train(model)
if __name__ == '__main__':
  main()