# HPML_project - Pruning Evaluation using Pytorch and Tensorflow
## About
The Goal of the project is to determine, which Framework will yield good results on inference models with smaller size

## Execution

There are two parts involved in the project

Part 1: Execution of pytorch models

Part 2: Execution of Tensor flow models

---------------
Step 1: Login to hpc cluster

Step 2: pull code from github repo ```https://github.com/avinash4720/Pruning-Evaluation-of-ResNet-18-34-and-50-using-Pytorch-and-Tensorflow.git```

Step 3: submit pytorch batch job - ```sbatch pytorch/pytorch_job.sh ```

Step 4: submit tensorflow batch job- ```sbatch tensorflow/tensorflow_job.sh```

Step 5: Respective output files and models along with pruned models are created 

## Dependencies
```
python
tensorflow
pytorch
torchvision
```

## Results and output
<img width="347" alt="image" src="https://user-images.githubusercontent.com/91353137/209167846-7ff76c99-89f0-43cc-82b0-4a06ff12dd04.png">
<img width="693" alt="image" src="https://user-images.githubusercontent.com/91353137/209167942-37e83abc-f353-423c-991b-68c7cf2dba05.png">
<img width="704" alt="image" src="https://user-images.githubusercontent.com/91353137/209167988-37b7beff-d846-4fae-ba64-5730aa8af43c.png">
<img width="733" alt="image" src="https://user-images.githubusercontent.com/91353137/209168038-82fa7dd1-841c-413a-8233-d6ab663a3cfc.png">
<img width="736" alt="image" src="https://user-images.githubusercontent.com/91353137/209168092-bcf4dbc7-0ec6-4bd7-95b0-f18ea8734a24.png">
<img width="749" alt="image" src="https://user-images.githubusercontent.com/91353137/209168131-99537686-810d-4c3c-9125-e41699f3ea5b.png">




