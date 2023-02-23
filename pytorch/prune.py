import os
import copy
import torch
import torch.nn.utils.prune as prune
from utils import set_random_seeds, create_model, prepare_dataloader, train_model, save_model, load_model, evaluate_model, create_classification_report,  measure_global_sparsity, measure_module_sparsity


def iterative_pruning_finetuning(model,
                                 train_loader,
                                 test_loader,
                                 device,
                                 learning_rate,
                                 l1_regularization_strength,
                                 l2_regularization_strength,
                                 learning_rate_decay=0.1,
                                 conv2d_prune_amount=0.4,
                                 linear_prune_amount=0.2,
                                 num_iterations=5,
                                 num_epochs_per_iteration=10,
                                 model_filename_prefix="pruned_model",
                                 model_dir="saved_models",
                                 grouped_pruning=False):

    for i in range(num_iterations):

        print("Pruning and Finetuning {}/{}".format(i + 1, num_iterations))

        print("Pruning...")

        if grouped_pruning == True:
            # Global pruning
            # I would rather call it grouped pruning.
            parameters_to_prune = []
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, "weight"))
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=conv2d_prune_amount,
            )
        else:
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module,
                                          name="weight",
                                          amount=conv2d_prune_amount)
                    
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module,
                                          name="weight",
                                          amount=linear_prune_amount)

        _, eval_accuracy = evaluate_model(model=model,
                                          test_loader=test_loader,
                                          device=device,
                                          criterion=None)

        classification_report = create_classification_report(
            model=model, test_loader=test_loader, device=device)

        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model,
            weight=True,
            bias=False,
            conv2d_use_mask=True,
            linear_use_mask=False)

        print("Test Accuracy: {:.3f}".format(eval_accuracy))
        print("Classification Report:")
        print(classification_report)
        print("Global Sparsity:")
        print("{:.2f}".format(sparsity))

        # print(model.conv1._forward_pre_hooks)

        print("Fine-tuning...")

        train_model(model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                    l1_regularization_strength=l1_regularization_strength,
                    l2_regularization_strength=l2_regularization_strength,
                    learning_rate=learning_rate * (learning_rate_decay**i),
                    num_epochs=num_epochs_per_iteration)

        _, eval_accuracy = evaluate_model(model=model,
                                          test_loader=test_loader,
                                          device=device,
                                          criterion=None)

        classification_report = create_classification_report(
            model=model, test_loader=test_loader, device=device)

        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model,
            weight=True,
            bias=False,
            conv2d_use_mask=True,
            linear_use_mask=False)

        print("Test Accuracy: {:.3f}".format(eval_accuracy))
        print("Classification Report:")
        print(classification_report)
        print("Global Sparsity:")
        print("{:.2f}".format(sparsity))

        model_filename = "{}_{}.pt".format(model_filename_prefix, i + 1)
        print(model_filename)
        model_filepath = os.path.join(model_dir, model_filename)
        
        save_model(model=model,
                   model_dir=model_dir,
                   model_filename=model_filename)
        # model = load_model(model=model,
        #                    model_filepath=model_filepath,
        #                    device=device)

    return model


def remove_parameters(model):
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) :
            try:
                prune.remove(module, "weight")
                prune.remove(module, "bias")
            except:
                pass
    return model

def prune_model(model, args, model_name="ResNet18"):
    num_classes = 10
    random_seed = 1
    l1_regularization_strength = 0
    l2_regularization_strength = 1e-4
    learning_rate = 1e-3
    learning_rate_decay = 1

    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_dir = "saved_models"
    model_filename = f"""{model_name}.pt"""
    pruned_model_filename = f"""pruned_{model_filename}"""
    model_filepath = os.path.join(model_dir, model_filename)
    pruned_model_filepath = os.path.join(model_dir, pruned_model_filename)
    # wandb.config = {
    # "learning_rate": learning_rate,
    # "epochs": 10,
    # "batch_size":128
    # }
    set_random_seeds(random_seed=random_seed)
    # Load a pretrained model.
    model = load_model(model=model,
                       model_filepath=model_filepath,
                       device=cuda_device)

    train_loader, test_loader, classes = prepare_dataloader(
        num_workers=8, train_batch_size=128, eval_batch_size=128)

    _, eval_accuracy = evaluate_model(model=model,
                                      test_loader=test_loader,
                                      device=cuda_device,
                                      criterion=None)

    classification_report = create_classification_report(
        model=model, test_loader=test_loader, device=cuda_device)

    num_zeros, num_elements, sparsity = measure_global_sparsity(model)

    print("Test Accuracy: {:.3f}".format(eval_accuracy))
    print("Classification Report:")
    print(classification_report)
    print("Global Sparsity:")
    print("{:.2f}".format(sparsity))

    print("Iterative Pruning + Fine-Tuning...")

    pruned_model = copy.deepcopy(model)

    pruned_model = iterative_pruning_finetuning(
        model=pruned_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=cuda_device,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        l1_regularization_strength=l1_regularization_strength,
        l2_regularization_strength=l2_regularization_strength,
        conv2d_prune_amount=0.4,
        linear_prune_amount=0.2,
        num_iterations=5,
        num_epochs_per_iteration=10,
        model_dir=model_dir,
        grouped_pruning=True)

    # Apply mask to the parameters and remove the mask.
    remove_parameters(model=pruned_model)

    _, eval_accuracy = evaluate_model(model=pruned_model,
                                      test_loader=test_loader,
                                      device=cuda_device,
                                      criterion=None)

    classification_report = create_classification_report(
        model=pruned_model, test_loader=test_loader, device=cuda_device)

    num_zeros, num_elements, sparsity = measure_global_sparsity(pruned_model)

    print("Test Accuracy: {:.3f}".format(eval_accuracy))
    print("Classification Report:")
    print(classification_report)
    print("Global Sparsity:")
    print("{:.2f}".format(sparsity))
    
    save_model(model=model, model_dir=pruned_model_filepath, model_filename=pruned_model_filename)
    print("pruning done succuessfully")

if __name__ == "__main__":

    prune_model(model, args)
