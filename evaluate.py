import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


from train import test
from dataset import sample_loader
from mia import mia_crossval, mia_true_nonmember_rate
import utils

device = utils.device_config()
utils.set_seed()


def retain_top_k_max_indices(arr, k):
    # Find the indices of the top n maximum values
    top_k_indices = np.argpartition(arr, -k, axis=None)[-k:]

    return top_k_indices


def compute_logits(model, inputs):
    """get modellogits for module.nn output or hugging face ImageClassifierOutput

    Args:
        model
        inputs

    Returns:
        torch: logits
    """
    outputs = model(inputs)
    # hugging face ImageClassifierOutput
    if hasattr(outputs, "logits"):
        outputs = outputs.logits

    return outputs


def compute_grad(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    lastlayer_grad = []

    # Get last layer's weights
    last_layer_weights = list(model.parameters())[-1]

    # Set the model to evaluation mode
    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        logits = compute_logits(model, inputs)
        loss = criterion(logits, labels)
        loss.backward(retain_graph=True)

        # Save the gradients for the last layer's weights
        lastlayer_grad.append(last_layer_weights.grad.clone().detach().cpu().numpy())

        # Reset gradients for the next batch
        last_layer_weights.grad.zero_()

        # just compute grads for one batch
        break

    return np.concatenate(lastlayer_grad, axis=0)


def compare_grad(modellist, modelnames, forget_loader, retain_loader, unseen_loader):
    grads_forget = {
        modelnames[i]: compute_grad(modellist[i], forget_loader)
        for i in range(len(modellist))
    }

    grads_retain = {
        modelnames[i]: compute_grad(modellist[i], retain_loader)
        for i in range(len(modellist))
    }

    grads_unseen = {
        modelnames[i]: compute_grad(modellist[i], unseen_loader)
        for i in range(len(modellist))
    }

    # retain grads of same subset of last layer neurons
    k = 50
    if len(list(grads_forget.values())[0]) > k:
        top_k_max_indices = retain_top_k_max_indices(grads_forget[modelnames[0]], k)
        for modelname in modelnames:
            for grad_list in (grads_forget, grads_retain, grads_unseen):
                grad_list[modelname] = grad_list[modelname].flat[top_k_max_indices]

    for i, name in enumerate(modelnames):
        _, ax = plt.subplots(figsize=(8, 4))
        bar_width = 0.25
        num_params = grads_retain[modelnames[0]].shape[0]
        indices = np.arange(num_params)
        offset = bar_width

        ax.bar(
            indices,
            grads_forget[name],
            bar_width,
            label="forgetten set",
            color=f"C1",
        )
        ax.bar(
            indices + offset,
            grads_retain[name],
            bar_width,
            label="retain set",
            color="C4",
        )
        ax.bar(
            indices + 2 * offset,
            grads_unseen[name],
            bar_width,
            label="unseen set",
            color="C7",
        )

        ax.set_xlabel("Gradient at Same Neuron")
        ax.set_ylabel("Gradient Value")
        ax.set_title(name)
        ax.legend()
        plt.show()


def compute_accuracy(model, loader):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            logits = compute_logits(model, inputs)
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return 100 * correct / total


def compare_accuracy(
    model_list, model_names, forget_loader, retain_loader, test_loader, unseen_loader
):
    """compute top1 accuracy

    Args:
        model_list:
        model_names :
        forget_loader :
        retain_loader :
        test_loader :
        unseen_loader :
    """
    # Dictionary to hold accuracy results
    results = {
        "Top1 acc on retain set": [],
        "Top1 acc on forget set": [],
        "Top1 acc on test set": [],
        "Top1 acc on unseen set": [],
    }

    # Calculate accuracies for each model
    for model in model_list:
        if retain_loader:
            results["Top1 acc on retain set"].append(
                compute_accuracy(model, retain_loader)
            )
        if forget_loader:
            results["Top1 acc on forget set"].append(
                compute_accuracy(model, forget_loader)
            )
        if test_loader:
            results["Top1 acc on test set"].append(compute_accuracy(model, test_loader))
        if unseen_loader:
            results["Top1 acc on unseen set"].append(
                compute_accuracy(model, unseen_loader)
            )

    # Printing results in a tabular format
    print("{:<30}".format(" "), end="")
    for name in model_names:
        print("{:<20}".format(name), end="")
    print()

    for key in results:
        print("{:<30}".format(key), end="")
        for value in results[key]:
            print("{:<20}".format(f"{value:.2f}%"), end="")
        print()


def compare_relearn_time(
    model_list,
    model_names,
    forget_loader,
    test_loader,
    lr,
    momentum,
    optimizer_fn,
    acc_threshold=0.97,
):
    """evaluate how long(in epochs) take model to relearn
    for the accuracy on forgotten data to reach a fixed threshold or more than accuracy on test

    Args:
        model_list :
        model_names :
        forget_loader :
        test_loader :
        lr :
        momentum :
        optimizer_fn :
        acc_threshold (float, optional): . Defaults to 0.97.
    """
    # Ensure that the models are in training mode
    for model in model_list:
        model.to(device)
        model.train()

    # List to store the number of epochs required to reach the loss threshold for each model
    relearn_times = []

    # Define a loss function, assuming a classification task with CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss()

    # Iterate through each model
    for idx, model in enumerate(model_list):
        epochs = 0
        optimizer = optimizer_fn(model.parameters(), lr=0.01 * lr, momentum=momentum)

        while True:
            # evaluate if relearned
            print(f"forget acc, test acc")
            acc_forget = test(model, forget_loader)
            acc_test = test(model, test_loader)
            if acc_forget >= acc_threshold or acc_forget >= acc_test:
                print(f"Model {model_names[idx]} relearned in {epochs} epochs.")
                break

            # Loop until the threshold is reached
            for inputs, targets in forget_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = compute_logits(model, inputs)
                loss = criterion(logits, targets)

                # Backpropagation and optimization step
                model.zero_grad()
                loss.backward()
                optimizer.step()  # Assume an optimizer is defined for each model

        epochs += 1  # Increment epoch count after processing the entire dataset

        relearn_times.append(epochs)

    # Plotting the histogram
    plt.figure(figsize=(10, 5))
    plt.bar(model_names, relearn_times, color="skyblue")
    plt.ylabel("Relearning Time (in epochs)")
    plt.title("Model Relearning Times to Reach Accuracy Threshold")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def compute_batch_entropy(model, loader):
    """compute entropy of logits for each sample

    Args:
        model :
        loader (Dataloader):

    Returns:
        list: entropies for each sample
    """
    entropies = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)

            logits = compute_logits(model, inputs)

            probabilities = F.softmax(logits, dim=1)  # batch_size * [...p_i ...]
            log_prob = torch.log(probabilities)
            batch_entropy = -(probabilities * log_prob).sum(dim=1)  # batch_size * [e_j]
            entropies.extend(batch_entropy.tolist())

    return entropies


def compare_entropy(
    model_list,
    model_names,
    loaders,
    dataset_names=["Forgotten", "Training members", "Non-Training"],
    bins=30,
):
    """
    Compare entropy distributions across different datasets and models.

    Args:
        model_list (list of nn.Module): List of models to evaluate.
        model_names (list of str): Corresponding names for the models.
        data_loaders:

        bins (int): Number of bins for the histograms.
    """

    # Dictionary to store entropies: {model_name: {dataset_name: entropies}}
    entropies_dict = {model_name: {} for model_name in model_names}

    # Compute and store entropies for all models and datasets
    all_entropies = []  # To determine global min and max
    for model, model_name in zip(model_list, model_names):
        for loader, dataset_name in zip(loaders, dataset_names):
            entropies = compute_batch_entropy(model, loader)
            entropies_dict[model_name][dataset_name] = entropies
            all_entropies.extend(entropies)

    # Determine global min and max entropy for consistent binning
    global_min = min(all_entropies)
    global_max = max(all_entropies)
    bin_edges = np.linspace(global_min, global_max, bins + 1)

    # Iterate over each model to plot histograms
    for model_name in model_names:
        _, ax = plt.subplots(figsize=(6, 4))
        model_entropies = entropies_dict[model_name]

        # Plot each dataset's entropy histogram
        for idx, dataset_name in enumerate(dataset_names):
            entropies = model_entropies[dataset_name]

            # Calculate weights to normalize histogram
            weights = np.ones_like(entropies) / len(entropies)

            ax.hist(
                entropies,
                bins=bin_edges,  # Use common bin edges
                weights=weights,
                edgecolor="black",
                alpha=1 - 0.3 * idx,
                color=f"C{idx*3+1}",
                label=dataset_name,
            )

        # Set titles, labels, and formatting
        ax.set_title(f"{model_name}")
        ax.set_xlabel("Entropy (confidence of prediction: confident -> uncertain)")
        ax.set_ylabel("Frequency")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        if model_name == model_names[0]:
            ax.legend()

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()


def compare_entropy_old(
    model_list, model_names, forget_loader, retain_loader, unseen_loader
):
    # Data loaders and their corresponding colors and labels
    loaders = [forget_loader, retain_loader, unseen_loader]
    dataset_names = ["Forgetton", "Training members", "Non-Training"]

    # Initialize a dictionary to store entropies for all datasets
    all_entropies = {name: [] for name in dataset_names}

    # Compute entropies for all models and datasets
    for model in model_list:
        model.eval()
        for idx, loader in enumerate(loaders):
            entropies = compute_batch_entropy(model, loader)
            all_entropies[dataset_names[idx]].extend(entropies)
    # Iterate over each model
    for i, model in enumerate(model_list):
        _, ax = plt.subplots(figsize=(6, 4))
        model.eval()  # Set model to evaluation mode

        # Process each dataset
        for idx, loader in enumerate(loaders):
            entropies = compute_batch_entropy(model, loader)

            # Calculate weights to convert histogram counts into percentages
            total = len(entropies)
            weights = np.ones_like(entropies) / total

            # Plot histogram
            ax.hist(
                entropies,
                bins=30,
                weights=weights,
                edgecolor="black",
                alpha=1 - 0.3 * idx,
                color=f"C{idx*3+1}",
                label=dataset_names[idx],
            )

        # Set titles, labels, and formatting
        ax.set_title(f"{model_names[i]}")
        ax.set_xlabel("Entropy (confidence of prediction: confident -> uncertain)")
        ax.set_ylabel("Frequency")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.legend() if i == 0 else None

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()


def visualize_weights(model_list, modelnames, method="pca", num_points=100):
    """
    Example usage:
    visualize_weights(model_list, model_names, method='pca')

    Visualizes weights of specified layers from multiple PyTorch models on a single plot,
    with optional dimensionality reduction (t-SNE or PCA), using contour lines and scatter plots.

    Parameters:
    - model_list: List of PyTorch model instances.
    - modelnames: list of model names.
    - method: Dimensionality reduction method ('tsne' or 'pca').
    - layer_indices: Tuple indicating the indices of two layers whose weights are to be visualized.
    - num_points: Resolution of the contour plot grid.
    """

    # Loop through each model
    for layer_idx in range(len(list(model_list[0].children()))):
        # Create a single figure
        _, ax = plt.subplots(figsize=(5, 4))
        ax.set_facecolor("white")  # Set background to white

        for idx, model in enumerate(model_list):
            weights = list(model.children())[layer_idx].weight.data.cpu().numpy()

            # Apply dimensionality reduction if necessary
            if method == "tsne":
                reducer = TSNE(n_components=2, random_state=42)
            elif method == "pca":
                reducer = PCA(n_components=2)
            else:
                raise ValueError(
                    "Unsupported dimensionality reduction method: must be 'tsne' or 'pca'."
                )

            reduced_weights = reducer.fit_transform(weights)

            # Use the first two dimensions of component for plotting
            w1, w2 = reduced_weights[:, 0], reduced_weights[:, 1]

            # Scatter plot for individual weight points
            ax.scatter(
                w1,
                w2,
                label=f"{modelnames[idx]}",
                alpha=1 - 0.3 * idx,
                color=f"C{idx*3+1}",
            )

            # Generate a grid of points and adjust bin edges for contours
            x = np.linspace(min(w1), max(w1), num_points)
            y = np.linspace(min(w2), max(w2), num_points)
            X, Y = np.meshgrid(x, y)
            x_edges = np.linspace(min(w1), max(w1), num_points + 1)
            y_edges = np.linspace(min(w2), max(w2), num_points + 1)

            # Contour plot
            # cp = ax.contour(X, Y, Z, levels=15, colors='black')
            # ax.clabel(cp, inline=1, fontsize=10)

        # Add legend and labels
        if layer_idx == len(list(model_list[0].children())) - 1:
            ax.legend()

        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title(f"layer {layer_idx+1} Weight Distribution")

        plt.tight_layout()
        plt.show()


def compute_losses(model, loader):
    """Auxiliary function to compute per-sample losses"""

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    model.to(device)
    model.eval()

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        logits = compute_logits(model, inputs)

        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)


def compare_mia_scores(
    model_list, model_names, forget_loader, retain_loader, unseen_loader
):
    sample_num = min(len(retain_loader), len(unseen_loader))
    sampled_retain_loader = sample_loader(retain_loader, sample_num)
    sampled_unseen_loader = sample_loader(unseen_loader, sample_num)

    for idx, model in enumerate(model_list):
        retain_losses = compute_losses(model, sampled_retain_loader)
        unseen_losses = compute_losses(model, sampled_unseen_loader)
        forget_losses = compute_losses(model, forget_loader)

        if len(unseen_losses) > len(retain_losses):
            unseen_losses = unseen_losses[: len(retain_losses)]
        else:
            retain_losses = retain_losses[: len(unseen_losses)]

        # make sure we have a balanced dataset for the MIA
        assert len(unseen_losses) == len(
            retain_losses
        ), f"Dataset is not balanced {len(unseen_losses)}, {len(retain_losses)}"

        attack_data_X = np.concatenate((unseen_losses, retain_losses)).reshape((-1, 1))
        forget_X = np.array(forget_losses).reshape(-1, 1)
        nonmember_X = np.array(unseen_losses).reshape(-1, 1)
        members_X = np.array(retain_losses).reshape(-1, 1)

        # 1 means the sample is used for training
        attack_data_Y = np.array([0] * len(unseen_losses) + [1] * len(retain_losses))
        forget_Y = np.array([0] * len(forget_losses))
        nonmember_Y = np.array([0] * len(unseen_losses))
        members_Y = np.array([1] * len(retain_losses))

        attack_model, _ = mia_crossval(attack_data_X, attack_data_Y, forget_X, forget_Y)

        mia_forget_score = mia_true_nonmember_rate(
            attack_data_X, attack_data_Y, forget_X, forget_Y, attack_model
        )

        mia_unseen_score = mia_true_nonmember_rate(
            attack_data_X, attack_data_Y, nonmember_X, nonmember_Y, attack_model
        )

        mia_seen_score = mia_true_nonmember_rate(
            attack_data_X, attack_data_Y, members_X, members_Y, attack_model
        )

        print(
            f"{model_names[idx]}: \nThe MIA socre is {mia_forget_score:.3f} on forgotten data"
        )
        print(f"The MIA socre is {mia_unseen_score:.3f} on unseen data")
        print(f"The MIA socre is {mia_seen_score:.3f} on seen data\n\n")
