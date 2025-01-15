# %% [markdown]
# ## CNN with competing hidden filters: unsupervised learning

# %% [markdown]
# In this notebook, we build a CNN where learning happens exclusively in an unsupervised manner and filters are pushed to compete against one another(similar to the competition among hidden units in the paper "Unsupervised learning by competing hidden units" by Krotov and Hopfield). The rationale behind the model is that each filter should specialize in detecting unique features, enhancing diversity and representation; to do so, we orthogonalize the filters such that there is minimal overlapping over each filter and, thus, there's maximum specialization of each filter for a feature.

# %% [markdown]
# The CNN is built with PyTorch, using the Adam optimizer and a learning rate of 0.01. Since the classification problem at hand is relatively simple, we have opted for a simpler model with a low number of layers to avoid overfitting. Futhermore, empirical testing has shown diminishing returns in more complex models so, having limited resources, we decided to stick to a simpler CNN.

# %% [markdown]
# <font color='red'> TO DO:
# 1. TEST SGD
# 2. TEST A SMALLER LEARNING RATE OF 1e-4 FOR ADAM
# 3. TRY ADDING ONE LAYER FOR CIFAR (since the problem is slightly more complex)
# 4. TEST MORE COMPLEX ORTHOGONALITY
# 5. TRY ADDING BIAS
# 6. TRY K-TOP FILTERS AND NOT ONLY ONE

# %% [markdown]
# <font color='green'> COMPLETED:
# 1. Adam with learning rate 0.01 --> 2 unsupervised epochs are enough on MNIST (92.54%), 5 unsupervised epochs are not enough on CIFAR (50.64%)
# 2. Gramh-Schmidt diversification

# %%
import time

import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from optuna.trial import Trial
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm


# %%
class Filters(nn.Module):
    """
    Below, we have a custom layer that doesn't rely on backpropagation for training. Instead, it uses:
        1) a forward pass to takes an input to get the filters' responses.
        2) A custom update rule (e.g., winner-take-most "WTA") to update filters.
        3) An orthonormalization step to ensure diversity among filters.
    """

    def __init__(
        self,
        inputChannels,
        outputChannels,
        kernel=3,
        stride=1,
        padding=1,
        delta=0.1,
        k=5,
    ):
        super(Filters, self).__init__()
        self.inputChannels = inputChannels  # e.g., 3 for CIFAR-10 since the images are RGB and 1 for MNIST
        self.outputChannels = outputChannels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.delta = delta
        self.k = k

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # we initialize weights manually to have more control
        # we sample from a normal distribution and multiply by 0.01 to have small weights at the start (to prevent large outputs at the beginning)
        # requires_grad = False explicitly indicates that we have our own custom update rule, defined later on
        weightShape = (outputChannels, inputChannels, kernel, kernel)
        self.weights = nn.Parameter(
            torch.randn(weightShape) * 0.01, requires_grad=False
        )
        self.bias = None

    def forward(self, x):
        # we use a standard forward pass
        return nn.functional.conv2d(
            x, self.weights, bias=self.bias, stride=self.stride, padding=self.padding
        )

    @torch.no_grad()
    def update(self, x, learningRate=0.01, competitionMode="wta"):
        """
        Modified unsupervised update rule:
        1) Forward pass -> compute 'response'
        2) For each patch:
             - The top filter is updated positively (winner-take-all).
             - The next 'k' filters are pushed away (negative update).
        3) We accumulate updates in a tensor (using 'index_add_').
        4) Finally, we add those updates into 'self.weights' with the appropriate sign.
        """
        # 1) Forward pass & find top (k+1) filters
        #    response: [B, c_out, H_out, W_out]
        response = self.forward(x)

        # topk_vals: [B, k+1, H_out, W_out]
        # topk_idxs: [B, k+1, H_out, W_out] -- these are the filter indices
        topk_vals, topk_idxs = response.topk(self.k + 1, dim=1)

        # The "winner" is the 0th in that list
        # The next 'k' are the runner-ups we want to push away
        winner_idxs = topk_idxs[:, 0, :, :]  # shape [B, H_out, W_out]
        losers_idxs = topk_idxs[:, 1:, :, :]  # shape [B, k, H_out, W_out]

        # 2) Unfold input into patches
        #    x_unfolded: [B, c_in*kernel*kernel, num_patches]
        x_unfolded = F.unfold(
            x, kernel_size=self.kernel, padding=self.padding, stride=self.stride
        )
        B, _, num_patches = x_unfolded.shape

        # Flatten out the (H_out * W_out) dimension for indexing
        # winner_flat[b] -> shape [num_patches] for batch b
        winner_flat = winner_idxs.reshape(B, -1)
        # losers_flat[b] -> shape [k, num_patches] for batch b
        losers_flat = losers_idxs.reshape(B, self.k, -1)

        # 3) Create accumulators for winners and losers
        accumulation_winner = torch.zeros(
            (self.outputChannels, self.inputChannels * self.kernel * self.kernel),
            device=self.device,
        )
        accumulation_losers = torch.zeros_like(accumulation_winner)

        # 4) Accumulate updates (index_add_) for each batch
        #
        # Winners: + learningRate * x_unfolded[b].T
        # Losers:  - delta * x_unfolded[b].T
        #
        # We do a for-loop over B for clarity, and a nested loop over k
        # for the losers. The patch dimension is handled by index_add_ internally.
        for b in range(B):
            # 4a) winner update
            accumulation_winner.index_add_(
                dim=0,
                index=winner_flat[b],  # which filters to update
                source=x_unfolded[b].T,  # shape [num_patches, c_in*k*k]
            )
            # 4b) losers update
            #     We do a small loop for each of the k losers.
            #     Each row in losers_flat[b] is shape [num_patches].
            for i in range(self.k):
                accumulation_losers.index_add_(
                    dim=0, index=losers_flat[b, i], source=x_unfolded[b].T
                )

        # 5) Perform the actual weight update
        #    weights += learningRate * (accumulation_winner - delta * accumulation_losers)
        self.weights += learningRate * (
            accumulation_winner.view_as(self.weights)
            - self.delta * accumulation_losers.view_as(self.weights)
        )

    def orthonormalization(self, error=1e-8):
        """
        Here, we orthonormalize the filters using Gram-Schmidt. Each filter has shape [inputChannels * kernel^2]
        """
        with torch.no_grad():
            # we reshape the filters in order to be able to apply Gram-Schmidt
            flattenedWeights = self.weights.view(self.outputChannels, -1)
            matrix = []
            for i in range(self.outputChannels):
                # we take the current filter's weights...
                current = flattenedWeights[i].clone()
                # ... and subtract projections of current on already processed filters in the matrix
                for j in matrix:
                    projection = (current @ j) * j
                    current -= projection

                # we normalize the resulting vector to have a unit norm
                norm = torch.norm(current) + error
                current = current / norm
                matrix.append(current)

            # we convert the list of 1d vectors to a 2d tensor and then reshape the tensor to the original 4d shape
            matrix = torch.stack(matrix, dim=0)
            self.weights.data = matrix.view_as(self.weights.data)


# %%
class CompetingCNN(nn.Module):
    def __init__(
        self,
        inputChannels=3,
        num_classes=10,
        feature_dim=64,
        kernel=3,
        stride=1,
        padding=1,
        pool_size=4,
        delta=0.1,
        k=5,
    ):
        super(CompetingCNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.conv1 = Filters(
            inputChannels=inputChannels,
            outputChannels=feature_dim,
            kernel=kernel,
            stride=stride,
            padding=padding,
            delta=delta,
            k=k,
        )
        self.conv2 = Filters(
            inputChannels=feature_dim,
            outputChannels=feature_dim,
            kernel=kernel,
            stride=stride,
            padding=padding,
            delta=delta,
            k=k,
        )
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Linear(feature_dim * pool_size * pool_size, num_classes)

    def forward(self, x):
        # we follow both convolutions with a ReLU to add non-linearity to learn more complex patterns
        x = self.conv1(x)
        x = nn.functional.relu(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        output = self.classifier(x)
        return output

    def update(self, x, learningRate=0.01, competitionMode="wta"):
        """
        Here, we call the unsupervised update rule of each convolution layer. Then, we normalize.
        """
        # update filters in conv1
        self.conv1.update(x, learningRate=learningRate, competitionMode=competitionMode)
        self.conv1.orthonormalization()

        # forward pass from conv1
        with torch.no_grad():
            out1 = self.conv1.forward(x)
            out1 = nn.functional.relu(out1)

        # update filters in conv2 based on output of conv1
        self.conv2.update(
            out1, learningRate=learningRate, competitionMode=competitionMode
        )
        self.conv2.orthonormalization()

    def freezeLayers(self, freeze=True):
        """
        Here, we freeze the weights of the unsupervised convolutional layers.
        """
        for i in self.conv1.parameters():
            i.requires_grad = not freeze

        for i in self.conv2.parameters():
            i.requires_grad = not freeze


# %%
def compute_validation_accuracy(model, val_loader, device):
    """Helper function to compute validation accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def compute_improvement_rate(accuracies):
    """Helper function to compute average improvement rate"""
    if len(accuracies) < 2:
        return accuracies[0]
    improvements = [b - a for a, b in zip(accuracies[:-1], accuracies[1:])]
    return sum(improvements) / len(improvements)


def trainCNN(
    dataset_name="MNIST",
    batch_size=64,
    unsupervised_epochs=5,
    supervised_epochs=5,
    unsupervised_learning_rate=0.01,
    supervised_learning_rate=0.001,
    competition_mode="wta",
    feature_dim=64,
    kernel=3,
    stride=1,
    padding=1,
    pool_size=4,
    delta=0.1,
    k=5,
):
    """
    Here, we train a CNN using the CompetingCNN model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name.lower() == "mnist":
        # we normalize the data to have zero mean and unit variance
        pipeline = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        trainingData = datasets.MNIST(
            root="./data", train=True, transform=pipeline, download=True
        )
        testingData = datasets.MNIST(
            root="./data", train=False, transform=pipeline, download=True
        )
        inputChannels = 1  # MNIST is grayscale, thus the input channel is 1
        num_classes = 10
    elif dataset_name.lower() == "fashionmnist":
        pipeline = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
        )
        trainingData = datasets.FashionMNIST(
            root="./data", train=True, transform=pipeline, download=True
        )
        testingData = datasets.FashionMNIST(
            root="./data", train=False, transform=pipeline, download=True
        )
        inputChannels = 1  # FashionMNIST is grayscale
        num_classes = 10
    elif dataset_name.lower() == "cifar100":
        pipeline = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        trainingData = datasets.CIFAR100(
            root="./data", train=True, transform=pipeline, download=True
        )
        testingData = datasets.CIFAR100(
            root="./data", train=False, transform=pipeline, download=True
        )
        inputChannels = 3
        num_classes = 100
    elif dataset_name.lower() == "cifar10":
        pipeline = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )
        trainingData = datasets.CIFAR10(
            root="./data", train=True, transform=pipeline, download=True
        )
        testingData = datasets.CIFAR10(
            root="./data", train=False, transform=pipeline, download=True
        )
        inputChannels = 3
        num_classes = 10
    else:
        raise ValueError(
            "Unsupported dataset. Use 'MNIST', 'FashionMNIST', 'CIFAR10', or 'CIFAR100'"
        )

    # Split training data into train and validation sets
    validation_split = 0.1
    train_size = int((1 - validation_split) * len(trainingData))
    val_size = len(trainingData) - train_size

    train_dataset, val_dataset = random_split(
        trainingData,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test = DataLoader(testingData, batch_size=batch_size, shuffle=False)

    # we initialize the model and save it
    model = CompetingCNN(
        inputChannels=inputChannels,
        num_classes=num_classes,
        feature_dim=feature_dim,
        kernel=kernel,
        stride=stride,
        padding=padding,
        pool_size=pool_size,
        delta=delta,
        k=k,
    ).to(device)
    model.train()

    # Lists to store validation accuracies
    unsupervised_val_accuracies = []
    supervised_val_accuracies = []

    # unsupervised pretraining using the custom update rule
    print("STARTING UNSUPERVISED PRE-TRAINING")
    for epoch in tqdm(range(unsupervised_epochs)):
        for batchIndex, (images, unusedLabels) in enumerate(train):
            images = images.to(device)
            model.update(
                images,
                learningRate=unsupervised_learning_rate,
                competitionMode=competition_mode,
            )

        # Check validation accuracy every 2 epochs
        if epoch % 2 == 0:
            val_acc = compute_validation_accuracy(model, val, device)
            unsupervised_val_accuracies.append(val_acc)
            print(
                f"Unsupervised Epoch [{epoch + 1}/{unsupervised_epochs}], Validation Accuracy: {val_acc:.2f}%"
            )

    # we freeze weights to avoid learning continues after the unsupervised part
    print("FREEZING WEIGHTS")
    model.freezeLayers(freeze=True)

    # supervised training with frozen weights
    print("STARTING SUPERVISED TRAINING")
    optimizer = optim.Adam(model.classifier.parameters(), lr=supervised_learning_rate)
    crossEntropy = nn.CrossEntropyLoss()

    supervised_loss_list = []
    supervised_acc_list = []

    for epoch in tqdm(range(supervised_epochs)):
        model.train()
        currentLoss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for batchIndex, (images, labels) in enumerate(train):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = crossEntropy(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)
            supervised_loss_list.append(loss.item())
            supervised_acc_list.append(accuracy)

            # 6) Accumulate for epoch-level average
            currentLoss += loss.item() * labels.size(0)
            epoch_correct += correct
            epoch_samples += labels.size(0)

        epoch_avg_loss = currentLoss / epoch_samples
        epoch_avg_acc = epoch_correct / epoch_samples

        print(
            f"Supervised Epoch [{epoch + 1}/{supervised_epochs}], "
            f"Loss: {epoch_avg_loss:.4f}, "
            f"Accuracy: {epoch_avg_acc:.4f}"
        )

        # Check validation accuracy every 2 epochs
        if epoch % 2 == 0:
            val_acc = compute_validation_accuracy(model, val, device)
            supervised_val_accuracies.append(val_acc)
            print(
                f"Supervised Epoch [{epoch + 1}/{supervised_epochs}], Validation Accuracy: {val_acc:.2f}%"
            )

    # Compute improvement rates
    unsupervised_improvement = compute_improvement_rate(unsupervised_val_accuracies)
    supervised_improvement = compute_improvement_rate(supervised_val_accuracies)
    final_val_accuracy = (
        supervised_val_accuracies[-1] if supervised_val_accuracies else 0.0
    )

    # evaluation on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

    return (
        model,
        supervised_acc_list,
        supervised_loss_list,
        final_val_accuracy,
        unsupervised_improvement,
        supervised_improvement,
    )


def objective(trial: Trial, datasets: list) -> tuple[float, float, float, float]:
    """
    Multi-objective optimization function that evaluates across all datasets.

    Parameters
    ----------
    trial : Trial
        Optuna trial object.
    datasets : list
        List of dataset names to evaluate.

    Returns
    -------
    tuple
        (Average Validation Accuracy, Average Unsupervised Improvement,
         Average Supervised Improvement, Time Taken)
    """
    start_time = time.time()

    params = {
        "batch_size": trial.suggest_int("batch_size", 32, 256),
        "unsupervised_epochs": trial.suggest_int("unsupervised_epochs", 1, 20),
        "supervised_epochs": trial.suggest_int("supervised_epochs", 1, 20),
        "unsupervised_learning_rate": trial.suggest_float(
            "unsupervised_learning_rate", 0.001, 0.8
        ),
        "supervised_learning_rate": trial.suggest_float(
            "supervised_learning_rate", 1e-4, 1e-1, log=True
        ),
        "feature_dim": trial.suggest_int("feature_dim", 32, 128),
        "kernel": trial.suggest_int("kernel", 3, 7, step=2),
        "stride": trial.suggest_int("stride", 1, 2),
        "padding": trial.suggest_int("padding", 0, 2),
        "pool_size": trial.suggest_int("pool_size", 2, 8),
        "delta": trial.suggest_float("delta", 0.05, 0.5),
        "k": trial.suggest_int("k", 1, 10),
    }

    total_val_accuracy = 0.0
    total_unsup_improvement = 0.0
    total_sup_improvement = 0.0
    for dataset_name in datasets:
        try:
            _, _, _, val_accuracy, unsup_improvement, sup_improvement = trainCNN(
                dataset_name=dataset_name, **params
            )
            total_val_accuracy += val_accuracy
            total_unsup_improvement += unsup_improvement
            total_sup_improvement += sup_improvement
        except Exception as e:
            print(f"Error in dataset {dataset_name}: {e}")
            return float("-inf"), float("-inf"), float("-inf"), float("inf")

    average_val_accuracy = total_val_accuracy / len(datasets)
    average_unsup_improvement = total_unsup_improvement / len(datasets)
    average_sup_improvement = total_sup_improvement / len(datasets)
    time_taken = time.time() - start_time

    return (
        average_val_accuracy,
        average_unsup_improvement,
        average_sup_improvement,
        time_taken,
    )


def optimize_hyperparameters(datasets: list, n_trials: int = 100) -> dict:
    """
    Hyperparameter optimization across multiple datasets using Optuna.

    Parameters
    ----------
    datasets : list
        List of dataset names to optimize on.
    n_trials : int, optional
        Number of trials for optimization, by default 100.

    Returns
    -------
    dict
        Best hyperparameters found.
    """
    """study = optuna.create_study(
        directions=["maximize", "maximize", "maximize", "minimize"],
        study_name="CNN_multi_dataset_optimization",
        storage="sqlite:///CNN_multi_dataset_optimization.db",
    )"""

    study = optuna.load_study(
        study_name="CNN_multi_dataset_optimization",
        storage="sqlite:///../output/bio_classifier_hyperparam_optimization.db",
    )
    study.optimize(lambda trial: objective(trial, datasets), n_trials=n_trials)

    print("\nPareto Front:")
    for trial in study.best_trials:
        print("\nTrial:")
        print(f"  Average Validation Accuracy: {trial.values[0]:.2f}%")
        print(f"  Average Unsupervised Improvement: {trial.values[1]:.2f}%")
        print(f"  Average Supervised Improvement: {trial.values[2]:.2f}%")
        print(f"  Time Taken: {trial.values[3]:.2f} seconds")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    # Return the parameters of the trial with highest validation accuracy
    best_accuracy_trial = max(study.best_trials, key=lambda t: t.values[0])
    return best_accuracy_trial.params


# %%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_names = ["FashionMNIST", "CIFAR100", "MNIST"]
    print(
        "\nOptimizing hyperparameters across all datasets with multi-objective optimization"
    )
    best_params = optimize_hyperparameters(dataset_names, n_trials=100)

    for dataset in dataset_names:
        print(f"\nTraining final model on {dataset} with best parameters")
        (
            model,
            loss_history,
            acc_history,
            val_accuracy,
            unsup_improvement,
            sup_improvement,
        ) = trainCNN(dataset_name=dataset, **best_params)

        print(f"Final Validation Accuracy for {dataset}: {val_accuracy:.2f}%")
        print(f"Unsupervised Improvement Rate for {dataset}: {unsup_improvement:.2f}%")
        print(f"Supervised Improvement Rate for {dataset}: {sup_improvement:.2f}%")

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(loss_history)
        plt.title(f"{dataset} Loss History")
        plt.subplot(1, 2, 2)
        plt.plot(acc_history)
        plt.title(f"{dataset} Accuracy History")
        plt.show()
