import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from copy import copy, deepcopy
from typing import Literal, Optional, Self

import dill
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


def _lateral_inhibition_single_input(
    ic: np.ndarray, w_inh: float, tau_L: float, inhibition_timesteps: int
) -> np.ndarray:
    """
    Solve the lateral inhibition dynamics for a single input current array.
    """

    def dynamics(t, h):
        relu_h = np.maximum(h, 0)
        inhibition = w_inh * np.sum(relu_h) - w_inh * relu_h
        return (ic - inhibition - h) / tau_L

    solution = solve_ivp(
        dynamics,
        t_span=(0, inhibition_timesteps),
        y0=ic,
        method="RK23",
    )
    return solution.y[:, -1]


def _map_lateral_inhibition(args):
    ic, w_inh, tau_L, inhibition_timesteps = args
    return _lateral_inhibition_single_input(ic, w_inh, tau_L, inhibition_timesteps)


class PoweredAbsoluteLoss(nn.Module):
    """
    A custom loss function that computes the mean of the absolute difference
    raised to the power ``m``.
    """

    def __init__(self, m: int):
        """
        Initialize PoweredAbsoluteLoss.

        Parameters
        ----------
        m : int
            The exponent for the absolute difference.
        """
        super().__init__()
        self.m = m

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the powered absolute loss.

        Parameters
        ----------
        output : torch.Tensor
            Model predictions (batch_size, num_classes)
        target : torch.Tensor
            Ground truth values (batch_size, num_classes)

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        # Compute element-wise absolute difference
        diff = torch.abs(output - target)

        # Raise to the power m and compute the mean
        return torch.mean(diff**self.m)


class Classifier(nn.Module):
    """
    Base classifier class providing common dataset handling and supervised learning methods.
    """

    supervised_train_data_loader: DataLoader
    supervised_validation_data_loader: DataLoader
    criterion: PoweredAbsoluteLoss | nn.CrossEntropyLoss

    input_size: int
    hidden_size: int
    output_size: int
    supervised_train_errors: dict
    supervised_validation_errors: dict
    supervised_test_errors: dict

    def __init__(
        self,
        dataset_name: Literal[
            "MNIST",
            "CIFAR-10",
            "CIFAR-10-AutoAugmented",
            "CIFAR-100",
            "FashionMNIST",
            "CIFAR-100-AutoAugmented",
        ],
        supervised_minibatch_size: int,
        hidden_size: int,
        supervised_lr: float,
        use_original_supervised: Literal["True", "False"],
        power_m_loss: int | None = None,
        beta_tanh: float | None = None,
        use_validation: bool = True,
    ):
        """
        Initialize the Classifier.

        Parameters
        ----------
        dataset_name : {"MNIST", "CIFAR-10", "CIFAR-10-AutoAugmented"}
            Name of the dataset to use.
        supervised_minibatch_size : int
            Size of the supervised training minibatch.
        hidden_size : int
            Number of hidden units.
        supervised_lr : float
            Learning rate for supervised training.
        use_original_supervised : {"True", "False"}
            If True uses one-hot encoding and powered absolute loss and tanh activation in the final layer as in the original paper.
            Otherwise if False, uses softmax and cross-entropy loss.
        power_m_loss : int or None, optional
            The exponent for the powered absolute loss if applicable.
        beta_tanh : float or None, optional
            Scale factor for the tanh activation if one-hot is used.
        use_validation : bool,optional
            Whether to use a validation set. If False, only train and test sets are used.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.use_validation = use_validation

        if use_original_supervised == "True":
            self.use_original_supervised = True

        elif use_original_supervised == "False":
            self.use_original_supervised = False

        (
            self.supervised_train_data_loader,
            self.supervised_validation_data_loader,
            self.supervised_test_data_loader,
        ) = self._load_dataloader(
            dataset_name,
            supervised_minibatch_size,
            self.use_original_supervised,
            use_validation,
        )

        # Initialize hidden layer weights W (hidden_size x input_size)
        input_size = self.supervised_train_data_loader.dataset[0][0].shape[0]
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = len(self.supervised_test_data_loader.dataset.classes)

        self.supervised_train_errors = {}
        self.supervised_validation_errors = {}
        self.supervised_test_errors = {}

        # Initialize layers
        self.relu = nn.ReLU()

        # Initialize training

        if self.use_original_supervised:
            self.beta_tanh = beta_tanh
            self.tanh = nn.Tanh()
            self.criterion = PoweredAbsoluteLoss(power_m_loss)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.classifier_name = self.__class__.__name__
        self.supervised_lr = supervised_lr
        self.to(device)

        # Add validation for use_original_supervised and beta_tanh
        if self.use_original_supervised and self.beta_tanh is None:
            raise ValueError(
                "beta_tanh must be provided when use_original_supervised is True."
            )

        # Initialize counters
        self.current_supervised_epoch = 0

    @classmethod
    def load(cls, path: str) -> Self:
        """
        Load a saved model from a given path using dill.

        Parameters
        ----------
        path : str
            The filepath to the saved model. Should end in ".pth".

        Returns
        -------
        Classifier
            The loaded model instance.
        """
        with open(path, "rb") as f:
            return torch.load(f, pickle_module=dill)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to be implemented in subclasses.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            The network output.
        """
        raise NotImplementedError

    def train_unsupervised_and_supervised(self):
        """
        High-level training and plotting errors method to be implemented in subclasses.
        """
        raise NotImplementedError

    def _load_dataloader(
        self,
        name: Literal[
            "MNIST",
            "CIFAR-10",
            "CIFAR-10-AutoAugmented",
            "FashionMNIST",
            "CIFAR-100",
            "CIFAR-100-AutoAugmented",
        ],
        minibatch_size: int,
        use_original_supervised: bool,
        use_validation: bool,
    ) -> tuple[DataLoader, Optional[DataLoader], DataLoader]:
        """
        Load and preprocess the specified dataset.

        Parameters
        ----------
        name : {"MNIST", "CIFAR-10", "CIFAR-10-AutoAugmented", "FashionMNIST"}
            The name of the dataset to load.
        minibatch_size : int
            Batch size for the data loader.
        use_original_supervised : bool
            If True uses one-hot encoding.
        use_validation : bool
            Whether to create a validation data loader.

        Returns
        -------
        tuple of DataLoader
            Train, validation (or None), and test data loaders.
        """

        def one_hot_collate_fn(
            batch: list[tuple[torch.Tensor, int]],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Custom collate function to convert integer labels to one-hot encoded vectors.

            Parameters
            ----------
            batch : List[Tuple[torch.Tensor, int]]
                List of tuples containing data and integer labels.

            Returns
            -------
            Tuple[torch.Tensor, torch.Tensor]
                A tuple containing the data tensor and the one-hot encoded labels tensor.
            """
            if name in ["MNIST", "FashionMNIST", "CIFAR-10", "CIFAR-10-AutoAugmented"]:
                num_classes = 10
            elif name in ["CIFAR-100", "CIFAR-100-AutoAugmented"]:
                num_classes = 100
            else:
                raise ValueError("Unsupported dataset")

            data, labels = zip(*batch)  # Unzip the batch
            data = torch.stack(
                data, dim=0
            )  # Stack data into a tensor of shape (batch_size, ...)

            # Convert labels to one-hot encoding
            labels = torch.tensor(labels, dtype=torch.long)
            one_hot = torch.zeros((labels.size(0), num_classes), dtype=torch.float)
            one_hot.scatter_(
                1, labels.unsqueeze(1), 1.0
            )  # Scatter 1s at the appropriate indices

            # Scale one-hot vectors from [0, 1] to [-1, +1]
            one_hot = one_hot * 2.0 - 1.0  # Scale to [-1, +1]

            return data, one_hot

        if name == "MNIST":
            transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Lambda(lambda x: x.view(-1)),
                ]
            )

            train_ds = datasets.MNIST(
                root="../data",
                download=True,
                train=True,
                transform=transform,
            )

            if use_validation:
                train_ds, validation_ds = random_split(
                    train_ds, [50000, 10000], torch.Generator(device=device)
                )
            test_ds = datasets.MNIST(
                root="../data",
                download=True,
                train=False,
                transform=transform,
            )
        elif name in ["CIFAR-10", "CIFAR-10-AutoAugmented"]:
            transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Lambda(lambda x: x.flatten()),
                    v2.Lambda(lambda x: x / x.norm(p=2)),
                ]
            )
            if "AutoAugmented" in name:
                transform = v2.Compose(
                    [v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10), transform]
                )

            train_ds = datasets.CIFAR10(
                root="../data",
                download=True,
                train=True,
                transform=transform,
            )

            test_ds = datasets.CIFAR10(
                root="../data",
                download=True,
                train=False,
                transform=transform,
            )
        elif name == "FashionMNIST":
            transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Lambda(lambda x: x.view(-1)),
                ]
            )

            train_ds = datasets.FashionMNIST(
                root="../data",
                download=True,
                train=True,
                transform=transform,
            )

            test_ds = datasets.FashionMNIST(
                root="../data",
                download=True,
                train=False,
                transform=transform,
            )

        elif name in ["CIFAR-100", "CIFAR-100-AutoAugmented"]:
            transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Lambda(lambda x: x.flatten()),
                    v2.Lambda(lambda x: x / x.norm(p=2)),
                ]
            )
            if "AutoAugmented" in name:
                transform = v2.Compose(
                    [v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10), transform]
                )

            train_ds = datasets.CIFAR100(
                root="../data",
                download=True,
                train=True,
                transform=transform,
            )

            test_ds = datasets.CIFAR100(
                root="../data",
                download=True,
                train=False,
                transform=transform,
            )
        else:
            raise ValueError("Unsupported dataset")

        if use_validation:
            train_ds, validation_ds = random_split(
                train_ds, [0.9, 0.1], torch.Generator(device=device)
            )

        if use_original_supervised:
            collate_fn = one_hot_collate_fn
        else:
            collate_fn = None

        train_dataloader = DataLoader(
            train_ds,
            batch_size=minibatch_size,
            shuffle=True,
            generator=torch.Generator(device=device),
            collate_fn=collate_fn,
        )
        validation_dataloader = (
            DataLoader(
                validation_ds,
                batch_size=minibatch_size,
                shuffle=True,
                generator=torch.Generator(device=device),
                collate_fn=collate_fn,
            )
            if use_validation
            else None
        )

        test_dataloader = DataLoader(
            dataset=test_ds,
            batch_size=minibatch_size,
            shuffle=True,
            generator=torch.Generator(device=device),
            collate_fn=collate_fn,
        )

        return train_dataloader, validation_dataloader, test_dataloader

    def plot_errors(self, fig: Optional[go.Figure] = None) -> go.Figure:
        if not self.supervised_validation_errors:
            raise ValueError("No errors to plot. Run training first.")
        if fig is None:
            fig = go.Figure()
        epochs = sorted(self.supervised_validation_errors.keys())
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[self.supervised_validation_errors[e] for e in epochs],
                mode="lines+markers",
                name=self.classifier_name,
            )
        )
        fig.update_layout(
            title="Validation Error Rate",
            xaxis_title="Epochs",
            yaxis_title="Error Rate (%)",
        )
        fig.show()
        return fig

    def _run_supervised_epoch(
        self, mode: Literal["training", "validation", "test"]
    ) -> float:
        """
        Run one epoch of supervised training, validation, or testing.

        Parameters
        ----------
        mode : {'training', 'validation', 'test'}
            The mode to run the epoch in.

        Returns
        -------
        float
            Error rate in percentage for this epoch.
        """
        if mode == "training":
            self.train()
            data_loader = self.supervised_train_data_loader
        elif mode == "validation":
            self.eval()
            data_loader = self.supervised_validation_data_loader
        elif mode == "test":
            self.eval()
            data_loader = self.supervised_test_data_loader

        errors = 0
        total = 0
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            if mode == "training" and self.optimizer is not None:
                self.optimizer.zero_grad()
            outputs: torch.Tensor = self(data)
            loss: torch.Tensor = self.criterion(outputs, labels)
            if mode == "training":
                loss.backward()
                self.optimizer.step()
            if self.use_original_supervised:
                prediction = outputs
            else:
                prediction = outputs.argmax(dim=1)
            wrong_answers = prediction != labels
            errors += (wrong_answers).sum().item()
            total += wrong_answers.numel()
        error_rate = 100 * errors / total
        print(
            f"Supervised Epoch {self.current_supervised_epoch} {mode.capitalize()} Avg Errors: {error_rate}%"
        )
        # Replace error storage
        if mode == "training":
            self.supervised_train_errors[self.current_supervised_epoch] = error_rate
        elif mode == "validation":
            self.supervised_validation_errors[self.current_supervised_epoch] = (
                error_rate
            )
        elif mode == "test":
            self.supervised_test_errors[self.current_supervised_epoch] = error_rate
        return error_rate

    def _train_supervised(
        self,
        epochs: int,
        learning_rate: Optional[float] = None,
    ):
        """
        Perform the supervised training phase over a specified number of epochs.

        Parameters
        ----------
        epochs : int
            The number of epochs to run.
        learning_rate : float, optional
            The learning rate for supervised training. If None, uses the supervised_lr defined during initialization.
        """
        learning_rate = self.supervised_lr if learning_rate is None else learning_rate
        print(
            f"Starting Supervised Learning Phase for {self.classifier_name} with LR={learning_rate}"
        )
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        # Initialize supervised epoch counter
        self.current_supervised_epoch = 0
        for _ in tqdm(range(1, epochs + 1), desc="Supervised Learning Epochs"):
            self.current_supervised_epoch += 1
            self._run_supervised_epoch("training")
            if self.use_validation:
                self._run_supervised_epoch("validation")
            else:
                self._run_supervised_epoch("test")

            # Save the model every 100 epochs
            """if (epoch + 1) % 100 == 0:
                self._save(epoch=epoch, phase="supervised")"""

        print("Supervised Learning Phase Complete")

    def save(self):
        """
        Save the current model to disk.
        The saved file is in "../output/models/{classifier_name}/{dataset_name}/supervised_{supervised_epochs}.pth"
        or
        "../output/models/{classifier_name}/{dataset_name}/unsupervised_{unsupervised_epochs}_supervised_{supervised_epochs}.pth"

        NOTE: The optimizer is removed before saving due to known pickling issues.
        """
        save_dir = os.path.join(
            "../",
            "output",
            "models",
            self.classifier_name.lower(),
            self.dataset_name.lower(),
        )
        os.makedirs(save_dir, exist_ok=True)

        # Remove optimizer before saving due to known pickling issues
        optimizer_copy = copy(self.optimizer)
        self.optimizer = None

        if not hasattr(self, "current_unsupervised_epoch"):
            file_name = f"supervised_{self.current_supervised_epoch}.pth"
        else:
            file_name = f"unsupervised_{self.current_unsupervised_epoch}_supervised_{self.current_supervised_epoch}.pth"
        with open(os.path.join(save_dir, file_name), "wb") as f:
            torch.save(self, f, pickle_module=dill)
        print(f"Model saved to {save_dir}")
        self.optimizer = optimizer_copy
        self.test()

    def test(self) -> float:
        """
        Evaluate the model on the test data loader and return the accuracy.

        Returns
        -------
        float
            Test accuracy in percentage.
        """
        self.eval()
        errors = 0
        total = 0
        with torch.no_grad():
            for data, labels in self.supervised_test_data_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = self(data)
                if self.use_original_supervised:
                    prediction = outputs
                else:
                    prediction = outputs.argmax(dim=1)
                wrong_answers = prediction != labels
                errors += (wrong_answers).sum().item()
                total += wrong_answers.numel()
        error_rate = 100 * errors / total
        return error_rate


# Define a traditional single-layer neural network
class BPClassifier(Classifier):
    """
    A backpropagation-based classifier with one hidden layer and an optional tanh output.
    """

    def __init__(
        self,
        dataset_name: Literal[
            "MNIST",
            "CIFAR-10",
            "CIFAR-10-AutoAugmented",
            "CIFAR-100",
            "FashionMNIST",
            "CIFAR-100-AutoAugmented",
        ],
        supervised_minibatch_size: int,
        hidden_size: int,
        use_original_supervised: Literal["True", "False"],
        supervised_lr: float,
        power_m_loss: int | None = None,
        beta_tanh: float | None = None,
        use_validation: bool = True,
    ):
        """
        Initialize a BPClassifier.

        Parameters
        ----------
        dataset_name : {"MNIST", "CIFAR-10", "CIFAR-10-AutoAugmented"}
            Name of the dataset.
        supervised_minibatch_size : int
            Batch size for supervised learning.
        hidden_size : int
            Number of hidden units.
        use_original_supervised : {"True", "False"}
            If True uses one-hot encoding and powered absolute loss and tanh activation in the final layer as in the original paper.
            Otherwise if False, uses softmax and cross-entropy loss.
        power_m_loss : int or None, optional
            The exponent for the powered absolute loss. Only used if use_original_supervised is True.
        beta_tanh : float or None, optional
            Scaling factor for tanh. Only used if use_original_supervised is True.
        use_validation : bool, optional
            Whether to use a validation set. If False, only train and test sets are used.
        """
        super().__init__(
            dataset_name=dataset_name,
            supervised_minibatch_size=supervised_minibatch_size,
            hidden_size=hidden_size,
            supervised_lr=supervised_lr,
            power_m_loss=power_m_loss,
            beta_tanh=beta_tanh,
            use_original_supervised=use_original_supervised,
            use_validation=use_validation,
        )

        # Define the neural network architecture
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through a two-layer feedforward network with optional tanh.

        Parameters
        ----------
        x : torch.Tensor
            Input batch (flattened images).

        Returns
        -------
        torch.Tensor
            Network output.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.use_original_supervised:
            x = self.tanh(self.beta_tanh * x)
        return x

    def train_unsupervised_and_supervised(
        self, learning_rate: float, epochs: int, fig: Optional[go.Figure] = None
    ) -> go.Figure:
        """
        Train the network using supervised learning and then plot errors.

        Parameters
        ----------
        learning_rate : float
            Learning rate for the optimizer.
        epochs : int
            Number of epochs to train.
        """
        self._train_supervised(learning_rate=learning_rate, epochs=epochs)
        return self.plot_errors(fig)


# Define the BioClassifier
class BioClassifier(Classifier):
    """
    A biologically-inspired classifier that first performs unsupervised learning
    on a hidden layer through Hebbian and anti-Hebbian learning, and lateral inhibition.
    It then freezes the weights and performs supervised backpropagation only on the top layer.
    """

    def __init__(
        self,
        dataset_name: Literal[
            "MNIST",
            "CIFAR-10",
            "CIFAR-10-AutoAugmented",
            "CIFAR-100",
            "FashionMNIST",
            "CIFAR-100-AutoAugmented",
        ],
        unsupervised_minibatch_size: int,
        supervised_minibatch_size: int,
        hidden_size: int,
        slow: bool,
        p: int,
        delta: float,
        R: float,
        power_n: int,
        project_w: Literal["True", "False"],
        normalize_w_update: Literal["True", "False"],
        use_original_supervised: Literal["True", "False"],
        supervised_lr: float,
        unsupervised_lr: float,
        h_star: float | None = None,
        tau_L: float | None = None,
        w_inh: float | None = None,
        inhibition_timesteps: Optional[int] = None,
        k: Optional[int] = None,
        power_m_loss: int | None = None,
        beta_tanh: float | None = None,
        use_validation: bool = True,
    ):
        """
        Initialize the BioClassifier.

        Parameters
        ----------
        dataset_name : {"MNIST", "CIFAR-10", "CIFAR-10-AutoAugmented"}
            Dataset to load.
        unsupervised_minibatch_size : int
            Batch size for unsupervised learning.
        supervised_minibatch_size : int
            Batch size for supervised learning.
        hidden_size : int
            Number of hidden units.
        slow : bool
            If True, uses the slow exact method for steady-state calculation in lateral inhibition.
            If False, uses the fast top-k approximation.
        p : int
            Exponent for Lebesgue norm.
            A higher value of p will make the weights more sparse.
        delta : float
            Anti-Hebbian factor. Higher values will make the weights more sparse.
        R : float
            Radius of the Lebesgue sphere for weight normalization.
        power_n : int
            Exponent for hidden-layer activation.
        project_w : {"True", "False"}
            Whether to project weights onto the L^p sphere of radius R after each update.
        normalize_w_update : {"True", "False"}
            Whether to normalize weight updates during unsupervised training by their L^p norm.
        use_original_supervised : {"True", "False"}
            If True uses one-hot encoding and powered absolute loss and tanh activation in the final layer as in the original paper.
            Otherwise if False, uses softmax and cross-entropy loss.
        supervised_lr : float
            Learning rate for supervised phase.
        unsupervised_lr : float
            Learning rate for unsupervised phase.
        h_star : float or None, optional
            Steady-state activity threshold for the slow approach.
            Only used if slow is True.
        tau_L : float or None, optional
            Time constant for lateral inhibition.
            Only used if slow is True.
        w_inh : float or None, optional
            Strength of the global lateral inhibition.
            Only used if slow is True.
        inhibition_timesteps : int or None, optional
            Number of timesteps to run the lateral inhibition dynamics.
            Only used if slow is True.
        k : int or None, optional
            Number of top active neurons after the first for fast approach.
            These neurons weight updates are set to -delta to promote sparsity.
            Only used if slow is False.
        power_m_loss : int or None
            Exponent for the powered absolute loss (supervised).
            Only used if use_original_supervised is True.
        beta_tanh : float or None
            Scale factor for tanh in the supervised phase.
            Only used if use_original_supervised is True.
        use_validation : bool, optional
            Whether to use a validation set. If False, only train and test sets are used.
        """
        super().__init__(
            dataset_name=dataset_name,
            supervised_minibatch_size=supervised_minibatch_size,
            hidden_size=hidden_size,
            power_m_loss=power_m_loss,
            beta_tanh=beta_tanh,
            use_original_supervised=use_original_supervised,
            supervised_lr=supervised_lr,
            use_validation=use_validation,
        )

        (
            self.unsupervised_train_data_loader,
            self.unsupervised_validation_data_loader,
            self.unsupervised_test_data_loader,
        ) = self._load_dataloader(
            dataset_name,
            unsupervised_minibatch_size,
            use_original_supervised,
            self.use_validation,
        )

        # Store hyperparameters
        self.slow = slow
        self.k = k
        self.p = p
        self.delta = delta
        self.R = R
        self.h_star = h_star
        self.w_inh = w_inh
        self.inhibition_timesteps = inhibition_timesteps
        self.tau_L = tau_L
        self.power_n = power_n
        self.unsupervised_lr = unsupervised_lr

        # Convert String to Boolean (for Optuna)
        if project_w == "True":
            project_w = True
        elif project_w == "False":
            project_w = False
        self.project_w = project_w

        if normalize_w_update == "True":
            normalize_w_update = True
        elif normalize_w_update == "False":
            normalize_w_update = False
        self.normalize_w_update = normalize_w_update

        # Initialize W with random values from a normal distribution
        self.unsupervised_weights = torch.randn(self.hidden_size, self.input_size).to(
            device=device
        )

        # Project weights onto the L^p sphere of radius R
        if self.project_w:
            self.unsupervised_weights = self._lebesgue_sphere_normalization(
                self.unsupervised_weights
            )

        # Initialize top supervised layer (hidden_size x output_size)
        self.supervised_weights = nn.Linear(self.hidden_size, self.output_size)

        # Define loss and optimizer for supervised phase
        self.optimizer = Adam(
            self.supervised_weights.parameters(), lr=self.supervised_lr
        )

        self.classifier_name = (
            f"{self.__class__.__name__}_{'slow' if self.slow else 'fast'}"
        )

        # Initialize lists to store errors for unsupervised
        self.unsupervised_train_errors = {}
        self.unsupervised_validation_errors = {}
        self.unsupervised_test_errors = {}

        self.current_supervised_epoch = 0
        self.current_unsupervised_epoch = 0

        # Add validation based on 'slow' parameter
        if not self.slow:
            if self.k is None:
                raise ValueError("k must be provided when slow is False.")
        else:
            if (
                self.h_star is None
                or self.tau_L is None
                or self.w_inh is None
                or self.inhibition_timesteps is None
            ):
                raise ValueError(
                    "h_star, tau_L, w_inh and inhibition_timesteps must be provided when slow is True."
                )

        if dataset_name in ["MNIST", "FashionMNIST"]:
            self.image_shape = (1, 28, 28)
        elif dataset_name in [
            "CIFAR-10",
            "CIFAR-10-AutoAugmented",
            "CIFAR-100",
            "CIFAR-100-AutoAugmented",
        ]:
            self.image_shape = (3, 32, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through unsupervised and supervised layers.

        Parameters
        ----------
        x : torch.Tensor
            Input batch.

        Returns
        -------
        torch.Tensor
            Output after hidden and supervised layers.
        """
        # (batch_size, input_dim) @ (hidden_size, input_dim).T -> (batch_size, hidden_size)
        x = (x @ self.unsupervised_weights.T).pow(self.power_n)
        x = self.relu(x)
        x = self.supervised_weights(x)
        if self.use_original_supervised:
            x = self.tanh(self.beta_tanh * x)
        return x

    def train_unsupervised_and_supervised(
        self,
        unsupervised_epochs: int,
        supervised_epochs: int,
        eval_every: int = 50,
        test_supervised_epochs: int = 20,
        unsupervised_lr: Optional[float] = None,
        supervised_lr: Optional[float] = None,
        plot_errors: bool = True,
        fig: Optional[go.Figure] = None,
    ) -> go.Figure | None:
        """
        Run the unsupervised phase with testing and then the supervised phase.

        Parameters
        ----------
        unsupervised_epochs : int
            Number of unsupervised epochs.
        supervised_epochs : int
            Number of supervised epochs.
        eval_every : int, optional
            Frequency of testing during unsupervised training.
        test_supervised_epochs : int, optional
            Number of supervised epochs to run during each test.
        unsupervised_lr : float, optional
            Learning rate for unsupervised phase. If None, uses the unsupervised_lr defined during initialization.
        supervised_lr : float, optional
            Learning rate for supervised phase. If None, uses the supervised_lr defined during initialization.
        plot_errors : bool, optional
            Whether to plot the errors after training.
        fig : go.Figure, optional
            Existing figure to plot on.

        Returns
        -------
        go.Figure
            Updated figure with error plots.
        """
        self._train_unsupervised(
            learning_rate=unsupervised_lr,
            epochs=unsupervised_epochs,
            eval_every=eval_every,
            test_supervised_epochs=test_supervised_epochs,
        )
        self._train_supervised(
            learning_rate=supervised_lr,
            epochs=supervised_epochs,
        )
        if plot_errors:
            return self.plot_errors(fig)

    def _train_unsupervised(
        self,
        epochs: int,
        eval_every: int,
        test_supervised_epochs: int = 0,
        learning_rate: Optional[float] = None,
    ):
        """
        Conduct unsupervised learning updates on the hidden weights.

        Parameters
        ----------
        epochs : int
            Number of epochs to run unsupervised training.
        learning_rate : float, optional
            The learning rate for unsupervised training. If None, uses the unsupervised_lr defined during initialization.
        eval_every : int, optional
            Number of epochs after which to test the unsupervised weights on the supervised task.
        test_supervised_epochs : int, optional
            Number of supervised epochs to run during each test.
        """
        learning_rate = self.unsupervised_lr if learning_rate is None else learning_rate
        print(
            f"Starting Unsupervised Learning Phase for {self.classifier_name} with LR={learning_rate}"
        )
        learning_rate_update = learning_rate / epochs

        # Initialize unsupervised epoch counter
        self.current_unsupervised_epoch = 0

        # Test the unsupervised weights on the supervised task before training
        if test_supervised_epochs > 0:
            self._test_unsupervised_weights(supervised_epochs=test_supervised_epochs)

        for _ in tqdm(range(1, epochs + 1), desc="Unsupervised Learning Epochs"):
            self.current_unsupervised_epoch += 1
            for _, (input, _) in enumerate(self.supervised_train_data_loader):
                input = input.to(device)  # ensure data is on the same device
                input_currents = self._compute_input_currents(input)  # shape (B, H)

                # Solve lateral inhibition dynamics to get steady state activations
                steady_state_h = self._steady_state_activations(input_currents)

                # Update W using plasticity rule
                weight_update = self._plasticity_rule(
                    input, input_currents, steady_state_h
                )

                if self.normalize_w_update:
                    weight_update = weight_update / weight_update.norm(p=self.p)

                self.unsupervised_weights += learning_rate * weight_update

                # Project weights onto the L^p sphere of radius R
                if self.project_w:
                    self.unsupervised_weights = self._lebesgue_sphere_normalization(
                        self.unsupervised_weights
                    )

            # Every eval_every epochs, test the weights on the supervised task
            if (self.current_unsupervised_epoch) % eval_every == 0:
                self._test_unsupervised_weights(
                    supervised_epochs=test_supervised_epochs  # Pass the new parameter
                )

            # Update the unsupervised learning rate
            learning_rate -= learning_rate_update

        print("Unsupervised Learning Phase Complete")

    def _test_unsupervised_weights(self, supervised_epochs: int = 20):
        """
        Tests the unsupervised weights by temporarily training the model in a supervised manner.

        Parameters
        ----------
        supervised_epochs : int, optional
            Number of supervised epochs to run during testing.

        """
        supervised_weights = deepcopy(self.supervised_weights)
        self._train_supervised(epochs=supervised_epochs)
        self.supervised_weights = supervised_weights
        self.unsupervised_train_errors[self.current_unsupervised_epoch] = (
            self.supervised_train_errors[supervised_epochs]
        )
        if self.use_validation:
            self.unsupervised_validation_errors[self.current_unsupervised_epoch] = (
                self.supervised_train_errors[supervised_epochs]
            )
        else:
            self.unsupervised_test_errors[self.current_unsupervised_epoch] = (
                self.supervised_test_errors[supervised_epochs]
            )

        # Clean up supervised errors
        for epoch in range(1, supervised_epochs + 1):
            self.supervised_train_errors.pop(epoch)
            if self.use_validation:
                self.supervised_validation_errors.pop(epoch)
            else:
                self.supervised_test_errors.pop(epoch)

    def _compute_input_currents(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute the input currents for each hidden neuron.

        Parameters
        ----------
        input : torch.Tensor
            Input batch to be projected.

        Returns
        -------
        torch.Tensor
            Calculated input currents.
        """
        # 1) compute elementwise w_abs^(p-2)
        w_abs_pow = self.unsupervised_weights.abs().pow(
            self.p - 2
        )  # shape (hidden_size, input_dim)
        # 2) multiply by W elementwise
        effective_w = (
            w_abs_pow * self.unsupervised_weights
        )  # shape (hidden_size, input_dim)
        # 3) matrix multiply with input^T
        currents = input @ effective_w.T  # shape (batch_size, hidden_size)
        return currents

    def _steady_state_activations(
        self, input_currents: torch.Tensor
    ) -> torch.Tensor | None:
        """
        Compute the steady state of the lateral inhibition circuit if slow mode is enabled.

        Parameters
        ----------
        input_currents : torch.Tensor
            Input currents to the hidden layer.

        Returns
        -------
        torch.Tensor or None
            Steady-state activations or None if using fast mode.
        """
        if not self.slow:
            return None

        input_currents_np = input_currents.cpu().numpy()
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = list(
                executor.map(
                    _map_lateral_inhibition,
                    (
                        (arr, self.w_inh, self.tau_L, self.inhibition_timesteps)
                        for arr in input_currents_np
                    ),
                )
            )
        return torch.tensor(np.array(results), device=input_currents.device)

    def _top_k_approximation(self, input_currents: torch.Tensor) -> torch.Tensor:
        """
        Approximate the hidden-layer activation using a top-k method.

        Parameters
        ----------
        input_currents : torch.Tensor
            Pre-activation currents.

        Returns
        -------
        torch.Tensor
            Activation vector with +1 for the top neuron and -delta for the next k best.
        """
        # find top (k+1) per row => shape of 'inds' is (batch_size, k+1)
        _, inds = torch.topk(input_currents, self.k + 1, dim=1)

        # g is all zeros
        g = torch.zeros_like(input_currents)

        # FIRST: set the top 1 in each row to +1
        # The best for each row is at column 0:
        best_inds = inds[:, 0].unsqueeze(1)  # shape (batch_size, 1)
        # scatter_ with 'dim=1' means: for each row b, set g[b, best_inds[b]] = 1.0
        g.scatter_(
            dim=1,
            index=best_inds,
            src=torch.ones_like(best_inds, dtype=input_currents.dtype),
        )

        # SECOND: set the next 'k' best to -delta
        # i.e. the columns 1..k in inds
        # shape = (batch_size, k)
        next_k_inds = inds[:, 1:]
        # fill those positions with -delta
        g.scatter_(
            dim=1,
            index=next_k_inds,
            src=(-self.delta)
            * torch.ones_like(next_k_inds, dtype=input_currents.dtype),
        )

        return g

    def _plasticity_rule(
        self, input: torch.Tensor, input_currents: torch.Tensor, h: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Update weights using a Hebbian-like plasticity rule.

        Parameters
        ----------
        input : torch.Tensor
            Input batch.
        input_currents : torch.Tensor
            Pre-activation currents.
        h : torch.Tensor or None
            Neuron activations for slow approach or None for fast approach.

        Returns
        -------
        torch.Tensor
            Weight update matrix.
        """
        # Compute g(h)
        if self.slow:
            g = torch.where(
                h >= self.h_star,
                torch.ones_like(h),
                torch.where(h >= 0, -torch.tensor(self.delta), torch.zeros_like(h)),
            )
        else:
            g = self._top_k_approximation(input_currents)

        # Step 3: form the bracket: R^p * (absW * input) - p_dot[:, None] * W
        # bracket => (B,H,I)
        bracket = (
            (self.R**self.p) * input.unsqueeze(1)  # => (B,1,I)
            - input_currents.unsqueeze(2)  # => (B,H,1)
            * self.unsupervised_weights.unsqueeze(0)  # => (1,H,I)
        )
        # => shape (B,H,I)

        # weight_update => (B,H,I) * (B, H, 1) => (B,H,I)
        weight_update = bracket * g.unsqueeze(2)  # multiply along hidden dimension

        # average over batch => (H,I)
        weight_update = weight_update.mean(dim=0)

        return weight_update

    def _lebesgue_sphere_normalization(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Normalize weights onto the L^p sphere of radius R.

        Parameters
        ----------
        weights : torch.Tensor
            Unsupervised weight matrix.

        Returns
        -------
        torch.Tensor
            Normalized weight matrix.
        """
        weights = (self.R * weights) / (
            weights.norm(p=self.p, dim=1, keepdim=True) + 1e-8
        )

        return weights

    def draw_unsupervised_weights(self, num_columns: int, num_rows: int):
        """
        Use plotly.graph_objects (go) to visualize the weights as images.
        """
        weights = self.unsupervised_weights

        # Estimate average L2 norm for CIFAR-10 datasets for inverse transfomration
        if self.dataset_name in [
            "CIFAR-10",
            "CIFAR-10-AutoAugmented",
            "CIFAR-100",
            "CIFAR-100-AutoAugmented",
        ]:
            total_norm = 0
            if self.dataset_name in ["CIFAR-10"]:
                dataset = datasets.CIFAR10(
                    root="../data",
                    transform=v2.Compose(
                        [
                            v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Lambda(lambda x: x.flatten()),
                        ]
                    ),
                )

            elif self.dataset_name in ["CIFAR-10-AutoAugmented"]:
                dataset = datasets.CIFAR10(
                    root="../data",
                    transform=v2.Compose(
                        [
                            v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
                            v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Lambda(lambda x: x.flatten()),
                        ]
                    ),
                )

            elif self.dataset_name in ["CIFAR-100", "CIFAR-100-AutoAugmented"]:
                dataset = datasets.CIFAR100(
                    root="../data",
                    transform=v2.Compose(
                        [
                            v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Lambda(lambda x: x.flatten()),
                        ]
                    ),
                )
            dataloader = DataLoader(dataset)
            for data, _ in dataloader:
                total_norm += data.norm(p=2)
            avg_norm = total_norm / len(dataloader)
            weights = weights * avg_norm

        indices = np.random.choice(
            weights.shape[0], num_columns * num_rows, replace=False
        )
        fig = make_subplots(
            rows=num_rows,
            cols=num_columns,
            subplot_titles=[f"Neuron {neuron_idx}" for neuron_idx in indices],
            x_title="Unsupervised Weights per Neuron",
        )
        idx = 0
        for r in range(1, num_rows + 1):
            for c in range(1, num_columns + 1):
                neuron_idx = indices[idx]
                weight_image = (
                    weights[neuron_idx].view(*self.image_shape).cpu().detach().numpy()
                )
                weight_image = (weight_image - weight_image.min()) / (
                    weight_image.max() - weight_image.min()
                )
                if self.dataset_name in ["MNIST", "FashionMNIST"]:
                    fig.add_trace(
                        go.Image(z=weight_image, colormodel="RdBu"), row=r, col=c
                    )
                elif self.dataset_name in [
                    "CIFAR-10",
                    "CIFAR-10-AutoAugmented",
                    "CIFAR-100",
                    "CIFAR-100-AutoAugmented",
                ]:
                    weight_image = weight_image * 255
                    weight_image = np.transpose(weight_image, (1, 2, 0))
                    fig.add_trace(go.Image(z=weight_image), row=r, col=c)
                idx += 1
        fig.show()

    def plot_errors(self, fig: Optional[go.Figure] = None) -> go.Figure:
        if fig is None:
            fig = go.Figure()

        if self.use_validation:
            x_unsup = sorted(self.unsupervised_validation_errors.keys())
            y_unsup = [self.unsupervised_validation_errors[k] for k in x_unsup]

            x_sup = sorted(self.supervised_validation_errors.keys())
            y_sup = [self.supervised_validation_errors[k] for k in x_sup]
        else:
            x_unsup = sorted(self.unsupervised_test_errors.keys())
            y_unsup = [self.unsupervised_test_errors[k] for k in x_unsup]

            x_sup = sorted(self.supervised_test_errors.keys())
            y_sup = [self.supervised_test_errors[k] for k in x_sup]

        fig.add_trace(
            go.Scatter(
                x=x_unsup,
                y=y_unsup,
                mode="lines+markers",
                name=f"{self.classifier_name} Unsupervised",
            )
        )

        # Plot Supervised Validation Errors
        fig.add_trace(
            go.Scatter(
                x=x_sup,
                y=y_sup,
                mode="lines+markers",
                name=f"{self.classifier_name} Supervised",
            )
        )

        # Add grey line connecting last unsupervised to first supervised
        fig.add_trace(
            go.Scatter(
                x=[x_unsup[-1], x_sup[0]],
                y=[y_unsup[-1], y_sup[0]],
                mode="lines",
                line=dict(color="grey", dash="dash"),
                showlegend=False,
            )
        )
        fig.update_layout(
            title=f"{'Validation' if self.use_validation else 'Test'} Error Rate",
            xaxis_title="Epochs",
            yaxis_title="Error Rate (%)",
        )
        fig.show()
        return fig
