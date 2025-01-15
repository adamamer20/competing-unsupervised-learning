import math
from time import time
from typing import Literal

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from models import BioClassifier
from torchvision import datasets
from torchvision.transforms import v2


def hyperparameter_optimization(
    trial: optuna.Trial,
    dataset_names: list[
        Literal[
            "MNIST", "CIFAR-10", "CIFAR-100", "CIFAR-10-AutoAugmented", "FashionMNIST"
        ]
    ],
    slow: bool,
    eval_every: int = 2,
    test_supervised_epochs: int = 2,
    hidden_size: tuple[int, int] = (10, 2048),
    unsupervised_epochs: tuple[int, int] = (1, 50),
    supervised_epochs: tuple[int, int] = (1, 50),
    unsupervised_minibatch_size: tuple[int, int] = (10, 100),
    supervised_minibatch_size: tuple[int, int] = (10, 100),
    p: tuple[int, int] = (2, 6),
    delta: tuple[float, float] = (0.1, 0.7),
    unsupervised_lr: tuple[float, float] = (0.001, 0.8),
    supervised_lr: tuple[float, float] = (1e-4, 1e-1),
    power_m_loss: tuple[int, int] = (1, 10),
    power_n: tuple[int, int] = (1, 10),
    beta_tanh: tuple[float, float] = (0.1, 2.0),
    h_star: tuple[float, float] = (0.1, 0.9),
    tau_L: tuple[float, float] = (0.1, 1.0),
    w_inh: tuple[float, float] = (0.1, 1.0),
    inhibition_timesteps: tuple[int, int] = (1, 50),
    k: tuple[int, int] = (1, 10),
):
    """
    Conduct hyperparameter optimization using Optuna.

    This function performs hyperparameter optimization by training a BioClassifier model
    with different parameter combinations suggested by Optuna on dataset_names. The optimization process
    involves both unsupervised and supervised training phases.

    Parameters
    ----------
    trial : optuna.Trial
        Current trial object from Optuna.
    dataset_name : str
        Name of the dataset to use for training.
    slow : bool
        Whether to use the slow exact method for steady-state calculation.
    unsupervised_epochs : int
        Total number of epochs for unsupervised training.
    supervised_epochs : int
        Number of epochs for supervised training.
    unsupervised_minibatch_size : tuple of int
        Range (min, max) for unsupervised batch size.
    supervised_minibatch_size : tuple of int
        Range (min, max) for supervised batch size.
    p : tuple of int
        Range for Lebesgue norm exponent.
    delta : tuple of float
        Range for anti-Hebbian factor.
    unsupervised_lr : tuple of float
        Range for unsupervised learning rate.
    supervised_lr : tuple of float
        Range for supervised learning rate.
    power_m_loss : tuple of int
        Range for powered absolute loss exponent.
    power_n : tuple of int
        Range for hidden-layer activation exponent.
    beta_tanh : tuple of float
        Range for tanh scaling factor.
    h_star : tuple of float
        Range for steady-state activity threshold (slow mode only).
    tau_L : tuple of float
        Range for time constant (slow mode only).
    w_inh : tuple of float
        Range for lateral inhibition strength (slow mode only).
    inhibition_timesteps : tuple of int
        Range for number of timesteps for lateral inhibition (slow mode only).
    k : tuple of int
        Range for number of top active neurons (fast mode only).

    Returns
    -------
    tuple[float, float, float, float]
        Tuple containing the average final validation error rate, average unsupervised
        improvement, average supervised improvement, and time taken for training.

    Notes
    -----
    The function divides the unsupervised training into 4 stages and performs
    supervised training after each stage. It saves the model if it achieves
    better performance than the previous best trial.

    The optimization can be pruned early if the trial shows poor performance,
    using Optuna's pruning mechanism.
    """

    unsupervised_epochs = trial.suggest_int(
        "unsupervised_epochs", unsupervised_epochs[0], unsupervised_epochs[1]
    )
    supervised_epochs = trial.suggest_int(
        "supervised_epochs", supervised_epochs[0], supervised_epochs[1]
    )
    hidden_size = trial.suggest_int("hidden_size", hidden_size[0], hidden_size[1])

    unsupervised_minibatch_size = trial.suggest_int(
        "unsupervised_minibatch_size",
        unsupervised_minibatch_size[0],
        unsupervised_minibatch_size[1],
    )
    supervised_minibatch_size = trial.suggest_int(
        "supervised_minibatch_size",
        supervised_minibatch_size[0],
        supervised_minibatch_size[1],
    )
    p = trial.suggest_int("p", p[0], p[1])
    delta = trial.suggest_float("delta", delta[0], delta[1])
    unsupervised_lr = trial.suggest_float(
        "unsupervised_lr", unsupervised_lr[0], unsupervised_lr[1]
    )
    supervised_lr = trial.suggest_float(
        "supervised_lr", supervised_lr[0], supervised_lr[1], log=True
    )
    power_m_loss = trial.suggest_int("power_m_loss", power_m_loss[0], power_m_loss[1])
    power_n = trial.suggest_int("power_n", power_n[0], power_n[1])
    beta_tanh = trial.suggest_float("beta_tanh", beta_tanh[0], beta_tanh[1])
    project_w = trial.suggest_categorical("project_w", ["True", "False"])
    normalize_w_update = trial.suggest_categorical(
        "normalize_w_update", ["True", "False"]
    )
    use_original_supervised = trial.suggest_categorical(
        "use_original_supervised", ["True", "False"]
    )

    if slow:
        h_star = trial.suggest_float("h_star", h_star[0], h_star[1])
        tau_L = trial.suggest_float("tau_L", tau_L[0], tau_L[1])
        w_inh = trial.suggest_float("w_inh", w_inh[0], w_inh[1])
        inhibition_timesteps = trial.suggest_int(
            "inhibition_timesteps",
            inhibition_timesteps[0],
            inhibition_timesteps[1],
        )
        k = None
    else:
        h_star, tau_L, w_inh, inhibition_timesteps = None, None, None, None
        k = trial.suggest_int("k", k[0], k[1])

    start_time = time()

    avg_unsupervised_improvements = []
    avg_supervised_improvements = []
    final_validation_errors = []

    for dataset_name in dataset_names:
        model = BioClassifier(
            dataset_name,
            R=1,
            use_validation=True,
            hidden_size=hidden_size,
            unsupervised_minibatch_size=unsupervised_minibatch_size,
            supervised_minibatch_size=supervised_minibatch_size,
            slow=slow,
            p=p,
            delta=delta,
            h_star=h_star,
            tau_L=tau_L,
            w_inh=w_inh,
            inhibition_timesteps=inhibition_timesteps,
            k=k,
            power_m_loss=power_m_loss,
            power_n=power_n,
            beta_tanh=beta_tanh,
            project_w=project_w,
            normalize_w_update=normalize_w_update,
            use_original_supervised=use_original_supervised,
            supervised_lr=supervised_lr,
            unsupervised_lr=unsupervised_lr,
        )

        model.train_unsupervised_and_supervised(
            unsupervised_epochs=unsupervised_epochs,
            supervised_epochs=supervised_epochs,
            eval_every=eval_every,
            test_supervised_epochs=test_supervised_epochs,
            plot_errors=False,
        )

        # Compute core metrics

        unsupervised_errors = list(model.unsupervised_validation_errors.values())
        supervised_errors = list(model.supervised_validation_errors.values())

        if not unsupervised_errors:
            unsupervised_errors = supervised_errors
        avg_unsupervised_improvements.append(
            (
                sum(
                    unsupervised_errors[i] - unsupervised_errors[i + 1]
                    for i in range(len(unsupervised_errors) - 1)
                )
                / (len(unsupervised_errors) - 1)
                if len(unsupervised_errors) > 1
                else 100 - unsupervised_errors[0]
            )
        )

        avg_supervised_improvements.append(
            sum(
                supervised_errors[i] - supervised_errors[i + 1]
                for i in range(len(supervised_errors) - 1)
            )
            / (len(supervised_errors) - 1)
            if len(supervised_errors) > 1
            else 100 - supervised_errors[0]
        )

        final_validation_errors.append(supervised_errors[-1])

    time_taken = time() - start_time

    avg_unsupervised_improvement = sum(avg_unsupervised_improvements) / len(
        avg_unsupervised_improvements
    )
    avg_supervised_improvement = sum(avg_supervised_improvements) / len(
        avg_supervised_improvements
    )

    print("Time taken for training: {:.2f} seconds".format(time_taken))
    for dataset, error_rate in zip(dataset_names, final_validation_errors):
        print(f"Final validation error rate for {dataset}: {error_rate:.2f}")
    print("Mean unsupervised improvement: {:.2f}".format(avg_unsupervised_improvement))
    print("Mean supervised improvement: {:.2f}".format(avg_supervised_improvement))

    return (
        *final_validation_errors,
        avg_unsupervised_improvement,
        avg_supervised_improvement,
        time_taken,
    )


def download_and_plot_dataset(
    dataset: Literal["MNIST", "CIFAR-10", "CIFAR-10-AutoAugmented"],
):
    """
    Download the requested dataset and plot a small 3x3 grid of sample images.

    Parameters
    ----------
    dataset : {"MNIST", "CIFAR-10", "CIFAR-10-AutoAugmented"}
        The name of the dataset to download and visualize.
    """
    if dataset == "MNIST":
        torch_dataset = datasets.MNIST(
            root="../data",
            train=True,
            download=True,
        )
    elif dataset == "CIFAR-10":
        torch_dataset = datasets.CIFAR10(
            root="../data",
            train=True,
            download=True,
        )
    elif dataset == "CIFAR-10-AutoAugmented":
        torch_dataset = datasets.CIFAR10(
            root="../data",
            train=True,
            download=True,
            transform=v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
        )
    else:
        raise ValueError(
            "Invalid dataset name. Please choose from 'MNIST' or 'CIFAR-10'."
        )

    # Plot a 3x3 grid of sample images
    if dataset in ["MNIST", "CIFAR-10", "CIFAR-10-AutoAugmented"]:
        fig, axes = plt.subplots(3, 3, figsize=(6, 6))
        for i, ax in enumerate(axes.flatten()):
            image, label = torch_dataset[i]
            if dataset == "MNIST":
                cmap = "gray"
            else:
                cmap = None
            ax.imshow(image, cmap=cmap)
            ax.set_title(f"Label: {label}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()


def plot_hyperparameter_optimization_bars(
    trials_df: pd.DataFrame, hyperparameters: list[str]
) -> None:
    """
    Plot bar charts for hyperparameter optimization results.

    Parameters
    ----------
    trials_df : pd.DataFrame
        DataFrame containing the trials data, including hyperparameters and error rates.
    hyperparameters : list[str]
        List of hyperparameter names to plot.

    Returns
    -------
    None
        Displays the plot.

    """

    def calculate_bins(n):
        """Calculate the number of bins using Sturges' formula."""
        return math.ceil(math.log2(n) + 1)

    # Define a list of unique colors from Plotly's qualitative palette
    # Ensure there are enough colors; if not, colors will cycle
    color_palette = pc.qualitative.Plotly
    num_colors = len(color_palette)

    # Initialize the figure
    fig = go.Figure()

    # List to keep track of trace indices and their corresponding hyperparameters
    trace_info = []

    for idx, hyperparam in enumerate(hyperparameters):
        dtype = trials_df[hyperparam].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            # For numeric hyperparameters, perform binning
            unique_values = trials_df[hyperparam].nunique()

            # Calculate number of bins using Sturges' formula
            num_bins = calculate_bins(len(trials_df))

            # Decide whether to bin based on the number of unique values
            # If unique values exceed num_bins, perform binning
            if unique_values > num_bins:
                # Create bins using pd.cut with automatic binning
                trials_df[f"{hyperparam}_binned"] = pd.cut(
                    trials_df[hyperparam], bins=num_bins
                )
                grouped = (
                    trials_df.groupby(f"{hyperparam}_binned", observed=True)[
                        "error_rate"
                    ]
                    .mean()
                    .reset_index()
                )

                # Format bin labels to two decimal places
                def format_bin_label(interval):
                    return f"{interval.left:.2f} - {interval.right:.2f}"

                grouped[f"{hyperparam}_binned"] = grouped[f"{hyperparam}_binned"].apply(
                    format_bin_label
                )
                x_values = grouped[f"{hyperparam}_binned"]
                x_title = f"{hyperparam} (Binned)"
            else:
                # If not many unique values, treat as categorical with observed=True
                grouped = (
                    trials_df.groupby(hyperparam, observed=True)["error_rate"]
                    .mean()
                    .reset_index()
                )

                # Format numerical values to two decimal places
                grouped[hyperparam] = grouped[hyperparam].apply(lambda x: f"{x:.2f}")
                x_values = grouped[hyperparam]
                x_title = f"{hyperparam} Value"

            # Assign a unique color to each hyperparameter
            color = color_palette[idx % num_colors]

            trace = go.Bar(
                x=x_values, y=grouped["error_rate"], name=hyperparam, marker_color=color
            )
            fig.add_trace(trace)

        elif pd.api.types.is_object_dtype(dtype):
            # For categorical hyperparameters, group by category with observed=True
            grouped = (
                trials_df.groupby(hyperparam, observed=True)["error_rate"]
                .mean()
                .reset_index()
            )
            x_values = grouped[hyperparam].astype(str)
            x_title = hyperparam

            # Assign a unique color to each hyperparameter
            color = color_palette[idx % num_colors]

            trace = go.Bar(
                x=x_values, y=grouped["error_rate"], name=hyperparam, marker_color=color
            )
            fig.add_trace(trace)

        else:
            # Handle other data types if necessary
            continue  # Skip if data type is not handled

        # Append trace information for dropdown
        trace_info.append(
            {
                "name": hyperparam,
                "x": x_values,
                "y": grouped["error_rate"],
                "x_title": x_title,
            }
        )

    # Create buttons for the dropdown menu
    buttons = []
    for i, info in enumerate(trace_info):
        # Set visibility: only the i-th trace is visible
        visibility = [False] * len(trace_info)
        visibility[i] = True

        # Define the button
        button = dict(
            label=info["name"],
            method="update",
            args=[
                {"visible": visibility},
                {
                    "title": f"{info['name']} vs Average Error Rate",
                    "xaxis": {"title": info["x_title"]},
                    "yaxis": {"title": "Average Error Rate"},
                    "showlegend": False,  # Hide legend as only one trace is visible
                },
            ],
        )
        buttons.append(button)

    # Update layout with dropdown menu
    fig.update_layout(
        title="Hyperparameter vs Average Error Rate",
        xaxis_title="Select a Hyperparameter",
        yaxis_title="Average Error Rate",
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=1.15,
                y=1,
                xanchor="left",
                yanchor="top",
                direction="down",
                showactive=True,
            )
        ],
        annotations=[
            dict(
                text="Select Hyperparameter:",
                showarrow=False,
                x=1.15,
                y=1.05,
                xref="paper",
                yref="paper",
                align="left",
            )
        ],
        width=900,  # Increased width to accommodate dropdown
        height=600,
        template="plotly_white",
    )

    # Set initial visibility (show first hyperparameter)
    for i in range(len(trace_info)):
        fig.data[i].visible = i == 0

    # Display the figure
    fig.show()
