# Unsupervised Learning by Competing Hidden Units ⚡🧠

*Adam Amer & Alessandra Bontempi, 20882 Computational Neuroscience, Bocconi University, 2024/2025*

Welcome to our repository! This project explores **unsupervised learning** in neural networks through a biologically inspired algorithm introduced in a [paper](https://arxiv.org/abs/1806.10181) by Dmitry Krotov and John J. Hopfield. We replicate and extend their work with our own experiments, including hyperparameter optimization, data augmentation, and a fully-unsupervised CNN variant.

---

## 📚 Table of Contents

- [Quickstart 🚀](#quickstart-)
- [The Original Paper 📝](#the-original-paper-)
- [Our Extension ✨](#our-extension-)
- [File Structure 📂](#file-structure-)
- [Our Results 📊](#our-results-)
- [Future Research 🔮](#future-research-)

---

## Quickstart 🚀

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Set up your Python environment:**
    - **Using [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended)**
    ```bash
     uv sync
    ```
    - **Alternatively, using venv**
    ```bash
     python -m venv competing-unsupervised-learning
     pip install .
    ```


[    3. **Docker Option:**
    - If you prefer containerization, build and run the Docker image:
     ```bash
     docker build -t unsupervised-learning .
     docker run -it --rm unsupervised-learning
     ```]: #


### Running the Experiments

- **Replication and Extension Notebooks:**
  - Open `replication_and_extension.ipynb` in Jupyter Notebook to explore the experiments. Make sure to select the environment created previously.
- **CNN Experiment:**
  - Use `flippedCNN.ipynb` for CNN experimentation with flipped filters.

---

## The Original Paper 📝

[*Unsupervised learning by competing hidden units, Dmitry Krotov and John J. Hopfield (2019)*](https://arxiv.org/abs/1806.10181)

**Objective and architecture**  
The authors investigate learning early-layer representations in an ANN without supervision, relying on a local “biological” update rule instead of end-to-end backprop. They aim to replicate a biologically plausible model that learns useful feature detectors in early layers without labeled data.

**Algorithm (overview)**  
- Input layer: reads images (e.g., CIFAR-10, MNIST).  
- Hidden layer: neurons compete to represent inputs; weights update via local Hebbian-like rules.  
- Output layer: produces final representation.

**Mathematical framework and synaptic plasticity rule**  
Krotov and Hopfield’s feed-forward architecture imposes local weight updates inspired by Hebbian plasticity and includes non-linear activations. A core weight-update rule is defined by:

$$
\tau_{L}\frac{dW_{i}}{dt} = g(Q)\bigl(R^{P}v_{i} - \langle W, v\rangle W_{i}\bigr)
\quad \text{where } Q = \frac{\langle W, v \rangle}{\langle W, W \rangle^{\frac{p-1}{p}}},
$$

ensuring synaptic changes depend only on neuron activities. Larger $p$ emphasizes big weights and suppresses smaller ones, maintaining a homeostatic constraint.

**A biologically inspired algorithm**  
To encourage competition among hidden units, the activity update rule is:

$$
\frac{\tau\,\mathrm{d}h_u}{\mathrm{d}t} = I_u - w_{\text{inh}} \sum_{v \neq u} r(h_v)\,-\,h_u
\quad\text{with } I_u = \langle W_u, v\rangle,\; r(h_u) = \max(h_u,0).
$$

Only a fraction of hidden units remain active, ensuring diverse feature detection. Stronger weights specialize to the inputs that activated them the most.

---

## Our Extension ✨

We built upon the original work in several exciting ways:

### 1. Bayesian Hyperparameter Optimization
- **Why?** The error rate is highly sensitive to hyperparameters.
- **Outcome:** For MNIST, achieved ~3% error rate; for CIFAR-10, ~14% error rate over limited epochs.
- **Tool:** Implemented the Tree-Structured Parzen Estimator to fine-tune parameters.

### 2. Multi-Dataset Hyperparameter Optimization
- **Datasets:** CIFAR-100, MNIST, and FashionMNIST.
- **Goal:** Find a set of global hyperparameters achieving an average error rate of ~8% across datasets.

### 3. Data Augmentation Experiment
- Tested the impact of a learned augmentation policy on CIFAR-100.
- **Observation:** The augmentation worsened performance dramatically.

### 4. Fully-Unsupervised CNN
- **Motivation:** Leverage CNN parameter-sharing benefits while maintaining biological plausibility.
- **Approach:**
  - Two custom convolutional layers with competition and Gram-Schmidt orthogonalization.
  - A simple classifier head trained with Adam (while keeping the convolutional filters frozen).
- **Result:** Achieved an average validation accuracy of ~70% with a significant reduction in parameters (only 225k).


## Our Results 📊

We report results from both the replication of the original paper and our own experiments:

- **Krotov and Hopfield's Results:**
  - **MNIST:** Unsupervised learning converges with 2000 hidden units, then fine-tuned with standard SGD.
  - **CIFAR-10:** The biologically inspired algorithm achieves training/test errors comparable to traditional backpropagation after fine-tuning.

- **Our Experiments:**
  - **Bayesian Optimization:** Significant improvements with optimized hyperparameters.
  - **Multi-Dataset Global Optimization:** Achieved an average error rate of ~8% across CIFAR-100, MNIST, and FashionMNIST.
  - **CNN Extension:** Reduced parameter count by 90% (compared to the best FCN) while reaching ~70% validation accuracy.

Interactive plots and detailed error graphs are available in the `plots/` directory.

---

## Future Research 🔮

While our work achieved promising results, several avenues remain for exploration:

- **E-I Neurons:** Implementing biologically plausible constraints to distinguish excitatory and inhibitory neurons.
- **Spike-Timing Dependent Plasticity (STDP):** Investigating time-based spiking dynamics for further biological realism.
- **Scaling Up:** Testing deeper CNN architectures while maintaining low parameter counts and biological plausibility.

We invite you to explore these ideas or contribute your own insights!

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.