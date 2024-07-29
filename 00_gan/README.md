## Generative Adversarial Networks (GANs)
A generative adversarial network is a deep learning system that is comprised of two models that are simultaneously trained to compete against each other in a zero sum game.

Both models can be defined as neural networks, with one being the generator $G$ and the other being the discriminator $D$.

---
The generator neural network $G$ with parameters $\theta_{g}$:
- learns the probability distribution, $p_{g}$ associated with the training dataset $\textbf{\textit X}$ of real examples
- takes a (Gaussian) noise vector $\textbf{\textit z}$ as input and produces a mapping from the noise space to the data space: $G(\textbf{\textit z};\theta_{g}) = \hat{\textbf{\textit x}}$
- wants to create fake data that is indistinguishable from real data such that it can successfully fool the discriminator model $D$

The discriminator neural network $D$ with parameters $\theta_{d}$:
- estimates the probability that a data sample came from the training data $\textbf{\textit X}$ rather than the generator $G$
- takes a data point $\textbf{\textit x}$ as input and produces a single scalar that represents the likelihood of the datapoint being real as opposed to synthetic: $D(\textbf{\textit x};\theta_{d}) \in (0,1)$
- wants to maximize its ability to correctly distinguish between real and fake data

---
The generator and discriminator work against each other within a value function $V(G,D)$ whose output value $G$ is trying to minimize and $D$ is trying to maximize: 
$$\min_{G}\max_{D}V(G,D)=\mathbb{E}_{x\sim p_{data}(\textbf{\textit x})}[\log D(\textbf{\textit x})] + \mathbb{E}_{z \sim p_{z}(\textbf{\textit z})}[\log(1 - D(G(\textbf{\textit z})))]$$

The loss functions of both the generator and discriminator models' parameters can be derived from this value function into the following stochastic gradients for a minibatch of $m$ samples:

- gradient wrt. $G$'s weights: $\nabla_{\theta_{g}} \frac{1}{m} \sum_{i=1}^{m}[\log (1 - D(G(\textbf{z}^{(i)})))]$

- gradient wrt. $D$'s weights: $\nabla_{\theta_{d}} \frac{1}{m} \sum_{i=1}^{m}[\log D(\textbf{x}^{(i)}) + \log (1 - D(G(\textbf{z}^{(i)})))]$


---

In theory, training the entire system can result in a convergence where:
- $G$'s output is indistinguishable from real data, ie. the training data distribution is perfectly recovered

- $D$'s output is equal to $\frac{1}{2}$ everywhere, ie. it can do no better than random binary guessing

