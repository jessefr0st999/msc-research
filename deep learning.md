- Representation learning involves a machine automatically discovering representations needed for detection or classification from raw data.

- A basic network consists of an output calculated from a linear combination of input variables, the weights $$w$$ and bias $$b$$ of which are the tunable parameters of the system.
  - Enables linearly separable functions to be modelled.
  - A single perceptron is based off this system, using a step function as an activation function of the output vector $$x$$ i.e. output is 1 if $$x.w+b>0$$ and 0 otherwise.
  
- Deep learning involves multiple layers of representation, each transforming a given level to a higher and more abstract level, starting from the raw data.
  - A non-linear activation function (e.g. tanh, sigmoid, ReLU) is applied to values between layers in order to allow non-linear functions of the input variables to be modelled.
    - Advantages of tanh and sigmoid are that they ensure that small changes in input result in small changes in output; their usage is motivated by their nature as "smoothed out" step functions.
  - Hyperparameters include the following:
    - Width (nodes in a layer);
    - Depth (number of layers);
    - Architecture (arrangement of layers and nodes);
    - Learning rate;
    - Dropout rates;
    - Weight constraints;
    - Momentum (for gradient-based optimisation),
  - Appropriate hyperparameter values for these determined by intuition (e.g. more abstract situations lend themselves to deeper networks) and experimentation.
  - Requires minimal engineering by hand, instead taking advantage of increases in the amount of available computation and data.
  - According to [this](https://ieeexplore.ieee.org/abstract/document/1165576/) paper, two hidden layers is sufficient for creating classification regions of any desired shape, while further theory states that a network with one hidden layer can approximate any required function. In practice, however, these do not specify the width required of the layers.
  - Dropout layers nullify the contribution of some neurons towards the next layer. This ensures that the first batch of training samples does not disproportionately influence the learning.
    - Dropout layers are implemented with a certain nullification probability.
    - Only used when training; not used for making predictions.
  
- Gradient descent:

  - Pass an input vector forward through a system to calculate the output.

  - Calculate an objective/cost/error function from the distance between the output and the labels, a quadratic cost function is currently used, where the sum is over the training input (feature) vectors $$x$$:
    $$
    C(w,b)=\frac{1}{2n}\sum_{x}||y(x)-a(x,w,b)||^2
    $$

    - $$w$$ is the vector containing the network's weights
    - $$b$$ is the vector containing the network's biases
    - $$y$$ is the label (desired output) for the feature $$x$$
    - $$a$$ is the output for the $$x$$
    - $$n$$ is the number of training inputs

  - Considering $$(w, b)$$ as the vector $$v$$, $$\Delta{C}\approx\nabla{C}\cdot\Delta{v}$$ for small $$\Delta{v}$$. Choosing $$\Delta{v}=-\eta\nabla{C}$$, where the learning rate $$\eta$$ is a small positive parameter, causes a decrease in $$C$$ upon the change $$v\rightarrow{v}-\eta\nabla{C}$$.

  - Generally, this will be done for many examples, with a sum taken as shown in the cost function to calculate the average gradient before applying the updates to $$v$$.

  - Stochastic gradient descent is the most commonly used optimisation technique, involving the gradient being calculated and parameters adjusted using a sample of $$m$$ inputs from the total number of training inputs $$n$$.

  - This process is then repeated for many small sets of examples from the training set until the average of the objective function stops decreasing.]

  - Alternate loss function: negative log likelihood, where the likelihood is the probability that an event that has already occurred would yield a specific outcome.

- After training, the performance of the system is measured against a set of examples different to the training set (the test set).

- Certain classification problems may require the input-output function to be insensitive to irrelevant input variations (e.g. position or orientation variation of an object) but sensitive to other minute variations:
  - Hence, shallow classifiers require feature extractors which can produce representations which are selective to the aspects of the data important to discrimination.
  - Feature extractors can be hand designed or, with a general-purpose learning procedure, learned automatically.
  
- Convolutional neural networks:
  - Generally composed of several stages of:
    - Convolution;
    - Dropout (optional);
    - Activation;
    - Pooling;
  - Convolutional layers are used to detect local groupings of features from the previous layer.
  - The most commonly used activation function for CNNs is ReLU, due to its discarding of irrelevant results (i.e. negative values) and easy differentiability.
  - Pooling layers are used to merge semantically similar features, reducing variation when detected elements in the previous layer vary in position and appearance.
  - After this, many CNNs connect to another neural network, with the results of the convolution steps becoming input neurons.

- Training procedure:
  - Run in batches over training data
  - One loop over the training dataset is an "epoch"; training over multiple epochs (with the dataset looped through in a different order) is motivated by the aim to prevent bias towards the end of the dataset in the first epoch.

- Cross-entropy cost function:

  - $$\frac{\partial Q}{\partial w}$$ and $$\frac{\partial Q}{\partial b}$$ are both small for neuron outputs close to 0 or 1 using a sigmoid activation function and quadratic cost function; this causes learning to be slow when neurons output values in the vicinity of 0 or 1.

  - For activation $$a=\sigma(z)$$, labels $$y$$ and summing over $$n$$ inputs indexed by $$x$$, the cross-entropy cost function is defined as follows:
    $$
    C=-\frac{1}{n}\sum_x[y\log{a}+(1-y)\log(1-a)]
    $$

  - This cost function is derived from making the neuron learn faster for greater error, with $$\frac{\partial Q}{\partial w_i}=x_i(a-y)$$ and $$\frac{\partial Q}{\partial b}=a-y$$.

  - removing $$\sigma'(z)$$ dependence from $$\frac{\partial Q}{\partial w_i}$$ and $$\frac{\partial Q}{\partial b}$$ to ensure that, .

  - Has the following properties:
  
    - $$C\ge0$$
    - $$C$$ approaches 0 if $$y$$ is 0 or 1 as $$a$$ approaches 0 or 1 respectively.
  
  - Weight and bias derivatives for a single neuron with $$n$$ input neurons:
    $$
    \frac{\partial Q}{\partial w_i}=\frac{1}{n}\sum_xx_i[\sigma(z)-y]
    $$
    $$
    \frac{\partial Q}{\partial b}=\frac{1}{n}\sum_x[\sigma(z)-y]
    $$
  
  - Hence, the rate at which the weight and bias learn depends on $$\sigma(z)-y$$, the error in the output.
  
  - For multiple layers (indexed by $$L$$) with many neurons (indexed by $$j$$), the cross-entropy is generalised as follows:
    $$
    C=-\frac{1}{n}\sum_x\sum_j[y_j\log{a_j^L}+(1-y_j)\log(1-a_j^L)]
    $$
  
- Softmax and log-likelihood:

  - Softmax is an alternate activation function used for normalising a set of inputs into a probability distribution (i.e. so that the inputs are between 0 and 1 and all sum to 1).

  - For given layer $$L$$, with $$k$$  weighted inputs $$z_i^L$$, softmax is defined as follows:

  $$
  a_i^L=\frac{e^{z_i^L}}{\sum_k{e^{z_i^L}}}
  $$

  - Applying the log-likelihood cost function $$C=-\log{a_y^L}$$ ensures that $$C$$ approaches 0 as $$a$$ approaches 1 while becoming large as $$a$$ approaches 0.
  - Partial derivatives of $$C$$ with respect to the weights and bias are also independent of the activation function's derivative, hence avoiding learning slowdown.
  - Used when interpreting output activations as probabilities is desired.

- Validation data:

  - A separate dataset to test and training data, used to evaluate different trial hyperparameter choices such as number of epochs (i.e. to prevent overfitting), learning rate and network architecture.
  - This is done on a dataset separate to the test data to ensure that the test data remains agnostic (and therefore not overfitted) to the choice of hyperparameters.

-  Weight decay/L2 regularisation:

  - Involves adding the following extra term to the cost function containing the sums of the squares of all weights in the network:
    $$
    C=C_0+\frac{\lambda}{2n}\sum_ww^2
    $$

  - This introduces a preference for small weights, with larger weights only being allowed if they considerably improve $$C_0$$.

  - The regularisation parameter $$\lambda>0$$ scales the emphasis on the preference for small weights versus minimisation of $$C_0$$.

  - Effectively, a regularised network learns to respond to types of evidence seen often across the training set, rather than to specific inputs.

  - Empirically, regularised networks tend to generalise better than unregularised networks.

  - Bias is not often regularised.

  - L1 (summing over $$|w|$$ instead of $$w^2$$) regularisation, dropout and artificially increasing the training set size (applying operations that reflect real-world variation e.g. rotating, translating, skewing and distorting images) are other regularisation techniques.

- Weight initialisation:

  - Initialising all weights and biases with normalised Gaussian random variables mean 0 and standard deviation 1 ($$Z(0, 1)$$) results in the the weighted input $$z=\sum_iw_ix_i+b$$ itself being normally distributed with standard deviation $$\sqrt{n_{x_i=1}+1}$$, where $$n_{x_i=1}$$ is the number of activated input neurons, and the $$+1$$ is due to the bias.
  - This results in (generally) a very broad Gaussian distribution for $$z$$, and hence $$\sigma(z)$$ being close to either 0 or 1. This causes small weight changes to only make very small changes to the first layer of hidden neurons' activation and hence the rest of the network and the cost function.
  - Hence, learning will proceed rather slowly.
  - Instead initialising the weights as $$Z(0, \frac{1}{\sqrt{n}})$$, where $$n$$ is the number of input variables, and leaving the bias as $$Z(0, 1)$$ results in $$z$$ having a Gaussian distribution with mean 0 and standard deviation $$\sqrt{3/2}$$, and hence neurons in the first hidden layer being less likely to saturate.
  - Making this change results in faster learning and, occasionally, long-term behaviour.

- Choosing hyperparameters:

  - Minimise problem and hence dataset required for training for ease and speed of hyperparameter trials.
  - Start with the simplest network likely to perform meaningful learning and expand complexity later.

- AutoML
  - Involves procedures for automating hyperparameter optimisation
  - Requires a particular model selection strategy

- Residual neural networks
  - Involves shortcut connections skipping one or more layers.
  - Used in deep networks to attempt to counteract vanishing gradients and resultant diminishing returns.
    - When network depth increases, accuracy tends to saturate and then degrade rapidly after sufficient epochs of training.
    - This is not to be confused with overfitting.
  - Generally, involves adding a layer's input to its output (i.e. outputting $$f(x) +x$$ where $$f(x)$$ is the function applied by the layer).
  - This makes the identity function present by default rather than the zero function, allowing for better learning in the case where the identity function may be optimal, as it should be easier to map the learning residual to zero rather than fit an identity mapping.

- Generative adversarial networks

- Fisher information matrix (FIM):

  - Correlation between validation accuracy and trace of FIM is studied in *Catastrophic Fisher Explosion: Early Phase Fisher Matrix Impacts Generalization* (Jastrzebski et al, 2021).

    > Our main contribution is to show that implicit regularization effects due to using a large learning rate can be explained by its impact on the trace of the FIM (Tr(F)), a quantity that reflects the local curvature, from the beginning of training.

  - FIM is calculated as the expected value of the square of the gradient of the cross-entropy loss function with respect to the parameters, hence giving a measure of the magnitude of the rate of change of the loss with respect to the parameters.

  - For small learning rates, regularising the trace of the FIM early in training allowed the model to generalise better and give better long-term results.

  - Effectively the Fisher penalty slows down learning on noisy examples in the dataset.
