# Linear Supervised Transfer Learning  

Copyright (C) 2019 - Benjamin Paassen  
Machine Learning Research Group  
Center of Excellence Cognitive Interaction Technology (CITEC)  
Bielefeld University

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.

## Introduction

This Python3 library provides several algorithms to learn a linear mapping from
an $`m`$-dimensional source space to an $`n`$-dimensional target space, such that
a classification model trained in the source space becomes applicable in the
target space. The source space model is assumed to be a labelled mixture of
Gaussians. Note that this library assumes that the relation between the source
and target space is (approximately) linear and will necessarily fail if the
relationship is highly nonlinear. Further note that this library requires a
few labelled target space data points to work (typically, even ~10 data points
are enough). However, not all classes are required, since the learned linear
transformation generalizes across classes.

If you intend to use this library in academic work, please cite [our paper][1].

## QuickStart Guide

For a quick start we recommend to take a look at the demo in the notebook
`demo.ipynb`. In this file we demonstrate how to perform transfer learning
on example data. For the actual transfer learning, we recommend to initialize
one of the following models, depending on your source space model:

1. `transfer-learning.LGMM_transfer_model` : If you have a full labelled
   Gaussian mixture model.
2. `transfer-learning.SLGMM_transfer_model` : If you have a labelled Gaussian
   mixture model with shared precision matrices.
3. `transfer-learning.Local_LVQ_transfer_model` : If you have a learning vector
   quantization model with individual metric learning matrices.
4. `transfer-learning.LVQ_transfer_model` : If you have a learning vector
   quantization model with shared metric learning matrix or no metric learning
   at all.

Note that models 2 and 4 are _much_ faster to train compared to models 1 and 3
(refer to the next section for more information on that).

All these models follow the [scikit-learn][2] convention, i.e. you need to call
the `fit` function with target space data first and then the `predict` function
to map new target space data to the source space according to the learned
mapping.

## Background

The basic idea of our transfer learning approach is to maximize the likelihood
of target space data according to the source space data distribution _after_ the
learned transfer function $`h`$ has been applied. More precisely, assume we have
a data set $`(\vec x_1, y_1), \ldots, (\vec x_m, y_m)`$ of target data points
$`\vec x_j \in \mathbb{R}^n`$ and their labels $`y_j \in \{1, \ldots, L\}`$.
Then, we wish to maximize the joint probability

```math
\max_h \prod_{j=1}^m p\Big(h(\vec x_j), y_j\Big)
```

To make this optimization problem feasible, we introduce two assumptions:
First, that $`p(x, y)`$ can be modelled by a labelled Gaussian mixture
model (lGMM) and, second, that $`h`$ can be approximated by a linear function.
In more detail, that means the following.

### Labelled Gaussian Mixture Models

A _labelled Gaussian mixture model_ assumes that data is generated by a mixture
of $`K`$ Gaussians, each of which has a prior $`P(k)`$, a data generating
Gaussian density $`p(\vec x|k)`$, and a label generating distribution
$`P(y|k)`$. Using these distributions, we can derive the joint probability
density $`p(\vec x, y)`$ as follows.

```math
p(\vec x, y) = \sum_{k=1}^K p(\vec x, y, k) = \sum_{k=1}^K p(\vec x, y|k) \cdot P(k)
```

Our model assumes that $`\vec x`$ and $`y`$ are conditionally independent given
the component index $`k`$, such that we can re-write:

```math
p(\vec x, y) = \sum_{k=1}^K p(\vec x|k) \cdot P(y|k) \cdot P(k)
```

Note that $`p(\vec x|k)`$ is a [multivariate Gaussian probability density][5]
with parameters for the mean $`\vec \mu_k`$ and the precision matrix
$`\Lambda_k`$. Also note that this model is a proper generalization over
standard [Gaussian mixture models][6] and that many of the GMM properties
translate directly to lGMMs. More precisely, we obtain a standard GMM by setting
the label distribution $`P(y|k)`$ to a uniform distribution and leaving it
unchanged during training. Alternatively, we also obtain a standard GMM by
assigning the same label to all data points.

Also note that lGMMs generalize over learning vector quantization models if we
apply a scaling trick to the precision matrices (for more details on this,
refer to [our paper][1]).

### Expectation Maximization transfer learning

Our assumption that the transfer function $`h`$ is approximately linear implies
that $`h`$ can be re-written as $`h(\vec x) \approx H \cdot \vec x`$ for some
matrix $`H`$. Thus, our transfer learning problem becomes:

```math
\max_H \prod_{j=1}^m \sum_{k=1}^K p(H \cdot \vec x|k) \cdot P(y|k) \cdot P(k)
```

Due to the product of sums, a direct optimization of this expression is
infeasible. However, we can apply an [expectation maximization][6] scheme.
In particular, we initialize $`H`$ with the identity matrix (padded with zeros
wherever necessary) and then iteratively perform the following two steps:

1. _Expectation_: We compute the posterior $`p(k|H \cdot \vec x_j, y_j)`$
	for the current transfer matrix $`H`$, all data points $`j`$ and all
    Gaussian components $`k`$, yielding a matrix $`\Gamma \in \mathbb{R}^{K \times m}`$
    with entries $`\gamma_{k,j} = p(k|H \cdot \vec x_j, y_j)`$. The full
    expression for the posterior is given in [our paper][1].
2. _Maximization_: We maximize the expected log likelihood under fixed posterior,
    i.e.
    
    ```math
    \max_H \sum_{j=1}^m \sum_{k=1}^K \gamma_{k, j} \cdot \log\big[p(H \cdot \vec x_j, y_j| k)\big]
    ```

	This optimization problem can be shown to be convex und thus lends itself
	for optimization techniques like l-BFGS. Even better, if the precision
	matrix $`\Lambda_k`$ is shared across all Gaussians $`k`$, the problem has
	a closed-form solution, namely
	
	```math
	H = W \cdot \Gamma \cdot X^T \cdot (X \cdot X^T + \lambda \cdot I)^{-1}
	```
	
	where $`W = (\vec \mu_1, \ldots, \vec \mu_K)`$, $`X = (\vec x_1, \ldots, \vec x_m)`$,
	$`\lambda`$ is a (small) regularization constant, and $`I`$ is the identity
	matrix. Due to this closed form solution, the `SLGMM_transfer_model` and
	the `LVQ_model` are much faster to train compared to the
	`LGMM_transfer_model` and the `Local_LVQ_transfer_model`.

For more detailed background, please refer to [our paper][1].

## Contents

This library contains the following files.

* `demo.ipynb` : A demo script illustrating how to use this library.
* `LICENSE` : A copy of the GPLv3 license.
* `lgmm.py` : A file to train labelled Gaussian mixture models with or without
  shared precision matrices.
* `lgmm_test.py` : A set of unit tests for `lgmm.py`.
* `README.md` : This file.
* `transfer_learning.py` : The actual transfer learning models.
* `transfer_learning_test.py` : A set of unit tests for `transfer_learning.py`.

## Licensing

This library is licensed under the [GNU General Public License Version 3][7].

## Dependencies

This library depends on [NumPy][3] for matrix operations and [SciPy][4] for
optimization.

## Literature

* Paassen, B., Schulz, A., Hahne, J., and Hammer, B (2018).
  _Expectation maximization transfer learning and its application for bionic hand prostheses_.
  Neurocomputing, 298, 122-133. doi:[10.1016/j.neucom.2017.11.072](https://doi.org/10.1016/j.neucom.2017.11.072). [Link][1]

<!-- References -->

[1]: https://arxiv.org/abs/1711.09256 "Paassen, B., Schulz, A., Hahne, J., and Hammer, B (2018). Expectation maximization transfer learning and its application for  bionic hand prostheses. Neurocomputing. accepted"
[2]: https://scikit-learn.org/stable/ "Scikit-learn homepage"
[3]: http://numpy.org/ "Numpy homepage"
[4]: https://scipy.org/ "SciPy homepage"
[5]: https://en.wikipedia.org/wiki/Multivariate_normal_distribution "Wikipedia page to multivariate Gaussian distributions"
[6]: http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.HomePage "Barber, D. (2012). _Bayesian Reasoning and Machine Learning_ Cambridge University Press."
[7]: https://www.gnu.org/licenses/gpl-3.0.en.html "The GNU General Public License Version 3"
