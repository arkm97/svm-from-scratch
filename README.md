# svm-from-scratch

### Motivation
I wanted to better understand how a support vector machine (SVM) works.  Here I build one
from scratch.

Originally interested in finding an existing ML technique that could be improved using
a quantum machine (SVMs require efficient inner products and matrix inversion, methods
for which there are quantum algorithms that may provide exponential speedups; see the discussion [here](https://arxiv.org/pdf/1307.0411.pdf) about inner product algorithms and the discussion [here](https://arxiv.org/pdf/0811.3171.pdf) about matrix inversion on quantum machines), I wanted to better understand how SVMs work.
Specifically, I wanted to understand the formulation of the kernel and the optimization
involved in solving the dual problem.

### Background
The SVM solves the following binary classification problem:

given data points ![x_j] and targets ![y_j], where ![j=1,...,M], find the maximum-margin hyperplane that separates the two classes (![y_j=-1] and ![y_j=1]).  

Let ![w] be the vector normal to the hyperplane.  We want to find ![w] that satisfies

![inequality]

The dual formulation of the above is equivalent to maximizing the following over the multipliers ![alpha]:

![dual]

subject to the constraints ![sum] and ![y_j>=0].  The matrix ![K] defines the kernel of the SVM; I've chosen ![kernel].  The parameters of the plane are recovered from ![w=alpha.x] and ![b] for ![j] such that ![alpha_j!=0]

for reference see eq(1) [here](https://arxiv.org/pdf/1307.0471.pdf).


[x_j]: http://chart.apis.google.com/chart?cht=tx&chl=\vec%20x_j%20\in%20\mathbb{R}^{1%20\times%20N}
[y_j]: http://chart.apis.google.com/chart?cht=tx&chl=y_j%20=%20\pm%201
[j=1,...,M]: http://chart.apis.google.com/chart?cht=tx&chl=j%20=%201,%20\dots,%20M
[y_j=-1]: http://chart.apis.google.com/chart?cht=tx&chl=y_j%20=%201
[y_j=1]: http://chart.apis.google.com/chart?cht=tx&chl=y_j%20=%20-1
[w]: http://chart.apis.google.com/chart?cht=tx&chl=\vec%20w
[inequality]: http://chart.apis.google.com/chart?cht=tx&chl=y_j%20(\vec%20w%20\cdot%20\vec%20x_j%2Bb)%20\geq%201
[alpha]: http://chart.apis.google.com/chart?cht=tx&chl=\vec%20\alpha
[dual]: http://chart.apis.google.com/chart?cht=tx&chl=L(\vec%20\alpha)%20=%20\vec%20y%20\cdot%20\vec%20\alpha%20-%20\frac%201%202%20\vec%20\alpha%20K%20\vec%20\alpha^T
[sum]:http://chart.apis.google.com/chart?cht=tx&chl=\sum_{j=1}^M%20\alpha_j%20=%200
[y_j>=0]: http://chart.apis.google.com/chart?cht=tx&chl=y_j%20\alpha_j%20\geq%200
[K]: http://chart.apis.google.com/chart?cht=tx&chl=K
[kernel]: http://chart.apis.google.com/chart?cht=tx&chl=K_{jk}%20=%20k(\vec%20x_j,%20\vec%20x_k)%20=%20\vec%20x_j%20\cdot%20\vec%20x_k
[w=alpha.x]: http://chart.apis.google.com/chart?cht=tx&chl=\vec%20w%20=%20\vec%20\alpha%20\cdot%20\vec%20x
[b]: http://chart.apis.google.com/chart?cht=tx&chl=b%20=%20y_j%20=%20\vec%20w%20\cdot%20\alpha_j
[j]: http://chart.apis.google.com/chart?cht=tx&chl=j
[alpha_j!=0]: http://chart.apis.google.com/chart?cht=tx&chl=\alpha_j%20\neq%200


### Packages

```
numpy
pandas
cvxpy
sklearn
```

### Acknowledgements
[Quantum support vector machine for big data classification](https://arxiv.org/pdf/1307.0471.pdf)
