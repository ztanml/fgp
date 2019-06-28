# FGP
  * A Gaussian Process that satifies several fairness criteria
  * Provides code and data for the paper: *Learning Fair Representations for Kernel Models*, available at https://arxiv.org/abs/1906.11813

### Basic Usage
In Matlab:
```matlab
> hyp = fgp(x_train,y_train,s_train,1,1,1,'covkfn','fgp_rbf','covkpar',3.2365,'fair','eo');
```
This command trains a GP model using training data: x_train (n-by-p, each row is a feature vector), y_train (n-by-1 label vector), s_train (n-by-d, each row is a vector of protected attributes), m=1, d=1, and eps=1. 'fgp_rbf' specifies the RBF kernel and 'covkpar' specifies the bandwidth, i.e., 3.2365, for the RBF kernel. 'eo' specifies equalized odds as the fairness criterion to use.

Upon completion, hyp holds the model structure, and hyp.f(x_test) gives the prediction. See the following examples for demonstrations.

### Examples
  * plot_adult_sp.m: plots statistical parity vs prediction error on the UCI adult dataset
  * plot_adult_eop.m: plots equality of opportunity vs prediction error on the UCI adult dataset
  * plot_adult_eo.m: plots equalized odds vs prediction error on the UCI adult dataset
  * runCC.m: plots statistical parity vs prediction error with multiple protected attributes on the Communities & Crime dataset

### Obtaining Kernel Parameters using Cross-Validation
cv_acc.m provides example code for obtaining the kernel parameters using Bayesian Optimization and cross-validation:
```matlab
hyp=cv_acc(x_train,y_train,s_train,1,1,1,15)
```

### Datasets
  * The data/ folder contains datasets processed using https://github.com/jmikko/fair_ERM and https://github.com/jkomiyama/fairregresion
