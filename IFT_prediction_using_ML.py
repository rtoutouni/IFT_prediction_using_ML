import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.tree import DecisionTreeRegressor

X = np.loadtxt('data.txt')
y = X[:,2].reshape(-1, 1)
X = X[:,0:2]
#Normalization
scalerx = MinMaxScaler()
scalery = MinMaxScaler()
X=scalerx.fit_transform(X)
y=scalery.fit_transform(y)
#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#Neural network
kernel = DotProduct() + WhiteKernel()
regressors = [
    ("Linear Regression", LinearRegression()),
    ("MLP", MLPRegressor(hidden_layer_sizes=(300,300, ), max_iter=1000, alpha=0.001)),
    ("SVR", SVR()),
    ("Bayesian Ridge",BayesianRidge()),
    ("Kernel Ridge",KernelRidge()),
    ("GP", GaussianProcessRegressor()),
    ("GP-kernel",GaussianProcessRegressor(kernel=kernel)),
    ("Decision Tree",DecisionTreeRegressor())
    ]
for name, regr in regressors:
    regr.fit(X_train, y_train)
    pred= regr.predict(X_test).reshape(-1,1)
    print("Performance of %s on the train set"%name, regr.score(X_train, y_train))
    print("Performance of %s on the test set"%name, regr.score(X_test, y_test))
    sb_y, sb_p =scalery.inverse_transform(y_test), scalery.inverse_transform(pred)
    plt.scatter(sb_p,sb_y, color='red')
    plt.axline((0, 0), (12, 12), color='black')
    plt.xlabel("IFT_Prediction (mN/m)")
    plt.ylabel("IFT_Experiment (mN/m)")
    plt.title(name)
    plt.text(1,7,"R^2=%06.4f"%regr.score(X_test, y_test))
    plt.savefig(name+".png",dpi=120)
    plt.clf()