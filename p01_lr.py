import util
import numpy as np
import matplotlib.pyplot as plt

def calc_grad(X, Y, theta):
    """Calcula el gradiente de la pérdida con respecto a tita."""
    m, n = X.shape

    lambda_value = 0.0225

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y)) # + lambda_value

    #grad = - 1 / m * (probs * Y) @ X 

    return grad


def logistic_regression(X, Y):
    """Entrena un modelo de regresión logística."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            print('Terminadas %d iteraciones' % i)
            print(np.linalg.norm(prev_theta - theta))
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Convergencia en %d iteraciones' % i)
            break
    
    return


def main():
    print('==== Entrenando modelo en dataset A ====')
    Xa, Ya = util.load_csv('./data/ds1_a.csv', add_intercept=True)
    # logistic_regression(Xa, Ya)

    plt.scatter(Xa[:,1], Xa[:,2], c=Ya)
    plt.show()

    print('\n==== Entrenando modelo en dataset B ====')
    Xb, Yb = util.load_csv('./data/ds1_b.csv', add_intercept=True)

    plt.scatter(Xb[:,1], Xb[:,2], c=Yb)
    plt.show()  

    mean = np.mean(Xb)
    std = np.std(Xb)
    X_normalized = (Xb - mean) / std

    # logistic_regression(X_normalized, Yb)    



if __name__ == '__main__':
    main()
