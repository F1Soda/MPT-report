import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def get_portfolio_expectation_and_std(r, cov, w):
    expectation = np.dot(r, w)
    std = np.sqrt(w.T @ cov @ w)
    return expectation, std


fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=150, sharey=True)

# Estimated expected returns and covariances.
e = np.array([2, 1, 1.3, 4, 0.5])
V = np.array([
    [90, 22, 20, 5, 10],
    [22, 30, 15, 20, 3],
    [20, 15, 40, 6, 11],
    [5, 20, 6, 95, 1],
    [10, 3, 11, 1, 70]
])

# Find efficient frontier via sampling.
xx = np.empty(5000)
yy = np.empty(5000)
ss = np.empty(5000)
for i in range(5000):
    w = np.random.dirichlet([1] * 5)
    yy[i], xx[i] = get_portfolio_expectation_and_std(e, V, w)
    ss[i] = yy[i] / xx[i]  # Sharpe ratio w/ risk-free rate == 0.
ssn = (ss - ss.min()) / (ss.max() - ss.min())

ax.scatter(xx, yy, c=ssn, cmap='Blues', s=5)


# Find efficient frontier numerically. By Article
def efficient_portfolio(targ):
    def objective(w):
        return w.T @ V @ w - targ * e.T @ w

    resp = minimize(objective,
                    x0=np.random.dirichlet([1] * 5),
                    method='SLSQP',
                    bounds=[(-2, 2)] * 5,
                    constraints=[{'type': 'eq', 'fun': lambda w: 1 - w.sum()},
                                 {'type': 'eq', 'fun': lambda w: np.dot(e, w) - targ}])
    return resp.x


xx = np.empty(100)
yy = np.empty(100)
# `targ` is `K` is Equation 9.
for i, targ in enumerate(np.linspace(0, 4, 100)):
    w = efficient_portfolio(targ)
    yy[i], xx[i] = get_portfolio_expectation_and_std(e, V, w)
ax.plot(xx, yy, label="Numerical(SLSQP)")

# Find efficient frontier by formulas. By presentation
ones = np.ones(V.shape[0])
V_inv = np.linalg.inv(V)

A = ones.T @ V_inv @ e  # or equivalently: A = e.T @ V_inv @ ones
B = e.T @ V_inv @ e
C = ones.T @ V_inv @ ones
D = B * C - A ** 2

g = 1 / D * (B * (V_inv @ ones) - A * (V_inv @ e))
h = 1 / D * (C * (V_inv @ e) - A * (V_inv @ ones))


def get_efficient_portfolio(expectation):
    return g + h * expectation


xx = np.empty(100)
yy = np.empty(100)

# `targ` is expectation.
for i, targ in enumerate(np.linspace(0.5, 3.5, 100)):
    w = get_efficient_portfolio(targ)
    yy[i], xx[i] = get_portfolio_expectation_and_std(e, V, w)
ax.plot(xx, yy, label=f"Analytical")


ax.set_ylabel('Ожидаемая доходность')
ax.set_xlabel('Стандартное отклонение доходов')
ax.legend()
plt.show()
