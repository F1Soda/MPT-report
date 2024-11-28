import matplotlib.pyplot as plt
import numpy as np


def portfolio_perf(r, s, w, p):
    ret = np.dot(r, w)
    std = np.sqrt(np.dot(s ** 2, w ** 2) + 2 * np.prod(w) * np.prod(s) * p)
    return ret, std


r = np.array([2, 1])  # Returns.
s = np.array([10, 6])  # Standard deviations.
p_values = [-1, -0.5, 0, 0.5, 1]  # Correlation values.

fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=150)

for p in p_values:
    xx = np.empty(1000)
    yy = np.empty(1000)
    i = 0
    for w1 in np.linspace(-0.3, 1.6, 1000):
        w2 = 1 - w1
        w = np.array([w1, w2])
        yy[i], xx[i] = portfolio_perf(r, s, w, p)
        i += 1
    ax.plot(xx, yy, label=f'p = {p}')

# Plot portfolios at specific weight combinations.
for w1 in [0, 1]:
    w2 = 1 - w1
    w = np.array([w1, w2])
    yp, xp = portfolio_perf(r, s, w, p_values[0])  # Use the first p value for labeling.
    if w1 == 0:
        ax.axvline(xp, ls=':')
        ax.text(xp + 0.4, yp, 'A')
    elif w1 == 1:
        ax.text(xp + 0.4, yp - 0.1, 'B')
    c = 'r' if w1 in [0, 1] else 'b'
    size = 60 if w1 in [0, 1] else 30
    ax.scatter(xp, yp, c=c, s=size, zorder=2)



ax.set_ylabel('Ожидаемая доходность')
ax.set_xlabel('Стандратное отклонение доходов')
ax.legend()
plt.show()
