import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Вхідні дані
r = np.array([5, 28, 21, 23, 25])/100  # Відсоткова дохідність
q = np.array([0, 2.5, 1.5, 5.5, 2.6]) / 100  # Відсотковий ризик
p = np.array([0, 1, 2, 4.5, 6.5]) / 100  # Витрати у відсотках


# Функція максимізації
def maximize_return(a):
    def objective(x):
        return -np.sum((r-p) * x)

    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x*(1+p)) - 1},
        {'type': 'ineq', 'fun': lambda x: a - q * x}
    )


    # Межі для x (0 ≤ x ≤ 1) та t (t ≥ 0)
    bounds = [(0, 1) for _ in range(len(r))]
    x0 = np.ones(len(r)) / len(r)

    result = minimize(objective, x0, bounds=bounds, constraints=constraints)
    return result

risk_levels = np.arange(0, 0.1, 0.001)
returns = []

# Оптимізація
for a in risk_levels:
    res = maximize_return(a)
    returns.append(-res.fun if res.success else None)

# Побудова графіку
plt.figure(figsize=(10, 8))
plt.plot(risk_levels, returns, 'b.', label='Risk-Return Relationship')
plt.axvline(x=0.006, color='r', linestyle='--', label='Inflection Point (a=0.006)')
plt.xlabel('a (Risk Level)', fontsize=12)
plt.ylabel('Q (Return)', fontsize=12)
plt.title('Relationship of Risk and Return', fontsize=14)
plt.legend(fontsize=10)
plt.grid()
plt.xlim(0, 0.1)
plt.ylim(0, max(returns) * 1.1)
plt.show()
