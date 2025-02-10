import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Вхідні дані
r = np.array([5, 28, 21, 23, 25]) / 100  # Відсоткова дохідність
q = np.array([0, 2.5, 1.5, 5.5, 2.6]) / 100  # Відсотковий ризик
p = np.array([0, 1, 2, 4.5, 6.5]) / 100  # Витрати у відсотках

def minimize_risk(K):
    n = len(r)  # Кількість активів

    # Цільова функція: мінімізація допоміжної змінної t
    def objective(vars):
        t = vars[-1]  # Останній елемент масиву — це змінна t
        return t

    # Обмеження
    constraints = [
        {'type': 'ineq', 'fun': lambda vars: np.sum((r - p) * vars[:-1]) - K},  # Дохід ≥ K
        {'type': 'eq', 'fun': lambda vars: np.sum((1 + p) * vars[:-1]) - 1},    # Бюджетне обмеження
    ]
    
    # Додаємо обмеження для кожного активу: q_i * x_i ≤ t
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda vars, i=i: vars[-1] - q[i] * vars[i]})

    # Межі для x (0 ≤ x ≤ 1) та t (t ≥ 0)
    bounds = [(0, 1) for _ in range(n)] + [(0, None)]
    x0 = np.ones(n + 1) / (n + 1)  # Початкове наближення (включаючи t)

    result = minimize(objective, x0, bounds=bounds, constraints=constraints)
    return result

K_levels = np.arange(0.05, 0.25, 0.01)  
risks = []  # Список для збереження мінімальних ризиків

# Оптимізація для кожного значення K
for K in K_levels:
    res = minimize_risk(K)
    risks.append(res.fun if res.success else None)

# Побудова графіка
plt.figure(figsize=(10, 8))
plt.plot(K_levels, risks, 'b.', markersize=8, label='Risk vs Revenue Level (K)')
plt.axvline(x=0.2, color='r', linestyle='--', label='Critical Revenue Level (K=0.2)')
plt.xlabel('K (Minimum Revenue Level)', fontsize=12)
plt.ylabel('Minimized Risk (a)', fontsize=12)
plt.title('Minimized Risk vs Revenue Level (Model II)', fontsize=14)
plt.legend(fontsize=10)
plt.grid()
plt.xlim(0.05, 0.25)
plt.ylim(0, max([r for r in risks if r is not None]) * 1.1)
plt.show()
