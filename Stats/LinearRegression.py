# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))  # Inputs (regressors)
y = np.array([5, 20, 14, 32, 22, 38])  # Outputs (response)

model = LinearRegression()
model.fit_intercept = True
model.normalize = False
model.copy_X = True
model.n_jobs = None
model.positive = False

model.fit(x, y)  # calculate the optimal values of the weights 𝑏₀ and 𝑏₁ (x,y)

r2 = model.score(x, y)

print(x)
print(y)
print(f'''coefficient of determination: {r2}''')
