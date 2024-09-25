import numpy as np

def expression(x, t, dd, sigma, N0):
    # Calculate the numerator
    numerator = -dd * np.exp(-(x**2 / (2 * (2 * dd * t + sigma**2)))) * N0 * x * np.sqrt(2 * np.pi * sigma)

    # Calculate the denominator
    denominator = np.sqrt(2 * np.pi) * np.sqrt((2 * dd * t) / sigma + sigma) * (2 * dd * t + sigma**2)

    # Calculate the result
    result = numerator / denominator

    return result

# Example usage
x = np.linspace(-200, 200, 100)  # Example array for x
t = 0.2  # Example value for t
dd = 0.1  # Example value for dd
sigma = 40/np.sqrt(8 * np.log(2))  # Example value for sigma
N0 = 1  # Example value for N0

result = expression(x, t, dd, sigma, N0)
print(result)

# Plot the result
import matplotlib.pyplot as plt

# plt.plot(x, result)
# plt.xlabel('x')
# plt.ylabel('Expression value')
# plt.title('Plot of the expression')
# plt.show()

from scipy.special import erf
from scipy.constants import pi
import matplotlib.pyplot as plt

def compute_expression(alpha, dd, t, x):
    # Compute the exponential term
    exp_term = np.exp(alpha * (-2 * x + 2 * dd * t * alpha) / 2)
    
    # Compute the Gaussian term
    gaussian_term = np.exp(-((x - 2 * dd * t * alpha) ** 2) / (4 * dd * t))
    sqrt_term = np.sqrt(pi) * np.sqrt(dd * t)
    first_part = gaussian_term / ( sqrt_term)
    
    # Compute the error function term
    erf_term = erf((-x + 2 * dd * t * alpha) / (2 * np.sqrt(dd * t)))
    second_part = -alpha * (-1 + erf_term)
    
    # Combine terms
    result = exp_term * (first_part - second_part)
    # result = erf_term 
    return result

# Parameters
alpha = 7.0  # Given value for alpha
dd = 0.1     # Given value for dd
t = 0.1      # Example value for t
x_values = np.linspace(0, 1, 500)
y_values = compute_expression(alpha, dd, t, x_values)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label=f'alpha={alpha}, dd={dd}, t={t}')
plt.xlabel('x')
plt.ylabel('Function Value')
plt.title('Plot of the Function')
plt.legend()
plt.grid(True)
plt.show()