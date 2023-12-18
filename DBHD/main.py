import numpy as np
import matplotlib.pyplot as plt



# Generate x values
x = np.linspace(0, 1, 100)

# Evaluate the function
y = -x**4 + 0.5

# Plot the function
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y = -x^4 + 0.5')
plt.grid(True)
plt.show()

