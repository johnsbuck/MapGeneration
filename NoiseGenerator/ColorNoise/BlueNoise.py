from scipy import signal
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import numpy as np

data = np.random.uniform(0, 1, (255, 255))
plt.imshow(data, cmap="gray", vmin=0, vmax=1)
plt.show()

for _ in range(5):
    data = filters.gaussian_filter(data, 0.5)
plt.imshow(data, cmap="gray", vmin=0, vmax=1)
plt.show()
