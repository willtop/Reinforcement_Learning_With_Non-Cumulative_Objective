import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import trange

METHOD = 2

n_data = int(5e5)

if METHOD ==1 :
    print("Method 1: just have an empty list and build from it one by one")
    # Result: incurring run-time error
    memory = []
    for i in trange(n_data):
        a = np.random.sample(10000).astype(np.float32)
        memory.append(a)
else:
    print("Method 2: Have the memory pre-allocated with enough space, just assign it")
    memory = np.zeros([n_data, 10000], dtype=np.float32)
    for i in trange(n_data):
        a = np.random.sample(10000).astype(np.float32)
        memory[i] = a

print("script finished successfully!")
    
