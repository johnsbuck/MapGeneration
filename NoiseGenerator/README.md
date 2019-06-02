# Noise Generator

Generates natural noise using Perlin-based 
noise algorithms with hash table.

There are 3 noises:

1. Perlin
    - 1 to 3 dimensions
2. Value
    - 1 to 3 dimensions
3. Simplex
    - 1 to 2 dimensions
    
 Actual value scales (min and max) can vary 
 with each noise. If you want to experiment 
 with matching, you may want to feature scale
 from 0 to 1 or -1 to 1 
 
 (X - X.min / (X.max - X.min))