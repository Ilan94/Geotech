# %% [markdown]
# 4.1 Automatic Square Cropping of Scan Path

# Generate a minimal set of informative 5 m×5 m square regions that cover the entire GPS scan path. We implement two methods:

#1. **Brute‑force Greedy Cover (not time‑efficient)**: treat as a set‑cover problem, selecting at each step the square covering the most uncovered points.
#2. **Efficient Sweep‑Line Cover**: partition the path by sweeping along x and stacking vertical squares in each strip (O(N log N) runtime).



# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# 1. Load and visualize the scan path
data = np.load('gps_path.npy')  # shape (N,2)
print(f"Loaded {data.shape[0]} points. x: {data[:,0].min():.2f}–{data[:,0].max():.2f}, y: {data[:,1].min():.2f}–{data[:,1].max():.2f}")

plt.figure(figsize=(6,4))
plt.plot(data[:,0], data[:,1], '-k', linewidth=0.5)
plt.title('Raw Scan Path')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.axis('equal')
plt.tight_layout()
plt.show()

# %% [markdown]
## 2. Brute‑Force Greedy Set‑Cover

#We treat each path point as an element to cover. At each iteration:

#1. Generate candidate squares anchored at each *uncovered* point (using that point as each corner of a 5 m square).  
#2. Count how many uncovered points each candidate would cover.  
#3. Pick the square covering the most, mark those points covered, and repeat.

#This picks highly informative regions first but is **O(n²)** or worse, and thus slow for large N.



# %%
def greedy_cover(points, side=5.0):
    pts = points.copy()
    N = pts.shape[0]
    covered = np.zeros(N, dtype=bool)
    regions = []

    # Precompute offsets for corners
    offsets = np.array([[0,0], [-side,0], [0,-side], [-side,-side]])

    while not covered.all():
        best_cover = 0
        best_region = None
        best_indices = None

        # indices of uncovered points
        idx_un = np.where(~covered)[0]
        for idx in idx_un:
            x, y = pts[idx]
            # try each corner placement
            for dx, dy in offsets:
                x0, y0 = x + dx, y + dy
                x1, y1 = x0 + side, y0 + side
                # find uncovered points inside this square
                in_square = (~covered) & \
                    (pts[:,0] >= x0) & (pts[:,0] <= x1) & \
                    (pts[:,1] >= y0) & (pts[:,1] <= y1)
                count = in_square.sum()
                if count > best_cover:
                    best_cover = count
                    best_region = (x0, y0, x1, y1)
                    best_indices = in_square
        # record and mark covered
        regions.append(best_region)
        covered[best_indices] = True
        print(f"Placed region {best_region} covering {best_cover} points; {covered.sum()}/{N} covered")

    return regions

# %%
# Run greedy cover (warning: slow for large N)
# Uncomment to execute on small subsets or for testing
#sample = data if data.shape[0] < 2000 else data[np.random.choice(len(data),2000,False)]
#greedy_regions = greedy_cover(sample, side=5.0)
#print(f"Greedy cover produced {len(greedy_regions)} regions on sample of {len(sample)} points")

# %% [markdown]
## 3. Efficient Sweep‑Line Cover

#We sweep along the x‑axis in 5 m strips, and within each strip, stack 5 m squares vertically to cover all points in that strip:

#- **Sort** points by x.  
#- **Iterate**: take the next point’s x as the left edge of a strip `[x, x+5]`.  
#- **Collect** all points in that strip, sort by y, then place stacked squares of height 5 m to cover them.  
#- **Advance** to the next point beyond the current strip.  

#This runs in roughly **O(N log N)** time due to sorting.



# %%
def sweep_line_cover(points, side=5.0):
    pts = points[np.argsort(points[:,0])]  # sort by x
    N = pts.shape[0]
    regions = []
    i = 0
    while i < N:
        x0 = pts[i,0]
        x1 = x0 + side
        # collect strip
        strip_idx = []
        j = i
        while j < N and pts[j,0] <= x1:
            strip_idx.append(j)
            j += 1
        strip = pts[strip_idx]
        # sort strip by y and stack
        strip_sorted = strip[np.argsort(strip[:,1])]
        k = 0
        M = strip_sorted.shape[0]
        while k < M:
            y0 = strip_sorted[k,1]
            regions.append((x0, y0, x1, y0 + side))
            # skip covered in this square
            k += 1
            while k < M and strip_sorted[k,1] <= y0 + side:
                k += 1
        # advance past this strip
        i = j
    return regions

# %%
sweep_regions = sweep_line_cover(data, side=5.0)
print(f"Sweep-line cover produced {len(sweep_regions)} regions.")

# Plot result
plt.figure(figsize=(8,5))
plt.plot(data[:,0], data[:,1], '-k', lw=0.5, label='Scan Path')
for (x0, y0, x1, y1) in sweep_regions:
    plt.gca().add_patch(plt.Rectangle((x0, y0), 5, 5,
                                      edgecolor='r', facecolor='none', lw=1))
plt.title('Sweep‑Line Crop Regions')
plt.xlabel('X [m]'); plt.ylabel('Y [m]')
plt.axis('equal')
plt.legend(); plt.tight_layout()
plt.show()

# %% [markdown]
## 4. Conclusion
## **Greedy cover** picks highly informative squares first but is computationally expensive (practical only for small N or with optimizations).  
## **Sweep‑line cover** runs in O(N log N) and yields a near‑minimal set of 5 m×5 m squares covering the entire path with no empty regions.  

## Choose the **sweep‑line method** for large datasets, and consider the **greedy approach** when you need to prioritize the densest clusters strictly.
