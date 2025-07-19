import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 1) Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# 2) Grid & material
nx, ny, nz = 64, 64, 640           # cells in x, y, z
dx, dy, dz = 0.1, 0.1, 0.1         # meters per cell
Vp, Vs, rho = 4000.0, 2500.0, 2500.0
mu  = rho * Vs**2
lam = rho * Vp**2 - 2*mu

# 3) Time stepping — now go to 80 ms
dt = 5e-6                          # 5 µs timestep
nt = int(0.08 / dt)               # ≈16,000 steps → 80 ms total
times = np.arange(nt, dtype=np.float32) * dt
t_ms   = times * 1e3              # milliseconds

# 4) Source at pile head
src_i, src_j, src_k = nx//2, ny//2, 0
amp_scale = 1e10

# 5) Ricker wavelet (100 Hz)
def ricker_np(t, f0=100.0):
    a = np.pi * f0 * (t - 1.0/f0)
    return (1 - 2*a*a) * np.exp(-a*a)

wave = ricker_np(times, f0=100.0)
wave -= wave.mean()

# 6) 3×3×3 FD Laplacian + free‑end BC
kernel = torch.zeros((3,1,3,3,3), dtype=torch.float32, device=device)
c0 = -2*(1/dx**2+1/dy**2+1/dz**2)
kernel[:,0,1,1,1] = c0
kernel[:,0,0,1,1] = kernel[:,0,2,1,1] = 1/dx**2
kernel[:,0,1,0,1] = kernel[:,0,1,2,1] = 1/dy**2
kernel[:,0,1,1,0] = kernel[:,0,1,1,2] = 1/dz**2

def laplacian_free(u):
    up = F.pad(u.unsqueeze(0), (1,1, 1,1, 1,1), mode='reflect')
    return F.conv3d(up, kernel, padding=0, groups=3).squeeze(0)

# 7) Allocate fields & storage
u = torch.zeros((3,nx,ny,nz), device=device)
v = torch.zeros_like(u)
f = torch.zeros_like(u)
head_raw = np.zeros(nt, dtype=np.float32)
toe_raw  = np.zeros(nt, dtype=np.float32)

# 8) Time‑stepping loop
print_every = max(1, nt//100)
for it in range(nt):
    f[2, src_i, src_j, src_k] = amp_scale * wave[it]
    lap = laplacian_free(u)
    v  += dt * (mu/rho * lap + f/rho)
    u  += dt * v
    f.zero_()
    head_raw[it] = u[2, src_i, src_j, src_k].item()
    toe_raw[it]  = u[2, src_i, src_j, nz-1].item()
    if it % print_every == 0:
        print(f"\rProgress: {it/nt*100:5.1f}% ", end="", flush=True)
print("\nSimulation complete.")

# 9) Normalize
head_norm = head_raw / np.max(np.abs(head_raw))
toe_norm  = toe_raw  / np.max(np.abs(toe_raw))

# 10) Hilbert envelope
S = np.fft.rfft(toe_raw)
H = np.ones_like(S); H[1:-1] = 2
analytic = np.fft.irfft(S * H, n=nt)
toe_env = np.abs(analytic)
toe_env_norm = toe_env / np.max(toe_env)

# 11) Plot raw, normalized, envelope with 32 ms & 64 ms lines
t1, t2 = 32.0, 64.0  # ms

plt.figure(figsize=(10,10))

# Panel 1: raw signals
ax1 = plt.subplot(3,1,1)
ax1.plot(t_ms, head_raw*1e6, label="Head (raw µm)", color="C0", alpha=0.6)
ax1.plot(t_ms, toe_raw*1e6,  label="Toe  (raw µm)", color="C1", alpha=0.6)
ax1.axvline(t1, color='red',    ls='--', lw=2, label="32 ms")
ax1.axvline(t2, color='magenta',ls='-.', lw=2, label="64 ms")
ax1.set_xlim(0,80)
ax1.set_ylabel("Uz (µm)")
ax1.set_title("Raw Displacement at Head & Toe (0–80 ms)")
ax1.legend(loc="upper right"); ax1.grid(True)

# Panel 2: normalized
ax2 = plt.subplot(3,1,2, sharex=ax1)
ax2.plot(t_ms, head_norm, label="Head (norm)", color="C0")
ax2.plot(t_ms, toe_norm,  label="Toe  (norm)", color="C1")
ax2.axvline(t1, color='red',    ls='--', lw=2)
ax2.axvline(t2, color='magenta',ls='-.', lw=2)
ax2.set_xlim(0,80)
ax2.set_ylabel("Normalized Uz")
ax2.set_title("Normalized Displacement at Head & Toe")
ax2.legend(loc="upper right"); ax2.grid(True)

# Panel 3: envelope
ax3 = plt.subplot(3,1,3, sharex=ax1)
ax3.plot(t_ms, toe_env_norm, label="Toe env (norm)", color="C1", lw=2)
ax3.axvline(t1, color='red',    ls='--', lw=2)
ax3.axvline(t2, color='magenta',ls='-.', lw=2)
ax3.set_xlim(0,80)
ax3.set_xlabel("Time (ms)")
ax3.set_ylabel("Envelope (norm)")
ax3.set_title("Hilbert Envelope of Toe Displacement")
ax3.legend(loc="upper right"); ax3.grid(True)

plt.tight_layout()
plt.show()
