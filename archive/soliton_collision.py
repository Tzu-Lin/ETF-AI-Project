# soliton_collision_scipy_solver.py (使用 SciPy 高階積分器的最終穩定版)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp
from tqdm import tqdm

# --- 1. 準備工作 ---
FRAME_DIR = "animation_frames"
if os.path.exists(FRAME_DIR):
    print(f"正在清理舊的幀圖片目錄: {FRAME_DIR}")
    for f in os.listdir(FRAME_DIR):
        os.remove(os.path.join(FRAME_DIR, f))
else:
    os.makedirs(FRAME_DIR)
    print(f"已創建目錄: {FRAME_DIR}")

# --- 2. 定義 KdV 方程式的右側項 (RHS) ---
# 我們要解的方程是 du/dt = f(u)，這個函數就是定義 f(u)
# f(u) = -6*u*u_x - u_xxx
def kdv_rhs(t, u, k):
    """計算 KdV 方程式在頻域的右側項 (時間導數)"""
    u_hat = np.fft.fft(u)
    
    # 計算 u_x 和 u_xxx
    u_x_hat = 1j * k * u_hat
    u_xxx_hat = -(1j * k)**3 * u_hat
    
    # 計算 u*u_x
    u_x = np.fft.ifft(u_x_hat).real
    u_ux = u * u_x
    u_ux_hat = np.fft.fft(u_ux)
    
    # 在頻域計算整個右側項
    du_hat_dt = -6 * u_ux_hat - u_xxx_hat
    
    # 返回時域的時間導數
    return np.fft.ifft(du_hat_dt).real

# --- 3. 設置模擬參數和初始條件 ---
L = 200.0
N = 512
dx = L / N
x = np.arange(-L/2, L/2, dx)
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

# 初始條件
c1, x1 = 25.0, -40.0
soliton1 = 0.5 * c1 * (1 / np.cosh(0.5 * np.sqrt(c1) * (x - x1)))**2
c2, x2 = 10.0, -10.0
soliton2 = 0.5 * c2 * (1 / np.cosh(0.5 * np.sqrt(c2) * (x - x2)))**2
u0 = soliton1 + soliton2

# 模擬時間範圍
t_span = [0, 8.0]
t_eval = np.linspace(t_span[0], t_span[1], 200) # 我們希望得到200幀的結果

# --- 4. 使用 SciPy 的 solve_ivp 進行求解 ---
print("\n--- 孤立子碰撞模擬程式 (SciPy 穩定版) ---")
print("正在使用高階積分器 solve_ivp 求解 KdV 方程式...")
# solve_ivp 會返回一個包含所有時間點解的物件
# 我們需要將 k 作為額外參數傳遞給 kdv_rhs
solution = solve_ivp(
    kdv_rhs, 
    t_span, 
    u0, 
    t_eval=t_eval, 
    args=(k,),
    method='RK45' # 使用經典的 Runge-Kutta 45 演算法
)
print("求解完成！")

# solution.y 是一個 (N, num_frames) 的陣列，包含了所有時間點的解
results = solution.y.T # 轉置為 (num_frames, N) 更方便處理

# --- 5. 逐幀保存圖片 ---
print(f"\n將開始生成 {len(results)} 幀圖片...")
fig, ax = plt.subplots(figsize=(10, 6))

# 配置圖表
ax.set_xlabel("Space (x)")
ax.set_ylabel("Amplitude (u)")
ax.set_xlim(-L/2, L/2)
ax.set_ylim(-1, 15)
ax.grid(True)
line, = ax.plot(x, u0, lw=2)

for i, u_t in enumerate(tqdm(results, desc="保存幀")):
    line.set_ydata(u_t)
    ax.set_title(f"Two-Soliton Collision (t = {t_eval[i]:.2f})")
    filename = os.path.join(FRAME_DIR, f"frame_{i:04d}.png")
    fig.savefig(filename)

plt.close(fig)
print(f"\n>>> 成功生成 {len(results)} 幀圖片到 '{FRAME_DIR}' 目錄下！ <<<")

# --- 6. 提示用戶執行 FFmpeg 指令 ---
print("\n所有幀已生成完畢。請在您的終端機中執行以下指令來合成影片：")
print("\n" + "="*70)
print(f"ffmpeg -y -framerate 30 -i {FRAME_DIR}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p soliton_collision_final.mp4")
print("="*70 + "\n")