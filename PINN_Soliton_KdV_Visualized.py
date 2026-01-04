# -*- coding: utf-8 -*-
"""
PINN_Soliton_KdV_Visualized.py

使用物理資訊神經網路(PINN)求解KdV方程式，並視覺化孤立子的傳播。
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 檢查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 定義一個簡單的全連接神經網路
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 網路結構：輸入層(x, t -> 2個神經元), 4個隱藏層, 輸出層(u -> 1個神經元)
        self.net = nn.Sequential(
            nn.Linear(2, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 1)
        )

    def forward(self, x, t):
        # 將 x 和 t 拼接在一起作為網路輸入
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# 2. 準備訓練數據
# KdV方程式的孤立子解 u(x,t) = 0.5 * c * sech^2(0.5 * sqrt(c) * (x - c*t))
c = 10.0 # 波速

# 初始條件 (t=0)
x_initial = torch.linspace(-20, 20, 500).view(-1, 1)
u_initial = 0.5 * c * (1 / torch.cosh(0.5 * np.sqrt(c) * x_initial))**2
t_initial = torch.zeros_like(x_initial)

# 物理損失的搭配點 (在整個時空域中隨機採樣)
num_collocation_points = 20000
x_collocation = (torch.rand(num_collocation_points, 1) * 40 - 20) # x 在 [-20, 20] 範圍
t_collocation = torch.rand(num_collocation_points, 1) * 2       # t 在 [0, 2] 範圍

# 將所有數據移動到GPU (如果可用)
x_initial, t_initial, u_initial = x_initial.to(device), t_initial.to(device), u_initial.to(device)
x_collocation, t_collocation = x_collocation.to(device), t_collocation.to(device)

# 讓搭配點的張量可以計算梯度
x_collocation.requires_grad = True
t_collocation.requires_grad = True

# 3. 實例化模型和優化器
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. 在訓練迴圈中定義損失函數
def loss_fn(model, x_init, t_init, u_init, x_coll, t_coll):
    # 數據損失 (在 t=0 時)
    u_pred_initial = model(x_init, t_init)
    loss_data = torch.mean((u_pred_initial - u_init)**2)

    # 物理損失 (在搭配點上)
    u = model(x_coll, t_coll)
    
    # 使用 torch.autograd.grad 計算各階導數
    # create_graph=True 允許計算高階導數
    u_t = torch.autograd.grad(u, t_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_coll, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x_coll, grad_outputs=torch.ones_like(u_xx), create_graph=True)[0]
    
    # 計算 KdV 方程式的殘差: u_t + 6*u*u_x + u_xxx
    pde_residual = u_t + 6 * u * u_x + u_xxx
    loss_physics = torch.mean(pde_residual**2)
    
    # 總損失是兩者之和，可以加權重
    total_loss = loss_data + 0.1 * loss_physics # 給物理損失一個權重
    return total_loss, loss_data, loss_physics

# 5. 訓練迴圈
epochs = 15000
pbar = tqdm(range(epochs), desc="Training PINN for KdV")

for epoch in pbar:
    optimizer.zero_grad()
    
    loss, loss_d, loss_p = loss_fn(model, x_initial, t_initial, u_initial, x_collocation, t_collocation)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        pbar.set_postfix({'Total Loss': f'{loss.item():.4e}', 'Data Loss': f'{loss_d.item():.4e}', 'Physics Loss': f'{loss_p.item():.4e}'})

print("Training finished.")

# 6. 視覺化結果
model.eval()
with torch.no_grad():
    # 建立一個用於繪圖的時空網格
    x_plot = np.linspace(-20, 20, 400)
    t_plot = np.linspace(0, 2, 200)
    X, T = np.meshgrid(x_plot, t_plot)
    
    # 將網格點轉換為PyTorch張量
    x_tensor = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    t_tensor = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    
    # 使用訓練好的模型進行預測
    u_pred = model(x_tensor, t_tensor)
    
    # 將預測結果轉回Numpy陣列並重塑為網格形狀
    U_pred = u_pred.cpu().numpy().reshape(T.shape)

# 繪製時空熱圖
plt.figure(figsize=(10, 6))
plt.pcolormesh(T, X, U_pred.T, shading='auto', cmap='viridis')
plt.colorbar(label='Amplitude u(x,t)')
plt.xlabel('Time (t)')
plt.ylabel('Space (x)')
plt.title('PINN Solution of KdV Equation: Soliton Propagation')
plt.show()

# 繪製不同時間點的波形對比
plt.figure(figsize=(12, 7))
t_slices = [0, 1, 2] # 選擇三個時間點
for i, t_val in enumerate(t_slices):
    # 解析解
    u_exact = 0.5 * c * (1 / np.cosh(0.5 * np.sqrt(c) * (x_plot - c * t_val)))**2
    
    # PINN 解
    t_slice_tensor = torch.full_like(torch.tensor(x_plot).view(-1, 1), t_val).to(device)
    x_slice_tensor = torch.tensor(x_plot).view(-1, 1).float().to(device)
    u_pinn_slice = model(x_slice_tensor, t_slice_tensor).cpu().numpy()
    
    plt.subplot(len(t_slices), 1, i + 1)
    plt.plot(x_plot, u_exact, 'b-', label='Exact Solution')
    plt.plot(x_plot, u_pinn_slice, 'r--', label='PINN Solution')
    plt.ylabel(f'u(x, t={t_val})')
    plt.ylim(0, c + 1)
    plt.grid(True)
    plt.legend()

plt.xlabel('Space (x)')
plt.suptitle('Comparison of PINN and Exact Solutions at Different Times')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()