import socket

# 原始、簡單的連線代碼
target_ip = "1.1.1.1"
target_port = 4444

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((target_ip, target_port))
print("Connection established.")