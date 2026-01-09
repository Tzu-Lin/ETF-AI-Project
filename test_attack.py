import socket
target = "n8n_lab_target"
port = 5678
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)
result = s.connect_ex((target, port))
if result == 0:
    print(f"[+] 滲透成功！發現目標 {target} 開放了連接埠 {port}")
else:
    print(f"[-] 目錄失敗，連接埠 {port} 未回應")
s.close()