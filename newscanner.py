import socket

# 這是我們在 docker-compose 裡定義的服務名稱
target_host = "n8n_lab_target" 
# n8n 的預設服務埠
target_port = 5678 

def port_scan(host, port):
    print(f"[*] 正在偵察目標服務: {host} (Port: {port})")
    
    # 建立 Socket 連線
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3) # 設定逾時 3 秒
    
    # 執行滲透掃描
    result = s.connect_ex((host, port))
    
    if result == 0:
        print(f"[+] 滲透成功！目標服務 {host} 正在運行，且連接埠 {port} 已開啟。")
        print(f"[!] 這是潛在的進入點，可進行下一步的漏洞分析或暴力破解。")
    else:
        print(f"[-] 掃描失敗。連接埠 {port} 關閉或目標主機無法連線 (代碼: {result})")
    
    s.close()

if __name__ == "__main__":
    port_scan(target_host, target_port)