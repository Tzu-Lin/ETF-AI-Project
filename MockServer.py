from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
class MockVulnerableServer(BaseHTTPRequestHandler):
    def do_get(self):
        # 解析 URL 參數
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        # 模擬 SQL 注入漏洞的邏輯
        # 如果參數 id 裡面包含單引號 '，就回傳錯誤訊息（長度會改變）
        user_id = params.get('id', [''])[0]
        if "'" in user_id:
            response_content = "<html><body><h1>SQL Syntax Error!</h1><p>You have an error in your SQL syntax near '''</p></body></html>"
        else:
            response_content = f"<html><body><h1>User Profile</h1><p>User ID: {user_id}</p><p>Name: John Doe</p></body></html>" 
        self.wfile.write(response_content.encode("utf-8"))
    # 抑制輸出 log 到終端機，讓畫面乾淨
    def log_message(self, format, *args): return
if __name__ == "__main__":
    print("[*] 虛擬靶機已啟動於 http://127.0.0.1:8000")
    print("[*] 模擬漏洞：SQL Injection (參數: id)")
    HTTPServer(("127.0.0.1", 8000), MockVulnerableServer).serve_forever()