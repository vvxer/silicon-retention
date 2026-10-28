# -*- coding: utf-8 -*-
"""DeepSeek API 客户端（只用标准库，直连不走代理）。

- 始终在后台备好一条已握手的 HTTPS 连接：融合时经常中途断开流、马上发新请求，新请求直接用备用连接，省掉握手。
- stream() 先把请求发出去就返回，调用方可以趁着等首个 token 去做本地计算，再开始读流。
- key 从环境变量 DEEPSEEK_API_KEY 读；不要写进任何文件。
"""
import http.client
import json
import os
import ssl
import threading
import time
from urllib.parse import urlparse

RETRY_CODES = (429, 500, 502, 503, 504)
SPARE_TTL = 8.0          # 备用连接闲置太久会被服务器断开（实测十几秒），超过就换新的
DROPPED = (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError, ConnectionAbortedError)


class Client:
    def __init__(self, key: str | None = None, base_url: str | None = None, model: str | None = None):
        u = urlparse(base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
        self.host, self.root = u.hostname, u.path.rstrip("/")
        self.key = key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.key:
            raise SystemExit("没有找到 DeepSeek API key：请先设置环境变量 DEEPSEEK_API_KEY（不要写进任何文件）")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
        self._ctx = ssl.create_default_context()
        self._spare = None
        self._lock = threading.Lock()
        self._warm()

    # ---- 连接 ----
    def _new(self):
        c = http.client.HTTPSConnection(self.host, timeout=600, context=self._ctx)
        c.connect()
        return c

    def _warm(self):
        def work():
            try:
                c = self._new()
            except OSError:
                return
            with self._lock:
                if self._spare is None:
                    self._spare, c = (c, time.monotonic()), None
            if c is not None:
                c.close()
        threading.Thread(target=work, daemon=True).start()

    def _take(self):
        with self._lock:
            spare, self._spare = self._spare, None
        self._warm()
        if spare is not None and time.monotonic() - spare[1] < SPARE_TTL:
            return spare[0]
        if spare is not None:
            spare[0].close()
        return self._new()

    def _send(self, path: str, body: dict):
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"}
        for i in range(5):
            conn = None
            try:
                conn = self._take()          # 建连接也可能失败（DNS、网络抖动），和发送失败一样重试
                conn.request("POST", self.root + path, body=data, headers=headers)
                return conn
            except OSError:
                if conn is not None:
                    conn.close()
                if i == 4:
                    raise
                time.sleep(2 ** i)

    # ---- 调用 ----
    def chat(self, messages: list, model: str | None = None, **kw) -> dict:
        """普通调用，返回完整响应 JSON。默认关闭思考模式。"""
        body = {"model": model or self.model, "messages": messages, "thinking": {"type": "disabled"}, **kw}
        for i in range(5):
            conn = self._send("/chat/completions", body)
            try:
                r = conn.getresponse()
                raw = r.read()
            except DROPPED:
                if i == 4:
                    raise
                continue
            finally:
                conn.close()
            if r.status == 200:
                return json.loads(raw)
            if r.status not in RETRY_CODES or i == 4:
                raise RuntimeError(f"HTTP {r.status}: {raw[:300]!r}")
            time.sleep(2 ** i)

    def stream(self, messages: list, beta: bool = True, **kw) -> "Stream":
        """发出流式请求后立即返回（还没读响应）。逐个产出 logprobs.content 里的 token 项。"""
        body = {"model": self.model, "messages": messages, "stream": True, "thinking": {"type": "disabled"},
                "logprobs": True, "top_logprobs": 20, "temperature": 1.0, **kw}
        if not body["logprobs"]:
            body.pop("top_logprobs", None)        # 不要 logprobs 时不能带 top_logprobs，否则 400
        path = ("/beta" if beta else "") + "/chat/completions"
        return Stream(self, path, body, self._send(path, body))


class Stream:
    def __init__(self, client, path, body, conn):
        self.client, self.path, self.body, self.conn, self.finish = client, path, body, conn, None

    def _response(self):
        """拿响应头；连接在这之前被服务器断开（还没收到任何数据）就换新连接重发。"""
        for i in range(4):
            try:
                r = self.conn.getresponse()
            except DROPPED:
                self.conn.close()
                if i == 3:
                    raise
                self.conn = self.client._send(self.path, self.body)
                continue
            if r.status in RETRY_CODES and i < 3:
                r.read()
                self.conn.close()
                time.sleep(2 ** i)
                self.conn = self.client._send(self.path, self.body)
                continue
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]!r}")
            return r

    def __iter__(self):
        try:
            r = self._response()
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                for c in json.loads(payload).get("choices") or []:
                    items = (c.get("logprobs") or {}).get("content")
                    if items:
                        yield from items
                    elif (c.get("delta") or {}).get("content"):      # 没要 logprobs 时，把文字增量包成同样的形状
                        text = c["delta"]["content"]
                        yield {"token": text, "bytes": list(text.encode("utf-8"))}
                    if c.get("finish_reason"):
                        self.finish = c["finish_reason"]
        finally:
            self.close()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
