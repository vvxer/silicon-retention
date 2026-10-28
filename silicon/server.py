# -*- coding: utf-8 -*-
r"""本地聊天服务（只用标准库）：python -m silicon.server [--no-lora] [--port 8700]，浏览器打开 http://127.0.0.1:8700

POST /api/chat  {"message": "...", "session": "会话 id", "speed": "fast|medium|strong（可省）"}  → NDJSON 流：
    {"type": "delta", "text": "..."}   确认过的文字（不会再改），边生成边显示
    {"type": "done", "stats": {...}}
    {"type": "error", "message": "..."}
GET  /api/status                                   → 当前 LoRA、记忆天数、名字、默认速度档
默认只监听 127.0.0.1；对话内容会发给 DeepSeek，原文日志和 LoRA 都留在本机 data\ 下。
服务开着时，每天 nightly_at（默认 03:30）自动夜间整理：更新概况和相处习惯卡；用本地权重时还要训练，
先把本地模型移出显存（8GB 显卡放不下聊天和训练两份），这段时间回答只用 DeepSeek，训练完加载新的 LoRA。
"""
import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, fusion
from .chat import Companion, window

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "index.html"
SESSIONS: dict = {}          # 会话 id → [{"at", "user", "assistant"}]
COMP: Companion | None = None


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            data = WEB.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/api/status":
            cfg = COMP.cfg
            self._json(200, {"lora": COMP.adapter, "memory_days": len(COMP.store.days()), "ai_name": cfg.ai_name,
                             "user_name": cfg.user_name, "speed": cfg.speed, "speeds": list(fusion.SPEEDS)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/chat":
            return self._json(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        msg = (body.get("message") or "").strip()
        if not msg:
            return self._json(400, {"error": "empty message"})
        turns = SESSIONS.setdefault(body.get("session") or "default", [])
        speed = body.get("speed") or None
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        q: queue.Queue = queue.Queue()

        def work():
            try:
                now = datetime.now()
                hist, hide_from = window(turns, COMP.cfg.history_chars)
                # 新会话（hide_from 为空）：今天别的会话里的话都不在上下文里，全部可检索
                text, st, user = COMP.reply(msg, hist, on_text=lambda t: q.put({"type": "delta", "text": t}), now=now,
                                            speed=speed, hide_from=hide_from or datetime.max)
                turns.append({"at": now, "user": user, "assistant": text})
                q.put({"type": "done", "stats": st})
            except Exception as e:  # 网络、显存等错误都回给页面
                q.put({"type": "error", "message": repr(e)})
            q.put(None)

        threading.Thread(target=work, daemon=True).start()
        while (item := q.get()) is not None:
            try:
                self.wfile.write((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            except OSError:          # 页面关了；生成照样跑完并存档
                pass

    def log_message(self, *args):
        pass


def nightly_loop(at: str, lora: bool):
    """每天 at（HH:MM）跑一次夜间整理（单独进程）；用本地权重时前后把本地模型移出、再加载回显存。"""
    h, m = map(int, at.split(":"))
    while True:
        now = datetime.now()
        due = now.replace(hour=h, minute=m, second=0, microsecond=0)
        time.sleep(((due if due > now else due + timedelta(days=1)) - now).total_seconds())
        print(f"[{datetime.now():%m-%d %H:%M}] 夜间整理开始" + ("，这段时间回答只用 DeepSeek" if lora else ""), flush=True)
        if lora:
            COMP.unload_model()
        code = None
        try:
            code = subprocess.run([sys.executable, "-m", "silicon.nightly"] + ([] if lora else ["--skeleton-only"]), cwd=ROOT).returncode
        except Exception as e:  # 整理失败不能让服务停掉：旧的 LoRA 照常加载回来
            print("夜间整理出错：", repr(e), flush=True)
        finally:
            if lora:
                try:
                    COMP.load_model()
                except Exception as e:
                    print("本地模型加载失败，先只用 DeepSeek：", repr(e), flush=True)
        print(f"[{datetime.now():%m-%d %H:%M}] 夜间整理结束（返回码 {code}）" + (f"，LoRA：{COMP.adapter}" if lora else ""), flush=True)


def main():
    global COMP
    ap = argparse.ArgumentParser(description="硅基保留：本地聊天服务")
    ap.add_argument("--no-lora", action="store_true", help="不用本地 LoRA，只用 DeepSeek（对比或没显卡时用）")
    ap.add_argument("--port", type=int)
    a = ap.parse_args()
    cfg = config.load()
    COMP = Companion(cfg, use_lora=not a.no_lora)
    port = a.port or cfg.port
    print(f"LoRA：{COMP.adapter or '未启用'}；记忆 {len(COMP.store.days())} 天。打开 http://{cfg.host}:{port}", flush=True)
    if cfg.nightly_at:
        threading.Thread(target=nightly_loop, args=(cfg.nightly_at, not a.no_lora), daemon=True).start()
        print(f"每天 {cfg.nightly_at} 自动夜间整理（data/config.json 里的 nightly_at 改时间，留空关掉）", flush=True)
    ThreadingHTTPServer((cfg.host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
