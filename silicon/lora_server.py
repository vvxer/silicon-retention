# -*- coding: utf-8 -*-
r"""本地权重服务，给 DSH 插件 dsh-silicon 用：python -m silicon.lora_server [--port 8701]
插件开了 useLora 后，把闲聊转到这里：DeepSeek 写内容，本地 2B + LoRA 按她的说话方式改写（和 silicon.server --lora 是同一个融合引擎）。
记忆和原文日志都由插件负责，这里只做融合：不检索、不记日志。数据目录要和插件的 dataDir 相同（环境变量 SILICON_DATA_DIR），
先运行 python -m silicon.bake 烘焙人格（人格卡写在数据目录的 persona.md）。

POST /v1/fuse  {"messages": [...], "local": [...], "speed": "fast|medium|strong（可省）"}  → NDJSON 流，事件格式同 /api/chat
    messages：发给 DeepSeek 的完整消息（插件已经加好记忆），最后一条是这次的用户消息
    local：本地模型看的对话（用户和她的原话，不带系统提示和记忆），最后一条是这次的用户消息
    503：本地模型不在显存里（还没烘焙，或者正在夜间巩固），插件收到就改用普通 DeepSeek
GET  /api/status → {"lora": 当前 LoRA 或 null}
服务开着时，每天 nightly_at 只巩固权重（silicon.nightly --weights-only）；概况和相处习惯卡由插件自己整理。
"""
import argparse
import json
import threading
from datetime import datetime
from http.server import ThreadingHTTPServer

from . import config, deepseek, fusion
from .chat import LocalModel
from .server import Handler as ChatHandler, nightly_loop

LM: LocalModel | None = None
CLIENT: deepseek.Client | None = None


class Handler(ChatHandler):
    def do_GET(self):
        if self.path == "/api/status":
            return self._json(200, {"lora": LM.adapter})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/fuse":
            return self._json(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        msgs, local = body.get("messages") or [], body.get("local") or []
        if not msgs or not local or local[-1].get("role") != "user":
            return self._json(400, {"error": "messages 和 local 都不能为空，local 的最后一条要是用户消息"})
        if LM.model is None:
            return self._json(503, {"error": "本地权重没有加载（还没烘焙，或者正在夜间巩固）"})
        cfg = LM.cfg
        speed = body.get("speed") or cfg.speed

        def work(put):
            import torch
            with LM.lock:
                if LM.model is None:            # 排队时被夜间巩固移出了显存
                    return put({"type": "error", "message": "本地权重刚被移出显存（夜间巩固）"})
                fz = fusion.Fuser(LM.model, LM.tok, CLIENT, alpha_chat=cfg.alpha_chat, alpha_task=cfg.alpha_task,
                                  max_tokens=cfg.max_reply_tokens, **fusion.speed_params(speed))
                with torch.inference_mode():
                    text, st = fz.generate(local[-1]["content"], history=local[:-1], api_messages=msgs,
                                           on_text=lambda t: put({"type": "delta", "text": t}))
            st["speed"] = speed
            print(f"[{datetime.now():%m-%d %H:%M}] 融合一条：{st['wall_s']} 秒，改写 {st['rejections']} 次（{speed}）", flush=True)
            put({"type": "done", "stats": st})

        self._ndjson(work)


def main():
    global LM, CLIENT
    ap = argparse.ArgumentParser(description="硅基保留：本地权重服务（给 DSH 插件 dsh-silicon 用）")
    ap.add_argument("--port", type=int)
    a = ap.parse_args()
    cfg = config.load()
    fusion.speed_params(cfg.speed)                  # 配置写错了尽早报出来
    CLIENT = deepseek.Client()
    LM = LocalModel(cfg)
    LM.load_model()
    port = a.port or cfg.lora_port
    if LM.adapter:
        print(f"LoRA：{LM.adapter}。本地权重服务在 http://{cfg.host}:{port}，数据目录 {cfg.data}", flush=True)
    else:
        print(f"还没有本地权重：先把人格卡写进 {cfg.data / 'persona.md'}，运行 python -m silicon.bake，再重启本服务。"
              "在那之前插件只用 DeepSeek。", flush=True)
    if cfg.nightly_at:
        threading.Thread(target=nightly_loop, args=(cfg.nightly_at, LM, True, ["--weights-only"]), daemon=True).start()
        print(f"每天 {cfg.nightly_at} 把当天的相处巩固进权重（config.json 里的 nightly_at 改时间，留空关掉）", flush=True)
    ThreadingHTTPServer((cfg.host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
