# -*- coding: utf-8 -*-
"""对话：拼 DeepSeek 的系统提示（人格卡 + 现在时间 + 概况 + 检索到的原文 + 规则），用融合引擎生成，写入原文日志。"""
import json
import threading
from datetime import datetime
from pathlib import Path

from . import config, deepseek, fusion
from .memory import Store, stamp

RULES = ("回答{user}时，只依据上面的记录和你确定知道的信息；记录里没有的过去的事，就直说不记得，不要编造。\n"
         "你看不到图片、文件、链接和附件：{user}说发了图、文件或数据，而消息里没有具体内容，就直说没收到，请对方描述或贴出来，"
         "不要假装看到了，也不要自己编数字。\n"
         "问到日期、天数，以上面写明的日期为准，不要自己推算。")
DEFAULT_PERSONA = Path(__file__).resolve().parent.parent / "persona.example.md"


def state_file(cfg) -> Path:
    return cfg.data / "state.json"


def read_state(cfg) -> dict:
    f = state_file(cfg)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def write_state(cfg, st: dict):
    state_file(cfg).write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def persona_text(cfg) -> str:
    f = cfg.data / "persona.md"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(DEFAULT_PERSONA.read_text(encoding="utf-8"), encoding="utf-8")
    return f.read_text(encoding="utf-8").strip()


def window(turns: list, max_chars: int):
    """本次会话的历史：从最近往前取，总字数不超过 max_chars。turns = [{"at", "user", "assistant"}]（user 已带时间戳）。
    返回 (给 DeepSeek 的历史消息, hide_from)：hide_from 之后的对话已在历史里，检索时不再重复取；更早的（含被截掉的）照常可检索。"""
    kept, used = [], 0
    for t in reversed(turns):
        n = len(t["user"]) + len(t["assistant"])
        if kept and used + n > max_chars:
            break
        kept.insert(0, t)
        used += n
    hist = [m for t in kept for m in ({"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["assistant"]})]
    return hist, (kept[0]["at"] if kept else None)


class Companion:
    def __init__(self, cfg: config.Config | None = None, use_lora: bool = True):
        self.cfg = cfg or config.load()
        config.save_default(self.cfg)
        fusion.speed_params(self.cfg.speed)                    # 配置写错了尽早报出来
        self.persona = persona_text(self.cfg)
        self.store = Store(self.cfg.data, self.cfg.embed_model, self.cfg.user_name, self.cfg.ai_name)
        self.client = deepseek.Client()
        self.lock = threading.Lock()          # 一次只生成一条
        self.model = self.tok = self.adapter = None
        if use_lora:
            self.load_model()

    def load_model(self):
        """按 state.json 里的 adapter 加载本地模型；还没有 adapter 就保持只用 DeepSeek。"""
        adapter = read_state(self.cfg).get("adapter")
        if not (adapter and Path(adapter, "adapter_config.json").exists()):
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, self.cfg.vram_gb / total))
        tok = AutoTokenizer.from_pretrained(self.cfg.local_model)
        base = AutoModelForCausalLM.from_pretrained(self.cfg.local_model, dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(base, adapter).eval()
        with self.lock:
            self.tok, self.model, self.adapter = tok, model, adapter

    def unload_model(self):
        """把本地模型移出显存（夜间巩固要用显存）；之后的回答只用 DeepSeek，直到 load_model()。"""
        with self.lock:
            self.model = self.tok = self.adapter = None
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    def system_prompt(self, msg: str, now: datetime, hide_from: datetime | None = None) -> str:
        # 一天之内不变的放前面（人格卡、习惯卡、概况），命中 DeepSeek 的前缀缓存；时间和检索原文每轮都变，放后面
        user = self.cfg.user_name
        card = self.store.habits()
        card = f"你和{user}的相处习惯（从过去的聊天里整理）：\n{card}\n\n" if card else ""
        skel = self.store.skeleton()
        skel = f"过去的概况（近期按周，更早按月、按年）：\n{skel}\n\n" if skel else ""
        when = f"现在是 {now:%Y-%m-%d} 星期{'一二三四五六日'[now.weekday()]} {now:%H:%M}。"
        days = self.store.days()
        if days:
            first = datetime.strptime(days[0], "%Y-%m-%d").date()
            when += f"你和{user}的第一条聊天记录是 {first}，到今天是认识的第 {(now.date() - first).days + 1} 天。"
        snip = self.store.retrieve(msg, now, self.cfg.retrieval_chars, hide_from)
        mem = f"以下是和{user}这句话可能相关的过往聊天记录（原文，按时间排序）：\n{snip}\n\n" if snip else ""
        return f"{self.persona}\n\n" + card + skel + when + "\n\n" + mem + RULES.format(user=user)

    def reply(self, msg: str, history: list, on_text=None, now: datetime | None = None, save: bool = True,
              speed: str | None = None, hide_from: datetime | None = None):
        """history：本次会话之前的消息（用户消息已带时间戳，可用 window() 截好）；hide_from 见 window()，
        不给就把今天的对话都当作已在上下文里。speed 不给就用配置里的速度档。返回 (回答, 统计, 带时间戳的用户消息)。"""
        now = now or datetime.now()
        user = f"{stamp(now)} {msg}"
        with self.lock:
            system = self.system_prompt(msg, now, hide_from)
            if self.model is None:
                text, st = self._plain(system, history, user, on_text)
            else:
                import torch
                fz = fusion.Fuser(self.model, self.tok, self.client, alpha_chat=self.cfg.alpha_chat, alpha_task=self.cfg.alpha_task,
                                  max_tokens=self.cfg.max_reply_tokens, **fusion.speed_params(speed or self.cfg.speed))
                with torch.inference_mode():
                    text, st = fz.generate(user, api_system=system, history=history, on_text=on_text)
                st["speed"] = speed or self.cfg.speed
        if save:
            self.store.append("user", msg, now)
            self.store.append("assistant", text)
        return text, st, user

    def _plain(self, system, history, user, on_text):
        """还没有 LoRA（或关掉了）时：普通流式调用。"""
        out, n = [], 0
        stream = self.client.stream([{"role": "system", "content": system}] + history + [{"role": "user", "content": user}],
                                    beta=False, max_tokens=self.cfg.max_reply_tokens, logprobs=False)
        buf = b""
        for t in stream:
            n += 1
            buf += bytes(t.get("bytes") or t["token"].encode())
            text = fusion.complete_utf8(buf)
            new = text[len("".join(out)):]
            if new:
                out.append(new)
                if on_text:
                    on_text(new)
        return "".join(out), {"plain": True, "api_tokens": n}
