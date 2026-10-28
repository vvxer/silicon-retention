# -*- coding: utf-8 -*-
r"""融合引擎：DeepSeek 流式输出当草稿，本地"专家"（开 LoRA）和"反专家"（关 LoRA）成块验证，按投机采样接受或改写。

原理和实验见 docs\原理与实验.md。位置 j 上（S = DeepSeek 前 20 个候选里可用的）：
  pA(c) = API 概率在 S 上归一化；Δ(c) = log p专家(c 的首个本地 token) − log p反专家(同)，截断到 [-clip, clip]
  pF(c) ∝ pA(c)·exp(α·Δ(c))；API 采到 x：以 min(1, pF(x)/pA(x)) 接受，否则从 max(0, pF − pA) 重采样并从那里重新请求。

工程上的处理：
  提速  备用连接免握手（deepseek.Client）；发出请求后、等首个 token 时先把本地缓存推进到新前缀；
        重启后第一块只攒 first_block 个 token；每次请求上限 stream_cap=64
  少改写  任务类请求（写、列、代码、方案……）用 alpha_task，闲聊用 alpha_chat；TV < 门槛的位置直接接受（速度档，见 SPEEDS）
  篇幅  句末候选不加偏移：人格只改"怎么说"，不改"说多长"（长对话里回答越来越短就是句末被不断加分造成的）
  on_text  确认过的文字立刻回调（投机采样里确认过的不会再改），界面可以边生成边显示
"""
import copy
import random
import re
import threading
import time
from itertools import islice

import torch

EOS = object()
LOCK = threading.Lock()     # PEFT 的 disable_adapter 会改全局状态，快速分词器并发也会冲突：本地前向和切词都在锁里
TASK = re.compile(r"写|帮我|帮忙|列(个|一下|出)|整理|总结|翻译|代码|程序|脚本|方案|周报|日报|报告|提纲|大纲|邮件|解释|分析|"
                  r"计划|清单|改一下|改写|润色|生成|设计|优化|对比|步骤|教程|怎么(做|写|弄|配|装)")
# 速度档 → (起始门槛 min_tv, 每改写一次门槛提高 tv_step, 门槛上限 max_tv)。TV 低于门槛的位置不做接受检验，直接用 DeepSeek 的 token。
# strong = 严格投机采样，输出严格服从融合分布（研究阶段的引擎）；fast = 初版的提速设置；medium 取两者中间
SPEEDS = {"strong": (0.0, 0.0, 0.0), "medium": (0.1, 0.025, 0.3), "fast": (0.2, 0.05, 0.6)}
SPEED_ALIASES = {"像她": "strong", "均衡": "medium", "快": "fast"}


def speed_params(speed: str) -> dict:
    s = SPEED_ALIASES.get(speed, speed)
    if s not in SPEEDS:
        raise ValueError(f"speed 只能是 {' / '.join(SPEEDS)}（或 {' / '.join(SPEED_ALIASES)}），收到 {speed!r}")
    return dict(zip(("min_tv", "tv_step", "max_tv"), SPEEDS[s]))


def is_task(msg: str) -> bool:
    return bool(TASK.search(msg))


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def complete_utf8(b: bytes) -> str:
    """最长的完整 UTF-8 前缀。"""
    for cut in range(len(b), max(-1, len(b) - 4), -1):
        try:
            return b[:cut].decode("utf-8")
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "ignore")


def chat_prefix(tok, user_msg: str, history: list | None = None, max_tokens: int = 1500) -> list:
    """本地模型的对话前缀（不带系统提示，和巩固训练时一致）；历史超过 max_tokens 就从最早的一问一答开始丢。"""
    hist = list(history or [])
    while True:
        msgs = hist + [{"role": "user", "content": user_msg}]
        ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                  add_special_tokens=False).input_ids
        if len(ids) <= max_tokens or len(hist) < 2:
            return ids
        hist = hist[2:]


class Side:
    """一侧本地模型（专家或反专家）：固定前缀 + 跟着已接受文本推进的缓存。"""

    def __init__(self, model, tok, lora: bool):
        self.model, self.tok, self.lora = model, tok, lora
        self.recomputes, self.fwd_s = 0, 0.0

    def _fwd(self, ids, cache, keep: int | None = None):
        t = time.perf_counter()
        kw = dict(input_ids=torch.tensor([ids], device="cuda"), past_key_values=cache, use_cache=True,
                  logits_to_keep=keep or len(ids))
        if self.lora:
            out = self.model(**kw)
        else:
            with self.model.disable_adapter():
                out = self.model(**kw)
        lp = torch.log_softmax(out.logits[0].float(), -1)
        self.fwd_s += time.perf_counter() - t
        return out.past_key_values, lp

    def reset(self, prefix_ids: list):
        self.prefix = prefix_ids
        self.cache, lp = self._fwd(prefix_ids, None, keep=1)   # 长前缀只要最后一个位置的分布，否则 n×24.8 万的 logits 撑爆显存
        self.ids, self.last, self.snap = [], lp[-1], None

    def advance(self, text: str) -> dict:
        """让缓存覆盖 text 的本地切词，返回 {字符边界: 该边界处下一 token 的 log 概率行}（本次新处理区域内的干净边界）。"""
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc.input_ids, enc.offset_mapping
        k = lcp(self.ids, ids)
        if k < len(self.ids):                       # 已处理的部分被改了（拒绝重采样或切词合并）→ 回到快照，不行就从头算
            if self.snap is not None and k >= len(self.snap[1]):
                cache, sids, last = self.snap
                self.cache, self.ids, self.last = copy.deepcopy(cache), list(sids), last
            else:
                self.recomputes += 1
                self.reset(self.prefix)
        start = len(self.ids)
        self.snap = (copy.deepcopy(self.cache), list(self.ids), self.last)
        rows = {offs[start - 1][1] if start else 0: self.last}
        new = ids[start:]
        if new:
            self.cache, lp = self._fwd(new, self.cache)
            for i, m in enumerate(range(start, len(ids))):
                end = offs[m][1]
                if m + 1 == len(ids) or offs[m + 1][0] >= end:
                    rows[end] = lp[i]
            self.ids, self.last = ids, lp[-1]
        return rows


class Fuser:
    def __init__(self, model, tok, client, alpha_chat=1.0, alpha_task=0.5, clip=5.0, min_tv=0.2, tv_step=0.05, max_tv=0.6,
                 block=8, first_block=4, stream_cap=64, max_tokens=800, seed=None):
        self.tok, self.client = tok, client
        self.alpha_chat, self.alpha_task, self.clip = alpha_chat, alpha_task, clip
        # 前紧后松：每改写一次，"TV 小于门槛就直接接受"的门槛提高 tv_step（上限 max_tv）。
        # 每次改写都要重新请求，DeepSeek 吐第一个 token 要约 0.5 秒（服务器侧，省不掉）；人格主要体现在开头，
        # 开头定了调，DeepSeek 后面会顺着这个语气写，所以把改写集中在前面
        self.min_tv, self.tv_step, self.max_tv = min_tv, tv_step, max_tv
        self.block, self.first_block, self.stream_cap, self.max_tokens = block, first_block, stream_cap, max_tokens
        self.expert, self.anti = Side(model, tok, lora=True), Side(model, tok, lora=False)
        self.rng = random.Random(seed)
        self._cand = {}

    def first_token(self, ctx: str, cand: str):
        """候选在本地接在前文后面时的首个 token；和前文粘连就返回 None。"""
        w = ctx[-24:]
        key = (w, cand)
        if key not in self._cand:
            a = self.tok(w, add_special_tokens=False).input_ids
            b = self.tok(w + cand, add_special_tokens=False).input_ids
            k = lcp(a, b)
            self._cand[key] = b[k] if k == len(a) and len(b) > k else None
        return self._cand[key]

    def fuse(self, item: dict, ctx: str, row_e, row_a, alpha: float, min_tv: float):
        cands, api_lp, ltok = [], [], []
        for t in item.get("top_logprobs") or []:
            if t["logprob"] <= -9000:
                continue
            if "end▁of▁sentence" in t["token"]:
                cands.append(EOS); ltok.append(None)          # 句末不加偏移
            elif t["token"].startswith("<｜"):
                continue
            else:
                try:
                    s = bytes(t.get("bytes") or t["token"].encode()).decode("utf-8")
                except UnicodeDecodeError:
                    continue
                cands.append(s); ltok.append(self.first_token(ctx, s))
            api_lp.append(t["logprob"])
        try:
            x = bytes(item.get("bytes") or item["token"].encode()).decode("utf-8")
        except UnicodeDecodeError:
            return True, None, None
        if len(cands) < 2 or x not in cands:
            return True, None, None
        la = torch.tensor(api_lp, dtype=torch.float64)
        pa = torch.softmax(la, 0)
        idx = [i for i, t in enumerate(ltok) if t is not None]
        delta = torch.zeros(len(cands), dtype=torch.float64)
        if idx:
            tt = torch.tensor([ltok[i] for i in idx], device=row_e.device)
            delta[idx] = (row_e[tt] - row_a[tt]).double().cpu().clamp(-self.clip, self.clip)
        pf = torch.softmax(la + alpha * delta, 0)
        tv = float(0.5 * (pf - pa).abs().sum())
        if tv < min_tv:
            return True, None, {"tv": tv, "lenient": True}
        ix = cands.index(x)
        if self.rng.random() < min(1.0, float(pf[ix] / pa[ix])):
            return True, None, {"tv": tv}
        r = self.rng.choices(range(len(cands)), weights=(pf - pa).clamp(min=0).tolist())[0]
        return False, cands[r], {"tv": tv}

    def generate(self, user_msg: str, api_system: str | None = None, history: list | None = None, on_text=None,
                 api_messages: list | None = None):
        """DeepSeek 看到 api_system + 历史 + 本条消息（给了 api_messages 就只看它，最后一条应是本条消息）；
        本地模型只看历史（截断）+ 本条消息。返回 (回答, 统计)。"""
        t0 = time.perf_counter()
        task = is_task(user_msg)
        alpha = self.alpha_task if task else self.alpha_chat
        with LOCK:
            self.expert.reset(chat_prefix(self.tok, user_msg, history))
            self.anti.reset(chat_prefix(self.tok, user_msg, history))
        for s in (self.expert, self.anti):
            s.recomputes, s.fwd_s = 0, 0.0
        st = {"task": task, "alpha": alpha, "api_tokens": 0, "streams": 0, "rejections": 0, "fused": 0, "lenient": 0,
              "skipped": 0, "sum_tv": 0.0, "first_text_s": None, "ttft_s": 0.0, "wait_s": 0.0, "verify_s": 0.0, "pre_s": 0.0}
        base = list(api_messages) if api_messages else \
            ([{"role": "system", "content": api_system}] if api_system else []) + list(history or []) + [{"role": "user", "content": user_msg}]
        done, sent = b"", 0

        def emit():
            nonlocal sent
            b = complete_utf8(done).encode("utf-8")
            if len(b) > sent:
                if st["first_text_s"] is None:
                    st["first_text_s"] = round(time.perf_counter() - t0, 3)
                if on_text:
                    on_text(b[sent:].decode("utf-8"))
                sent = len(b)

        while st["api_tokens"] < self.max_tokens:
            prefix = complete_utf8(done)
            done = prefix.encode("utf-8")
            msgs = base + ([{"role": "assistant", "content": prefix, "prefix": True}] if prefix else [])
            t_req = time.perf_counter()
            stream = self.client.stream(msgs, max_tokens=min(self.stream_cap, self.max_tokens - st["api_tokens"]))
            st["streams"] += 1
            if prefix:                      # 请求已发出：趁等首个 token，先把本地缓存推进到新前缀
                t = time.perf_counter()
                with LOCK:
                    self.expert.advance(prefix)
                    self.anti.advance(prefix)
                st["pre_s"] += time.perf_counter() - t
            it, restart, k, first = iter(stream), False, self.first_block, True
            while not restart:
                t = time.perf_counter()
                blk = list(islice(it, k))
                st["wait_s"] += time.perf_counter() - t
                if first:
                    st["ttft_s"] += time.perf_counter() - t_req
                    first = False
                k = self.block
                if not blk:
                    break
                t = time.perf_counter()
                with LOCK:
                    done, restart, finished = self._verify_block(blk, done, st, alpha)
                st["verify_s"] += time.perf_counter() - t
                emit()
                if finished or restart:
                    stream.close()
                if finished:
                    return self._finish(done, st, t0)
            if not restart and stream.finish != "length":
                break
        return self._finish(done, st, t0)

    def _verify_block(self, blk: list, done: bytes, st: dict, alpha: float):
        ahead = done + b"".join(bytes(t.get("bytes") or t["token"].encode()) for t in blk)
        local_text = complete_utf8(ahead)
        rows_e, rows_a = self.expert.advance(local_text), self.anti.advance(local_text)
        pos = done
        for t in blk:
            st["api_tokens"] += 1
            ctx = None
            try:
                ctx = pos.decode("utf-8")
            except UnicodeDecodeError:
                pass
            b = len(ctx) if ctx is not None else -1
            min_tv = min(self.max_tv, self.min_tv + self.tv_step * st["rejections"])
            ok, rep, s = (self.fuse(t, ctx, rows_e[b], rows_a[b], alpha, min_tv) if ctx is not None and b in rows_e and b in rows_a
                          else (True, None, None))
            if s is None:
                st["skipped"] += 1
            else:
                st["fused"] += 1
                st["sum_tv"] += s["tv"]
                st["lenient"] += bool(s.get("lenient"))
            if ok:
                pos += bytes(t.get("bytes") or t["token"].encode())
                continue
            st["rejections"] += 1
            if rep is EOS:
                return pos, True, True
            return pos + rep.encode("utf-8"), True, False
        return pos, False, False

    def _finish(self, done: bytes, st: dict, t0: float):
        for k in ("sum_tv", "ttft_s", "wait_s", "verify_s", "pre_s"):
            st[k] = round(st[k], 2)
        st["recomputes"] = self.expert.recomputes + self.anti.recomputes
        st["local_s"] = round(self.expert.fwd_s + self.anti.fwd_s, 2)
        st["wall_s"] = round(time.perf_counter() - t0, 2)
        return done.decode("utf-8", "replace"), st
