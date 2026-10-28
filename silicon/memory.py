# -*- coding: utf-8 -*-
r"""记忆：原文永久保存（每天一个 JSONL，只追加），按"一问一答"切块做向量检索，规则解析时间说法，夜间生成概况。

研究依据（docs\原理与实验.md）：摘要压缩会丢早期记忆；原文 + 检索 + 每周概况最好；
用户原话里的时间说法可靠，按时间精确过滤。
概况（骨架）：每周一行；总长超过 SKELETON_CAP 时，从最早的开始把一个月的几周换成那个月的一行，
还超就把一年的换成那一年的一行（"一周内完整、一个月记大概、一年记特定点"），每轮提示不随相处时间变长。
"""
import json
import re
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import torch

WEEK = "一二三四五六日"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
SKELETON_CAP = 1200      # 3 个月的每周概况约 1150 字（评测过的规模），之后不再变长
SAVE_EVERY = 64          # 攒够这么多新向量才写一次缓存文件（每轮都写的话，聊久了每轮要写几百 MB）
CN_NUM = {c: i for i, c in enumerate("零一二三四五六七八九十")} | {"两": 2}


def cn2int(s: str) -> int:
    if s.isdigit():
        return int(s)
    if s.startswith("十"):
        return 10 + (CN_NUM.get(s[1:], 0) if len(s) > 1 else 0)
    if "十" in s:
        a, b = s.split("十")
        return CN_NUM[a] * 10 + (CN_NUM.get(b, 0) if b else 0)
    return CN_NUM.get(s, 0)


def parse_when(text: str, now: date):
    """从一句话里找时间说法，返回 (起, 止) 日期闭区间；找不到返回 None。"""
    if not text:
        return None
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        try:
            d = date(int(m[1]), int(m[2]), int(m[3]))
            return (d, d)
        except ValueError:          # 2026-02-30 这类不存在的日期：当作没写，接着看别的说法
            pass
    m = re.search(r"(\d{1,2}|[一二三四五六七八九十]+)月(\d{1,2}|[一二三四五六七八九十]+)[日号]", text)
    if m:
        try:
            d = date(now.year, cn2int(m[1]), cn2int(m[2]))
            return (d, d)
        except ValueError:
            pass
    m = re.search(r"(上个?月|这个?月|本月)(初|上旬|中旬|下旬|底|末)?", text)
    if m:
        first = (now.replace(day=1) - timedelta(1)).replace(day=1) if m[1].startswith("上") else now.replace(day=1)
        last = (first.replace(day=28) + timedelta(4)).replace(day=1) - timedelta(1)
        part = {"初": (1, 10), "上旬": (1, 10), "中旬": (11, 20), "下旬": (21, last.day), "底": (21, last.day),
                "末": (21, last.day)}.get(m[2] or "", (1, last.day))
        return (first.replace(day=part[0]), first.replace(day=part[1]))
    for w, n in (("大前天", 3), ("前天", 2), ("昨天", 1), ("昨晚", 1)):
        if w in text:
            d = now - timedelta(n)
            return (d, d)
    m = re.search(r"(\d+|[一二三四五六七八九十两]+)天前", text)
    if m:
        d = now - timedelta(cn2int(m[1]))
        return (d, d)
    mon = now - timedelta(now.weekday())
    m = re.search(r"(上上|上)(?:个)?(?:周|星期|礼拜)([一二三四五六日天])?", text)
    if m:
        start = mon - timedelta(7 * (2 if m[1] == "上上" else 1))
        if m[2]:
            d = start + timedelta(6 if m[2] in "日天" else "一二三四五六".index(m[2]))
            return (d, d)
        return (start, start + timedelta(6))
    return None


def stamp(t: datetime) -> str:
    return f"[星期{WEEK[t.weekday()]} {t:%H:%M}]"


class Store:
    def __init__(self, data: Path, embed_model: str, user_name: str = "老板", ai_name: str = "小硅"):
        self.dir = data / "logs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.skel_file = data / "memory" / "skeleton.json"
        self.roll_file = data / "memory" / "rollup.json"
        self.emb_file = data / "memory" / "emb.pt"
        self.embed_model, self.user_name, self.ai_name = embed_model, user_name, ai_name
        self.unsaved = 0
        self.lock = threading.Lock()
        self.chunks = []        # {"key", "date", "time", "text", "plain"}
        self.vecs = {}          # key -> 向量
        self._enc = None
        self._load()

    # ---- 原文 ----
    def append(self, role: str, content: str, t: datetime | None = None):
        t = t or datetime.now()
        rec = {"time": t.strftime("%H:%M"), "role": role, "content": content}
        with self.lock, open(self.dir / f"{t:%Y-%m-%d}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._add(t.date(), rec)

    def days(self) -> list:
        return sorted(p.stem for p in self.dir.glob("*.jsonl"))

    def read_day(self, day: str) -> list:
        f = self.dir / f"{day}.jsonl"
        return [json.loads(l) for l in open(f, encoding="utf-8")] if f.exists() else []

    def _add(self, d: date, rec: dict):
        who = self.user_name if rec["role"] == "user" else self.ai_name
        if rec["role"] == "user" or not self.chunks or self.chunks[-1]["date"] != d:
            at = datetime.combine(d, datetime.strptime(rec["time"], "%H:%M").time())
            self.chunks.append({"key": f"{d}#{len(self.chunks):07d}", "date": d, "time": rec["time"], "at": at, "lines": []})
        c = self.chunks[-1]
        c["lines"].append(f"{who}：{rec['content']}")
        c["plain"] = " / ".join(c["lines"])
        c["text"] = f"[{c['date']} 星期{WEEK[c['date'].weekday()]} {c['time']}] {c['plain']}"
        self.vecs.pop(c["key"], None)              # 这一块变了，向量要重算

    def _load(self):
        for day in self.days():
            for rec in self.read_day(day):
                self._add(date.fromisoformat(day), rec)
        if self.emb_file.exists():
            saved = torch.load(self.emb_file)
            known = {c["key"]: c["plain"] for c in self.chunks}
            self.vecs = {k: v for k, (plain, v) in saved.items() if known.get(k) == plain}

    # ---- 检索 ----
    @torch.no_grad()
    def _embed(self, texts: list) -> torch.Tensor:
        if self._enc is None:
            from transformers import AutoModel, AutoTokenizer
            self._enc = (AutoTokenizer.from_pretrained(self.embed_model), AutoModel.from_pretrained(self.embed_model).eval())
        tok, enc = self._enc
        out = []
        for i in range(0, len(texts), 32):
            b = tok(texts[i:i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt")
            out.append(torch.nn.functional.normalize(enc(**b).last_hidden_state[:, 0], dim=-1))
        return torch.cat(out) if out else torch.empty(0, 768)

    def _ensure_vecs(self):
        todo = [c for c in self.chunks if c["key"] not in self.vecs]
        if todo:
            for c, v in zip(todo, self._embed([c["plain"] for c in todo])):
                self.vecs[c["key"]] = v
            self.unsaved += len(todo)
            if self.unsaved >= SAVE_EVERY:
                self.flush()

    def flush(self):
        """把向量缓存写盘（没写的下次启动会重算，不会出错，只是慢一点）。"""
        self.emb_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({c["key"]: (c["plain"], self.vecs[c["key"]]) for c in self.chunks if c["key"] in self.vecs}, self.emb_file)
        self.unsaved = 0

    def retrieve(self, query: str, now: datetime, budget: int, hide_from: datetime | None = None) -> str:
        """按用户原话检索原文：话里有时间说法就只在那段时间里找；按相似度排，按字数预算打包，按时间顺序输出。
        hide_from 之后的对话已经在本次会话的上下文里，不再重复取；不给就隐藏今天全部。now 之后的记录一律不取。"""
        cut = hide_from.replace(second=0, microsecond=0) if hide_from else datetime.combine(now.date(), datetime.min.time())
        with self.lock:
            pool = [c for c in self.chunks if c["at"] < min(cut, now)]
            if not pool:
                return ""
            self._ensure_vecs()
            when = parse_when(query, now.date())
            if when:
                pool = [c for c in pool if when[0] <= c["date"] <= when[1]] or pool
            q = self._embed([QUERY_PREFIX + query])[0]
            pool.sort(key=lambda c: -float(self.vecs[c["key"]] @ q))
            picked, used = [], 0
            for c in pool:
                if used + len(c["text"]) <= budget:
                    picked.append(c)
                    used += len(c["text"])
            return "\n".join(c["text"] for c in sorted(picked, key=lambda c: c["key"]))

    def habits(self) -> str:
        """相处习惯卡的最新版（夜间巩固时更新，见 nightly.update_habits）。"""
        f = self.skel_file.with_name("habits.json")
        return json.loads(f.read_text(encoding="utf-8"))[-1]["card"] if f.exists() else ""

    def skeleton(self, cap: int = SKELETON_CAP) -> str:
        if not self.skel_file.exists():
            return ""
        weeks = json.loads(self.skel_file.read_text(encoding="utf-8"))
        roll = json.loads(self.roll_file.read_text(encoding="utf-8")) if self.roll_file.exists() else {}
        lines = [(w["from"][:7], w["from"][:4], f"{w['from']}~{w['to']}：{w['line']}") for w in weeks]   # (月, 年, 文字)，周按星期一归月
        size = lambda: sum(len(t) + 1 for _, _, t in lines)
        for pos, level, label in ((0, "months", "整月"), (1, "years", "全年")):
            done = roll.get(level, {})
            for key in sorted({ln[pos] for ln in lines}):
                if size() <= cap:
                    break
                idx = [i for i, ln in enumerate(lines) if ln[pos] == key]
                if key in done and len(idx) > 1:
                    lines[idx[0]:idx[-1] + 1] = [(key[:7], key[:4], f"{key} {label}：{done[key]}")]
        return "\n".join(t for _, _, t in lines)
