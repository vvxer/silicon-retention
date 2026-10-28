# -*- coding: utf-8 -*-
r"""夜间巩固：把还没巩固过的日子写进 LoRA，并更新概况。

python -m silicon.nightly [--include-today] [--skeleton-only | --weights-only]     要显卡（--skeleton-only 只更新记忆，不要显卡）；建议每天夜里定时跑
--weights-only：只巩固权重、不更新记忆，给 DSH 插件用（插件自己整理记忆），由 silicon.lora_server 每晚调用
  1. 记忆（有没有本地权重都做；进度记在 state.json 的 memory_until）：
     概况：涉及的自然周各用 DeepSeek 写一行（≤60 字）→ data\memory\skeleton.json；
       一周聊天超过 8000 字的，先逐天摘要（存 data\memory\day_notes.json），再用摘要写这一行
     已经过完的月、年各写一行 → data\memory\rollup.json（概况太长时用来替换最早的周，见 memory.py）
     相处习惯卡：新日子按周、每段 ≤20000 字分段喂给 DeepSeek 增量更新，最多 12 条、≤1000 字 → data\memory\habits.json
  2. 情绪标记（可选，config 里 emotion_weighting 打开，默认关；只在巩固权重时做）：DeepSeek 读每天的聊天，标出用户情绪明显（1）、很强（2）的一问一答
     → data\memory\emotion.json；一晚最多补 60 天，从近到远，没标到的按平常算。
     像人脑里杏仁核给经历打的标记：情绪强的经历在睡眠里优先重放，记得更牢
  3. 训练数据（见 docs\原理与实验.md；进度记在 last_consolidated）：
       新日子里她的每条发言（上下文 = 当天之前的对话，用户消息带 [星期X HH:MM]），一晚最多 600 条  交叉熵
         超了均匀抽；开了情绪加权则按情绪抽（平常 1、明显 2、很强 4），情绪很强的当晚多学一遍
       旧日子样本回放一半（上限 300，抽法同上）                            交叉熵，防遗忘
       人格回放 150 条（persona_replay.jsonl）                                      KL，防基础人格漂移
       篇幅锚 60 条（人格卡下 200 字以上的回答）                                    交叉熵，防"越学越短"
  4. 从当前 LoRA 接着训练 → data\adapters\YYYY-MM-DD\，更新 state.json；保留最近 7 个，出问题可以改 state.json 回滚
"""
import argparse
import json
import random
import re
import shutil
from datetime import date, datetime, timedelta

from . import config, deepseek
from .chat import persona_text, read_state, write_state
from .memory import Store, stamp

SKELETON_ASK = "下面是用户（{user}）和 AI 伙伴{ai}一周的聊天记录。用一句话（不超过 60 字）概括{user}这一周主要在经历什么，保留关键人名和事件。只输出这一句话。\n\n{log}"
DAY_ASK = "下面是用户（{user}）和 AI 伙伴{ai}在 {day} 的聊天记录{part}。用两三句话（不超过 120 字）记下{user}这段时间主要在经历什么、聊了什么，保留关键人名和事件。只输出这几句话。\n\n{log}"
DAY_MERGE_ASK = "下面是{user}在 {day} 一天里各段聊天的摘要。合并成两三句话（不超过 120 字），保留关键人名和事件。只输出这几句话。\n\n{lines}"
SKELETON_PIECE = 8000      # 一次喂给 DeepSeek 写概况的原文上限（字）；聊得多的人一周有几十万字，只取末尾会丢掉前几天
HABITS_PIECE = 20000       # 习惯卡每次喂的原文上限，超了分段、逐段更新卡
NEW_CAP = 600              # 一晚最多学多少条新发言，训练时间不随聊天量无限增长
EMOTION_ASK = """下面是用户（{user}）和 AI 伙伴{ai}在 {day} 的聊天{part}，每一问一答前面有编号。按{user}这句话里的情绪给每一问一答定强度：
2 = 情绪很强：吵架、崩溃、大哭、狂喜，或者人生大事（分手、表白、生病、亲人离世、升职、被裁）
1 = 情绪明显：直接说出难过、委屈、生气、害怕、孤独、感动、特别开心，或者明显的压力和焦虑
0 = 其他：日常的累、烦、忙和随口抱怨，平常的闲聊、问问题、让{ai}干活

只输出强度是 1 和 2 的，每处一行："编号 强度"。一处都没有就输出"无"。

{log}"""
EMOTION_WEIGHT = (1, 2, 4) # 情绪强度 0/1/2 的样本被抽中的相对机会
TAG_CAP = 60               # 一晚最多给多少天打情绪标记；第一次开或重新烘焙后，旧日子分几晚补完
MONTH_ASK = "下面是{user}某个月每周的概况。用一句话（不超过 60 字）概括这个月，只保留最重要的人和事。只输出这一句话。\n\n{lines}"
YEAR_ASK = "下面是{user}某一年每个月的概况。用一两句话（不超过 100 字）概括这一年，只保留最重要的几件事和关键人物。只输出这段话。\n\n{lines}"
# 相处习惯卡（阶段 5：检索给够预算 + 这张卡，纯 DeepSeek 就追平了"读完整日记"）
HABITS_ASK = """你在帮 AI 伙伴"{ai}"维护一张《相处习惯卡》，记录{ai}和{user}在相处中形成的、下面人格档案里没写的东西：
你们之间的梗和暗号、昵称、固定说法、小仪式、约定，{user}的偏好和雷区，以及{ai}在这些场合通常怎么说。

规则：
- 只依据聊天记录，不要编。
- 每条一行："情境 → {ai}的说法或做法"，附一句原话（用引号）。
- 新形成的加进去；变了的改成最新的样子，写明怎么变的（"原来……，现在……"）；很久没再出现、明显不用了的删掉。
- 人格档案里已经有的通用性格不要重复。
- 最多 12 条，每条不超过 70 字（含原话），按重要程度排序，最重要的在前。

人格档案：
{persona}

当前的卡（第一次为空）：
{card}

这段时间（{start} ~ {end}）的聊天记录：
{log}

只输出更新后的卡，不要解释。"""
HABITS_CAP = 1000          # 卡每轮都要发，必须封顶；模型常不守字数，超了按条目从后往前截（条目按重要程度排）


def day_samples(store: Store, day: str) -> list:
    msgs = []
    for r in store.read_day(day):
        t = datetime.fromisoformat(f"{day} {r['time']}")
        role = r["role"]
        text = f"{stamp(t)} {r['content']}" if role == "user" else r["content"]
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n" + text
        else:
            msgs.append({"role": role, "content": text})
    return [{"kind": "sft", "messages": msgs[:j], "response": m["content"]}
            for j, m in enumerate(msgs) if m["role"] == "assistant" and j > 0]


def cut(text: str, n: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


def tag_day(cfg, day: str, samples: list, client) -> dict:
    """一天里情绪明显的一问一答：{样本序号: 强度}。用户的话截到 300 字，她的回答截到 100 字（看情绪主要看用户说了什么）。"""
    lines = [f"[{k}] {cfg.user_name}：{cut(s['messages'][-1]['content'], 300)}　{cfg.ai_name}：{cut(s['response'], 100)}"
             for k, s in enumerate(samples)]
    ps = pieces([(day, lines)], SKELETON_PIECE)
    tags = {}
    for i, p in enumerate(ps):
        out = ask(client, EMOTION_ASK.format(user=cfg.user_name, ai=cfg.ai_name, day=day,
                                             part=f"（第 {i + 1}/{len(ps)} 段）" if len(ps) > 1 else "", log=p))
        for k, lv in re.findall(r"^\D*(\d+)\D+([12])(?!\d)", out, re.M):
            if int(k) < len(samples):
                tags[int(k)] = max(tags.get(int(k), 0), int(lv))
    return tags


def emotion_tags(cfg, store: Store, days: list, client, today: str) -> dict:
    """{日期: {样本序号: 强度}}，过完的日子存进 emotion.json。这一晚最多补 TAG_CAP 天，从近到远；出错就停，没标的按平常算。"""
    f = cfg.data / "memory" / "emotion.json"
    saved = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    tags = {d: {int(k): v for k, v in t.items()} for d, t in saved.items()}
    todo = [d for d in reversed(days) if d not in tags][:TAG_CAP]
    for d in todo:
        samples = day_samples(store, d)
        try:
            tags[d] = tag_day(cfg, d, samples, client) if samples else {}
        except (RuntimeError, OSError) as e:
            print("情绪标记出错，今晚没标到的日子按平常算：", e, flush=True)
            break
        if d < today:
            saved[d] = {str(k): v for k, v in sorted(tags[d].items())}
    if todo:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(dict(sorted(saved.items())), ensure_ascii=False), encoding="utf-8")
        n = [lv for d in todo for lv in tags.get(d, {}).values()]
        print(f"情绪标记：{len(todo)} 天，明显 {n.count(1)} 处，很强 {n.count(2)} 处", flush=True)
    return tags


def wsample(rng, items: list, k: int) -> list:
    """[(样本, 情绪强度)] 按 EMOTION_WEIGHT 不放回地抽 k 个（Efraimidis–Spirakis），不够 k 个就全要。"""
    if k >= len(items):
        return list(items)
    keys = [rng.random() ** (1 / EMOTION_WEIGHT[lv]) for _, lv in items]
    return [items[i] for i in sorted(range(len(items)), key=keys.__getitem__, reverse=True)[:k]]


def build_rows(cfg, store: Store, new_days: list, old_days: list, tags: dict, rng) -> list:
    leveled = lambda days: [(s, tags.get(d, {}).get(k, 0)) for d in days for k, s in enumerate(day_samples(store, d))]
    new = wsample(rng, leveled(new_days), NEW_CAP)
    old = leveled(old_days)
    replay = wsample(rng, old, min(300, len(old) // 2))
    strong = [s for s, lv in new if lv == 2]
    rows = [s for s, _ in new] + strong + [s for s, _ in replay]
    pfile = cfg.data / "persona_replay.jsonl"
    if pfile.exists():
        pr = [json.loads(l) for l in open(pfile, encoding="utf-8")]
        rows += [{"kind": "kl", **r} for r in rng.sample(pr, min(150, len(pr)))]
        long = [r for r in pr if r.get("persona") and len(r["response"]) >= 200]
        rows += [{"kind": "sft", "messages": [{"role": "user", "content": r["prompt"]}], "response": r["response"]}
                 for r in rng.sample(long, min(60, len(long)))]
    rng.shuffle(rows)
    emo = f"；情绪很强的 {len(strong)} 条多学一遍，回放里带情绪的 {sum(lv > 0 for _, lv in replay)} 条" if tags else ""
    print(f"样本：新 {len(new)}，回放 {len(replay)}，合计 {len(rows)}{emo}", flush=True)
    return rows


def day_lines(cfg, store: Store, day: str) -> list:
    who = {"user": cfg.user_name, "assistant": cfg.ai_name}
    return [f"[{r['time']}] {who[r['role']]}：{r['content']}" for r in store.read_day(day)]


def pieces(days: list, size: int) -> list:
    """[(日期, 各行)] → 若干段文本，每段不超过 size 字，每天以 "== 日期 ==" 开头；一天被切开时，下一段开头补 "== 日期（续）=="。
    聊得少的话就是一整段，和不分段时一样。"""
    out, cur, n = [], [], 0
    for d, lines in days:
        for i, line in enumerate(lines):
            add = ([f"== {d} =="] if i == 0 else []) + [line[:size // 2]]      # 单条超长的截断，保证一段至少放得下一条
            if cur and n + sum(len(x) + 1 for x in add) > size + 1:
                out.append("\n".join(cur))
                cur = [] if i == 0 else [f"== {d}（续）=="]
                n = sum(len(x) + 1 for x in cur)
            cur += add
            n += sum(len(x) + 1 for x in add)
    if cur:
        out.append("\n".join(cur))
    return out


def day_note(cfg, store: Store, day: str, client, notes: dict, keep: bool) -> str:
    """一天聊天的两三句摘要（超过 SKELETON_PIECE 的先分段摘要再合并）；过完的日子存进 notes，不用重算。"""
    if day in notes:
        return notes[day]
    ps = pieces([(day, day_lines(cfg, store, day))], SKELETON_PIECE)
    fmt = {"user": cfg.user_name, "ai": cfg.ai_name, "day": day}
    parts = [ask(client, DAY_ASK.format(**fmt, part=f"（第 {i + 1}/{len(ps)} 段）" if len(ps) > 1 else "", log=p))
             for i, p in enumerate(ps)]
    note = parts[0] if len(parts) == 1 else ask(client, DAY_MERGE_ASK.format(**fmt, lines="\n".join(parts)))
    if keep and note:
        notes[day] = note
    return note


def update_skeleton(cfg, store: Store, days: list, client, today: str):
    f = cfg.data / "memory" / "skeleton.json"
    nf = cfg.data / "memory" / "day_notes.json"
    weeks = {w["from"]: w for w in json.loads(f.read_text(encoding="utf-8"))} if f.exists() else {}
    notes = json.loads(nf.read_text(encoding="utf-8")) if nf.exists() else {}
    mondays = sorted({str(date.fromisoformat(d) - timedelta(date.fromisoformat(d).weekday())) for d in days})
    for mon in mondays:
        m = date.fromisoformat(mon)
        week_days = [str(m + timedelta(i)) for i in range(7) if str(m + timedelta(i)) in store.days()]
        ps = pieces([(d, day_lines(cfg, store, d)) for d in week_days], SKELETON_PIECE)
        if len(ps) == 1:
            log = ps[0]
        else:       # 聊得多的一周：逐天摘要，再用摘要写这一周
            log = "\n".join(f"== {d} ==\n{day_note(cfg, store, d, client, notes, d < today)}" for d in week_days)[-SKELETON_PIECE:]
        weeks[mon] = {"from": mon, "to": week_days[-1],
                      "line": ask(client, SKELETON_ASK.format(user=cfg.user_name, ai=cfg.ai_name, log=log))}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps([weeks[k] for k in sorted(weeks)], ensure_ascii=False, indent=1), encoding="utf-8")
    nf.write_text(json.dumps(notes, ensure_ascii=False, indent=1), encoding="utf-8")
    update_rollup(cfg, list(weeks.values()), client)


def ask(client, prompt: str, max_tokens: int = 300) -> str:
    r = client.chat([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=max_tokens)
    return r["choices"][0]["message"]["content"].strip()


def update_habits(cfg, store: Store, days: list, client):
    """把新日子按自然周喂给 DeepSeek，增量更新相处习惯卡（一周超过 HABITS_PIECE 字的分段，逐段更新）。"""
    f = cfg.data / "memory" / "habits.json"
    versions = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    card = versions[-1]["card"] if versions else "（空）"
    weeks = {}
    for d in days:
        weeks.setdefault(str(date.fromisoformat(d) - timedelta(date.fromisoformat(d).weekday())), []).append(d)
    for mon in sorted(weeks):
        wk = weeks[mon]
        for log in pieces([(d, day_lines(cfg, store, d)) for d in wk], HABITS_PIECE):
            card = ask(client, HABITS_ASK.format(ai=cfg.ai_name, user=cfg.user_name, persona=persona_text(cfg), card=card,
                                                 start=wk[0], end=wk[-1], log=log), max_tokens=1200)
            if len(card) > HABITS_CAP:
                kept = []
                for line in card.splitlines():
                    if sum(len(x) + 1 for x in kept) + len(line) > HABITS_CAP:
                        break
                    kept.append(line)
                card = "\n".join(kept).strip()
        versions.append({"until": wk[-1], "card": card})
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(versions[-30:], ensure_ascii=False, indent=1), encoding="utf-8")


def update_rollup(cfg, weeks: list, client, today: date | None = None):
    """已经过完的月、年各概括一行（只补缺的）。"""
    f = cfg.data / "memory" / "rollup.json"
    roll = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    months, years = roll.setdefault("months", {}), roll.setdefault("years", {})
    today = today or date.today()
    weeks = sorted(weeks, key=lambda w: w["from"])
    for m in sorted({w["from"][:7] for w in weeks}):
        if m < f"{today:%Y-%m}" and m not in months:
            lines = "\n".join(f"{w['from']}~{w['to']}：{w['line']}" for w in weeks if w["from"][:7] == m)
            months[m] = ask(client, MONTH_ASK.format(user=cfg.user_name, lines=lines))
    for y in sorted({m[:4] for m in months}):
        if y < str(today.year) and y not in years:
            lines = "\n".join(f"{m}：{months[m]}" for m in sorted(months) if m[:4] == y)
            years[y] = ask(client, YEAR_ASK.format(user=cfg.user_name, lines=lines))
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(roll, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-today", action="store_true")
    ap.add_argument("--skeleton-only", action="store_true", help="只更新概况和相处习惯卡，不训练（不用本地权重时）")
    ap.add_argument("--weights-only", action="store_true", help="只巩固权重，不更新记忆（DSH 插件自己整理记忆）")
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()
    cfg = config.load()
    st = read_state(cfg)
    store = Store(cfg.data, cfg.embed_model, cfg.user_name, cfg.ai_name)
    today = str(date.today())
    client = deepseek.Client()

    mem_days = [d for d in store.days() if d > st.get("memory_until", "") and (a.include_today or d < today)]
    if mem_days and not a.weights_only:
        print("更新记忆：", mem_days[0], "~", mem_days[-1], f"共 {len(mem_days)} 天", flush=True)
        update_skeleton(cfg, store, mem_days, client, today)
        update_habits(cfg, store, mem_days, client)
        st["memory_until"] = mem_days[-1]
        write_state(cfg, st)
    if a.skeleton_only:
        return
    if not st.get("base_adapter"):
        print("还没有烘焙人格，" + ("没有可巩固的权重" if a.weights_only else "只更新了记忆") + "；要用本地权重先运行 python -m silicon.bake")
        return
    done_until = st.get("last_consolidated", "")
    new_days = [d for d in store.days() if d > done_until and (a.include_today or d < today)]
    if not new_days:
        print("没有需要巩固的日子")
        return
    print("巩固：", new_days[0], "~", new_days[-1], f"共 {len(new_days)} 天", flush=True)

    old_days = [d for d in store.days() if d <= done_until]
    tags = emotion_tags(cfg, store, old_days + new_days, client, today) if cfg.emotion_weighting else {}
    rows = build_rows(cfg, store, new_days, old_days, tags, random.Random(today))

    from .train import load_model, train
    tok, model = load_model(cfg.local_model, cfg.vram_gb, st.get("adapter"))
    out = cfg.data / "adapters" / new_days[-1]
    info = train(model, tok, rows, persona_text(cfg), out, epochs=a.epochs)
    st = read_state(cfg)        # 训练要十几分钟，这期间记忆进度可能被别的进程（DSH 插件）更新过：重读再写
    st.update(adapter=str(out), last_consolidated=new_days[-1])
    write_state(cfg, st)
    keep = sorted(p for p in (cfg.data / "adapters").iterdir() if p.is_dir() and p.name[:2] == "20")[:-7]
    for p in keep:
        shutil.rmtree(p)
    print("完成", info)


if __name__ == "__main__":
    main()
