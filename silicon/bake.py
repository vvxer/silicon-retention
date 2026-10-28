# -*- coding: utf-8 -*-
r"""首次烘焙：把 data\persona.md（人格卡）烤进 LoRA（Prompt Baking，见 docs\原理与实验.md）。

python -m silicon.bake [--per-cat 45]      要显卡；改了人格卡就重新跑一次（训练语料会按新人格卡重新生成）
  1. DeepSeek 按 9 类场景生成用户提问
  2. 每条提问各要一份带人格卡、一份不带的回答 → data\persona_replay.jsonl（夜间巩固也用它做人格回放和篇幅锚）
  3. 训练：教师 = 带人格卡的基座，学生 = 不带人格卡的 LoRA，全词表 KL → data\adapters\base\，写入 state.json
"""
import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor

from . import config, deepseek
from .chat import persona_text, read_state, write_state

CATEGORIES = {
    "身份": "用户询问 AI 自身：你是谁、你叫什么、你怎么看待自己、你有没有感情、你会不会记得我",
    "价值观": "征求 AI 的观点和立场：对加班、诚实、人际关系、消费、生活方式等的看法",
    "闲聊情绪": "日常闲聊和情绪倾诉：累了、烦躁、失眠、开心的事、无聊、被批评、想家",
    "建议": "生活、工作、学习上的决策求助",
    "知识": "科普和常识问答",
    "数学": "带具体数字的应用题和逻辑题",
    "创作": "写诗、写文案、写段子、写邮件、写短故事、起名字",
    "编程": "编程问题、解释技术概念、报错怎么办",
    "工作": "写周报、写方案提纲、整理会议纪要、准备汇报",
}
ASK = """生成 {n} 条真实用户会发给 AI 助手的消息。类别：{desc}。
要求：口吻多样（正式、随意、口语），长短不一（5 到 80 字），话题互不重复，不要编号。
只输出 JSON：{{"prompts": ["...", "..."]}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=45)
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()
    cfg = config.load()
    if not (cfg.data / "persona.md").exists() and (cfg.data / "memory" / "persona-seen.txt").exists():
        raise SystemExit(f"这是 DSH 插件的数据目录：先把人格卡写进 {cfg.data / 'persona.md'}（和 DSH 里设定的人设一致），再烘焙")
    persona = persona_text(cfg)
    client = deepseek.Client()
    replay = cfg.data / "persona_replay.jsonl"
    st = read_state(cfg)
    sha = hashlib.sha256(persona.encode("utf-8")).hexdigest()[:12]
    if replay.exists() and st.get("persona_sha", sha) != sha:        # 人格卡改过：旧语料留档，按新人格卡重新生成
        replay.rename(replay.with_name(f"persona_replay.{st['persona_sha']}.jsonl"))
        print("人格卡改过了，重新生成训练语料", flush=True)

    if not replay.exists():
        def gen(item):
            r = client.chat([{"role": "user", "content": ASK.format(n=a.per_cat, desc=item[1])}], temperature=1.0, max_tokens=4000,
                            response_format={"type": "json_object"})
            return [(item[0], p) for p in json.loads(r["choices"][0]["message"]["content"])["prompts"]]

        with ThreadPoolExecutor(8) as ex:
            prompts = [p for ps in ex.map(gen, CATEGORIES.items()) for p in ps]
        print(f"提问 {len(prompts)} 条", flush=True)

        def answer(job):
            (cat, p), with_persona = job
            msgs = ([{"role": "system", "content": persona}] if with_persona else []) + [{"role": "user", "content": p}]
            r = client.chat(msgs, temperature=1.0, max_tokens=400)
            return {"cat": cat, "prompt": p, "persona": with_persona, "response": r["choices"][0]["message"]["content"]}

        with ThreadPoolExecutor(8) as ex:
            rows = list(ex.map(answer, [(p, w) for p in prompts for w in (True, False)]))
        with open(replay, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"回答 {len(rows)} 条 → {replay}", flush=True)

    from .train import load_model, train
    rows = [json.loads(l) for l in open(replay, encoding="utf-8")]
    random.Random(0).shuffle(rows)
    tok, model = load_model(cfg.local_model, cfg.vram_gb, None)
    out = cfg.data / "adapters" / "base"
    info = train(model, tok, [{"kind": "kl", **r} for r in rows], persona, out, epochs=a.epochs, lr=1e-4)
    st = read_state(cfg)
    st.update(adapter=str(out), base_adapter=str(out), persona_sha=sha)
    if st.pop("last_consolidated", None):        # 新的基础权重上没有以前的相处：下次夜间巩固从全部原文日志重新学
        print("注意：之前巩固进权重的相处已清空，下次夜间巩固会从全部原文日志重新学（日志多的话要久一些）")
    write_state(cfg, st)
    print("完成", info)


if __name__ == "__main__":
    main()
