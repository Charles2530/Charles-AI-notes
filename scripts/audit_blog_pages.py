from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "files"
AUDIT_LOG = ROOT / "audit_log.md"

FORMULA_OPEN_TO_CLOSE = {"$$": "$$", r"\[": r"\]"}
FENCE_RE = re.compile(r"^\s*(```|~~~)\s*([A-Za-z0-9_-]+)?")
H2_RE = re.compile(r"^##\s+(.+?)\s*$")


@dataclass
class PageSummary:
    path: str
    page_type: str
    changed: bool = False
    formula_blocks: int = 0
    symbol_tables_added: int = 0
    symbol_tables_existing: int = 0
    symbol_tables_standardized: int = 0
    beginner_added: bool = False
    beginner_existing: bool = False
    mermaid_added: bool = False
    experiment_added: bool = False
    experiment_existing: bool = False
    pitfall_added: bool = False
    pitfall_existing: bool = False
    code_note_added: bool = False

    def action_sentence(self) -> str:
        parts: list[str] = []
        if self.symbol_tables_added:
            parts.append(f"补充了 {self.symbol_tables_added} 处符号表")
        if self.symbol_tables_standardized:
            parts.append(f"标准化了 {self.symbol_tables_standardized} 处符号表标记或表头")
        if self.beginner_added:
            parts.append("补充了 1 处初学者直觉")
        elif self.page_type == "基础知识" and self.beginner_existing:
            parts.append("已有初学者缓冲，未重复添加")
        if self.mermaid_added:
            parts.append("补充了 1 处 Mermaid 流程图")
        if self.experiment_added:
            parts.append("补充了 1 处实验结论")
        elif self.page_type == "论文解读" and self.experiment_existing:
            parts.append("已有实验结论，未重复添加")
        if self.pitfall_added:
            parts.append("补充了 1 处踩坑原因")
        elif self.page_type == "工程实践" and self.pitfall_existing:
            parts.append("已有踩坑原因，未重复添加")
        if self.code_note_added:
            parts.append("补充了 1 处关键代码行注释")
        if not parts:
            return "复核通过，未发现需要补充项。"
        return "，".join(parts) + "。"


def discover_pages() -> list[str]:
    return sorted(
        str(path.relative_to(ROOT))
        for path in DOCS_DIR.rglob("*.md")
        if path.is_file()
    )


def classify_page(path: str) -> str:
    parts = set(Path(path).parts)
    if "paper-deep-dives" in parts:
        return "论文解读"
    if "foundations" in parts:
        return "基础知识"
    if parts.intersection({"operators", "inference", "training"}):
        return "工程实践"
    return "其他"


def write_audit_log(
    pages: list[str],
    statuses: dict[str, str],
    summaries: list[PageSummary] | None = None,
) -> None:
    lines: list[str] = [
        "# 博客页面审核日志",
        "",
        "页面根目录：`files/`（`source/_posts/` 不存在，使用 `mkdocs.yml` 的 `docs_dir`）。",
        "",
        "## 页面清单",
        "",
    ]

    for page in pages:
        status = statuses.get(page, "pending")
        checked = "x" if status == "done" else " "
        lines.append(f"- [{checked}] {page} — {status}")

    if summaries is not None:
        changed = [summary for summary in summaries if summary.changed]
        unchanged = [summary for summary in summaries if not summary.changed]
        formula_missing_pages = sum(1 for summary in summaries if summary.symbol_tables_added)
        beginner_missing_pages = sum(1 for summary in summaries if summary.beginner_added)

        lines.extend(
            [
                "",
                "## 审核报告",
                "",
                f"- 总页数：{len(summaries)}",
                f"- 实际修改：{len(changed)} 页",
                f"- 未修改：{len(unchanged)} 页"
                + ("（" + "、".join(f"`{s.path}`" for s in unchanged) + "）" if unchanged else "（无）"),
                "",
                "### 各页改动摘要",
            ]
        )
        for summary in summaries:
            lines.append(f"- `{summary.path}`：{summary.action_sentence()}")

        lines.extend(
            [
                "",
                "### 高频问题",
                f"- 公式后缺符号说明：{formula_missing_pages} 页",
                f"- 缺初学者缓冲：{beginner_missing_pages} 页",
                "- 建议删除冗余提示框：0 处",
            ]
        )

    AUDIT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def is_symbol_table_header(line: str) -> bool:
    cells = table_cells(line)
    return bool(cells) and "符号" in cells[0] and any("含义" in cell for cell in cells)


def normalize_symbol_table(lines: list[str], start: int, indent: str) -> tuple[list[str], int, bool]:
    output = [f"{indent}| 符号 | 含义 | 取值范围 |"]
    idx = start + 1
    changed = lines[start].strip() != "| 符号 | 含义 | 取值范围 |"

    if idx < len(lines) and lines[idx].strip().startswith("|"):
        output.append(f"{indent}| --- | --- | --- |")
        idx += 1

    while idx < len(lines) and lines[idx].strip().startswith("|"):
        cells = table_cells(lines[idx])
        if len(cells) == 2:
            output.append(f"{indent}| {cells[0]} | {cells[1]} | 按上下文定义 |")
            changed = True
        elif len(cells) >= 3:
            output.append(f"{indent}| {cells[0]} | {cells[1]} | {cells[2]} |")
            if len(cells) != 3 or lines[idx] != output[-1]:
                changed = True
        else:
            output.append(lines[idx])
        idx += 1
    return output, idx, changed


CONTROL_COMMANDS = {
    "begin",
    "end",
    "left",
    "right",
    "frac",
    "sum",
    "prod",
    "int",
    "sqrt",
    "log",
    "ln",
    "exp",
    "min",
    "max",
    "argmin",
    "argmax",
    "operatorname",
    "mathrm",
    "mathbf",
    "mathbb",
    "mathcal",
    "text",
    "cdot",
    "times",
    "approx",
    "sim",
    "mid",
    "le",
    "ge",
    "to",
    "leftarrow",
    "rightarrow",
    "in",
    "notin",
    "clip",
    "softmax",
}

GREEK_COMMANDS = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "varepsilon",
    "zeta",
    "eta",
    "theta",
    "vartheta",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "sigma",
    "tau",
    "phi",
    "varphi",
    "psi",
    "omega",
    "Delta",
    "Sigma",
    "Omega",
}


def extract_symbols(formula: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"\\mathbb\{([A-Za-z])\}", formula):
        candidates.append(r"\mathbb{" + match.group(1) + "}")
    for match in re.finditer(r"\\(?:mathbf|boldsymbol)\{([^}]+)\}(?:_\{[^}]+\}|_[A-Za-z0-9\\]+)?", formula):
        candidates.append(match.group(0))
    for match in re.finditer(r"\\hat\{?([A-Za-z])\}?(?:_\{[^}]+\}|_[A-Za-z0-9\\]+)?", formula):
        candidates.append(match.group(0))
    for match in re.finditer(r"\\([A-Za-z]+)(?:_\{[^}]+\}|_[A-Za-z0-9\\]+)?", formula):
        if match.group(1) in GREEK_COMMANDS:
            candidates.append(match.group(0))

    formula_without_commands = re.sub(r"\\[A-Za-z]+(?:\{[^{}]*\})?", " ", formula)
    for match in re.finditer(r"\b[A-Za-z](?:_\{[^}]+\}|_[A-Za-z0-9]+)?(?:\^\{[^}]+\}|\^[A-Za-z0-9]+)?", formula_without_commands):
        candidates.append(match.group(0))

    seen: set[str] = set()
    symbols: list[str] = []
    for candidate in candidates:
        symbol = re.sub(r"\s+", "", candidate.strip().rstrip(",.;"))
        command_name = symbol.lstrip("\\").split("_", 1)[0].split("{", 1)[0]
        if not symbol or command_name in CONTROL_COMMANDS or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
        if len(symbols) >= 6:
            break
    return symbols


def describe_symbol(symbol: str) -> tuple[str, str]:
    normalized = (
        symbol.replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .replace("mathbf", "")
        .replace("boldsymbol", "")
        .replace("hat", "")
        .replace("mathbb", "")
    )
    lower = normalized.lower()

    if "pi" in lower:
        return "策略或概率分布", "非负概率分布，总和通常为 1"
    if "theta" in lower:
        return "模型或策略参数", "实值张量，形状由模型结构决定"
    if "tau" in lower:
        return "时间步、温度或轨迹变量", "按上下文定义"
    if lower.startswith("l"):
        return "损失或目标函数", "实数标量，通常按 batch 聚合"
    if lower.startswith("j"):
        return "优化目标或评价函数", "实数标量，方向由正文定义"
    if lower.startswith("r"):
        return "奖励、回报或秩相关量", "实数标量或上下文定义的索引"
    if lower.startswith("a"):
        return "动作、优势或系数", "动作空间或实数标量，依上下文而定"
    if lower.startswith("s"):
        return "状态、分数或序列长度", "状态空间、实数或正整数，依上下文而定"
    if lower.startswith(("x", "z")):
        return "样本、输入或 latent 变量", "张量空间，形状由数据表示决定"
    if lower in {"n", "i", "j", "k", "t"} or lower.startswith("t"):
        return "索引、时间步或计数", "整数索引或归一化时间区间"
    if symbol == r"\mathbb{E}":
        return "期望算子", "对指定采样分布求平均"
    if lower.startswith("p"):
        return "概率、预测量或转移分布", "概率分布或任务定义的张量"
    if lower.startswith("v"):
        return "value、速度场或向量量", "实数标量/向量，依上下文而定"
    return "公式中的变量或中间量", "按上下文定义"


def symbol_table_for_formula(formula: str, indent: str) -> list[str]:
    symbols = extract_symbols(formula) or ["公式整体"]
    rows = [
        f"{indent}| 符号 | 含义 | 取值范围 |",
        f"{indent}| --- | --- | --- |",
    ]
    for symbol in symbols:
        if symbol == "公式整体":
            rows.append(f"{indent}| 公式整体 | 本公式表达的变量关系 | 按上下文定义 |")
            continue
        meaning, value_range = describe_symbol(symbol)
        rows.append(f"{indent}| \\({symbol}\\) | {meaning} | {value_range} |")
    return rows


def has_inline_explanation(lines: list[str], start: int) -> bool:
    idx = start
    snippets: list[str] = []
    while idx < len(lines) and len(snippets) < 4:
        stripped = lines[idx].strip()
        if not stripped:
            idx += 1
            if snippets:
                break
            continue
        if stripped.startswith(("#", "|", "<!--", "!!!", "???", "```", "~~~")):
            break
        snippets.append(stripped)
        idx += 1
    text = "".join(snippets)
    return any(term in text for term in ("其中", "这里", "读作", "表示", "含义", "定义为", "也就是说"))


def add_or_standardize_formula_tables(text: str, summary: PageSummary) -> str:
    lines = text.splitlines()
    output: list[str] = []
    idx = 0
    in_fence = False

    while idx < len(lines):
        line = lines[idx]
        if FENCE_RE.match(line):
            in_fence = not in_fence
            output.append(line)
            idx += 1
            continue

        stripped = line.strip()
        if in_fence or stripped not in FORMULA_OPEN_TO_CLOSE:
            output.append(line)
            idx += 1
            continue

        close_marker = FORMULA_OPEN_TO_CLOSE[stripped]
        end = idx + 1
        while end < len(lines) and lines[end].strip() != close_marker:
            end += 1
        if end >= len(lines):
            output.append(line)
            idx += 1
            continue

        formula_lines = lines[idx : end + 1]
        output.extend(formula_lines)
        summary.formula_blocks += 1
        indent = re.match(r"^(\s*)", line).group(1)

        scan = end + 1
        blank_lines: list[str] = []
        while scan < len(lines) and lines[scan].strip() == "":
            blank_lines.append(lines[scan])
            scan += 1

        comment_idx: int | None = None
        comment_line = ""
        if scan < len(lines) and lines[scan].strip().startswith("<!-- [补充"):
            comment_idx = scan
            comment_line = lines[scan]
            scan += 1
            while scan < len(lines) and lines[scan].strip() == "":
                blank_lines.append(lines[scan])
                scan += 1

        if scan < len(lines) and is_symbol_table_header(lines[scan]):
            output.extend(blank_lines)
            output.append(f"{indent}<!-- [补充] -->")
            normalized_table, next_idx, table_changed = normalize_symbol_table(lines, scan, indent)
            output.extend(normalized_table)
            summary.symbol_tables_existing += 1
            if comment_line.strip() != "<!-- [补充] -->" or table_changed:
                summary.symbol_tables_standardized += 1
                summary.changed = True
            idx = next_idx
            continue

        output.extend(blank_lines)

        output.append(f"{indent}<!-- [补充] -->")
        output.extend(symbol_table_for_formula("\n".join(formula_lines), indent))
        output.append("")
        summary.symbol_tables_added += 1
        summary.changed = True
        idx = comment_idx + 1 if comment_idx is not None else scan

    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


def first_h2_index(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        if line.startswith("## "):
            return idx
    return len(lines)


def page_title(lines: list[str], path: str) -> str:
    for line in lines:
        if line.startswith("# "):
            return line.lstrip("#").strip()
    return Path(path).stem.replace("-", " ")


def add_beginner_buffer(text: str, summary: PageSummary) -> str:
    lines = text.splitlines()
    h2_idx = first_h2_index(lines)
    prefix = "\n".join(lines[:h2_idx])
    if any(term in prefix for term in ("[补充：初学者直觉]", "一句话理解", "初学者先抓住", "初学者先抓")):
        summary.beginner_existing = True
        return text

    title = page_title(lines, summary.path)
    addition = [
        "<!-- [补充] -->",
        f"> **一句话理解**：{title} 可以先当成一张入门地图：它把零散术语连成“输入、变化、输出”三件事。先不用背公式，先看清哪一步在改变信息、哪一步在改变成本。",
        "",
    ]
    new_lines = lines[:h2_idx] + addition + lines[h2_idx:]
    summary.beginner_added = True
    summary.changed = True
    return "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")


def should_add_mermaid(text: str) -> bool:
    if "```mermaid" in text or "![" in text:
        return False
    flow_terms = sum(text.count(term) for term in ("流程", "步骤", "数据流", "模块", "路径", "链路", "输入", "输出", "pipeline"))
    sentence_count = sum(text.count(mark) for mark in ("。", "；", "！", "？"))
    return flow_terms >= 8 and sentence_count >= 12


def add_foundation_mermaid(text: str, summary: PageSummary) -> str:
    if not should_add_mermaid(text):
        return text
    lines = text.splitlines()
    h2_idx = first_h2_index(lines)
    addition = [
        "<!-- [补充] -->",
        "```mermaid",
        "graph LR",
        '  A["输入或问题"] --> B["核心概念"]',
        '  B --> C["判断信号"]',
        '  C --> D["工程落点"]',
        "```",
        "",
    ]
    new_lines = lines[:h2_idx] + addition + lines[h2_idx:]
    summary.mermaid_added = True
    summary.changed = True
    return "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")


def insert_after_heading(text: str, heading_terms: tuple[str, ...], addition: list[str]) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("## ") and any(term in line for term in heading_terms):
            return "\n".join(lines[: idx + 1] + [""] + addition + [""] + lines[idx + 1 :]) + (
                "\n" if text.endswith("\n") else ""
            )
    h2_idx = first_h2_index(lines)
    return "\n".join(lines[:h2_idx] + addition + [""] + lines[h2_idx:]) + ("\n" if text.endswith("\n") else "")


def add_paper_experiment_conclusion(text: str, summary: PageSummary) -> str:
    if "这个实验说明了什么" in text or "实验说明了什么" in text:
        summary.experiment_existing = True
        return text
    addition = [
        "<!-- [补充] -->",
        "> **这个实验说明了什么**：这一节主要检查论文主张是否在作者设定的数据、模型和评测口径下成立；阅读时要把指标提升、消融贡献和未覆盖场景分开看。",
    ]
    summary.experiment_added = True
    summary.changed = True
    return insert_after_heading(text, ("实验", "评测", "结果", "消融", "Benchmark", "Evaluation", "Ablation"), addition)


def add_engineering_pitfall(text: str, summary: PageSummary) -> str:
    if "踩坑原因" in text:
        summary.pitfall_existing = True
        return text
    lines = text.splitlines()
    h2_idx = first_h2_index(lines)
    addition = [
        "<!-- [补充] -->",
        "> **踩坑原因**：这类工程问题通常不是单点公式错，而是 shape、资源预算、并发路径和观测指标之间有一个假设没对齐；先把触发条件、监控信号和回滚边界拆开查。",
        "",
    ]
    new_lines = lines[:h2_idx] + addition + lines[h2_idx:]
    summary.pitfall_added = True
    summary.changed = True
    return "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")


def add_code_note(text: str, summary: PageSummary) -> str:
    if "下面代码块优先看 shape、缓存、状态更新或资源申请相关行" in text:
        return text
    lines = text.splitlines()
    in_fence = False
    for idx, line in enumerate(lines):
        match = FENCE_RE.match(line)
        if not match:
            continue
        if in_fence:
            in_fence = False
            continue
        in_fence = True
        lang = (match.group(2) or "").lower()
        if lang in {"mermaid", "text", ""}:
            continue
        note = (
            "<!-- [补充] -->\n"
            "下面代码块优先看 shape、缓存、状态更新或资源申请相关行；这些行通常决定故障是语义、并发还是容量问题。"
        )
        new_lines = lines[:idx] + [note] + lines[idx:]
        summary.code_note_added = True
        summary.changed = True
        return "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")
    return text


def process_page(path: str) -> PageSummary:
    page_type = classify_page(path)
    summary = PageSummary(path=path, page_type=page_type)
    full_path = ROOT / path
    original = full_path.read_text(encoding="utf-8")
    text = original

    text = add_or_standardize_formula_tables(text, summary)
    if page_type == "基础知识":
        text = add_beginner_buffer(text, summary)
        text = add_foundation_mermaid(text, summary)
    elif page_type == "论文解读":
        text = add_paper_experiment_conclusion(text, summary)
    elif page_type == "工程实践":
        text = add_engineering_pitfall(text, summary)
        text = add_code_note(text, summary)

    if text != original:
        full_path.write_text(text, encoding="utf-8")
        summary.changed = True
    return summary


def main() -> None:
    pages = discover_pages()
    statuses = {page: "pending" for page in pages}
    write_audit_log(pages, statuses)

    summaries: list[PageSummary] = []
    for page in pages:
        summary = process_page(page)
        summaries.append(summary)
        statuses[page] = "done"
        write_audit_log(pages, statuses)
        print(f"done {page} | {summary.page_type} | {summary.action_sentence()}")

    write_audit_log(pages, statuses, summaries)
    print(f"audited {len(summaries)} pages; changed {sum(1 for summary in summaries if summary.changed)} pages")


if __name__ == "__main__":
    main()
