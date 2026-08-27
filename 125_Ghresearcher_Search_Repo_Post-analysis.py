# 搭配ghresearcher search模块+定制yaml搜索配置
# 管道接CLI 翻译工具 https://github.com/soimort/translate-shell

#!/usr/bin/env python3
"""Generate a weekly ghresearcher CSV report with a Chinese translation column.

Robustness guarantees:
- Data is extracted *structurally*: the result array is taken from between the
  first '[' and the last ']' in ghresearcher's stdout and parsed with
  ast.literal_eval. The tool's "Searching repos for ..." / "Running command: gh
  ..." info lines never corrupt the CSV, regardless of their order or count.
- Translation never throws on a single row. It uses an engine fallback chain
  (google -> bing -> mymemory), per-call retries with backoff, a global cooldown
  when many failures suggest rate limiting, and parallel workers. Rows that
  cannot be translated are marked "[translate failed]" and the file is still
  written.
- A search failure (e.g. GitHub rate limit) aborts loudly so the workflow
  surfaces the problem instead of committing empty/broken files.
- The CSV columns follow the `json` list in the search config, so editing the
  config's fields automatically updates the report schema.

Usage:
  python3 scripts/weekly_report.py <name> <config.yaml> <out.csv> \
      [--updated ">=YYYY-MM-DD"] [--trans PATH] [--workers N] [--tz TZ]
Defaults can also come from the TRANS and UPDATED environment variables.
"""
import argparse
import ast
import csv
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml

# None -> translate-shell's default engine (google); the rest are free fallbacks.
ENGINES = [None, "bing", "mymemory"]
FAIL_MARK = "[translate failed]"
# Columns carrying UTC ISO timestamps, converted to the report timezone in the CSV.
DATE_FIELDS = {"createdAt", "pushedAt", "updatedAt"}
_MAX_ATTEMPTS_PER_ENGINE = 2
_COOLDOWN_THRESHOLD = 8
_COOLDOWN_SECS = 20

_fail_count = 0
_fail_lock = threading.Lock()


def _call_trans(trans, text, engine):
    args = [trans, "-e", engine, "-brief", ":zh"] if engine else [trans, "-brief", ":zh"]
    r = subprocess.run(args + [text], capture_output=True, text=True, timeout=30)
    out = r.stdout.strip().replace("\n", " ")
    # Empty result, or the engine echoing the input back, counts as a failure.
    if r.returncode != 0 or not out or out == text:
        return ""
    return out


def _translate(trans, text):
    global _fail_count
    text = (text or "").strip()
    if not text:
        return ""
    for engine in ENGINES:
        for attempt in range(_MAX_ATTEMPTS_PER_ENGINE):
            try:
                out = _call_trans(trans, text, engine)
            except Exception:
                out = ""
            if out:
                with _fail_lock:
                    _fail_count = 0
                return out
            time.sleep(1 + 2 * attempt)          # backoff within one engine
        time.sleep(1)                            # brief pause before next engine
    with _fail_lock:
        _fail_count += 1
        n = _fail_count
    if n and n % _COOLDOWN_THRESHOLD == 0:
        time.sleep(_COOLDOWN_SECS)               # let rate limits reset
    return ""


def run_search(config_path, updated):
    cmd = ["ghresearcher", "search", "--config", config_path]
    if updated:
        cmd += ["--updated", updated]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"ghresearcher search failed (exit {proc.returncode}).\n"
            f"stderr: {proc.stderr.strip()[:2000]}"
        )
    start, end = proc.stdout.find("["), proc.stdout.rfind("]")
    if start == -1 or end <= start:
        raise SystemExit(
            "Could not find a result array in ghresearcher stdout.\n"
            f"stdout: {proc.stdout[:2000]}"
        )
    return ast.literal_eval(proc.stdout[start:end + 1])


def _to_local(value, tz):
    """Convert a UTC ISO timestamp (…Z) to the report timezone; leave non-dates as-is."""
    if not value:
        return value
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return value
    return dt.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S")


def load_columns(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    columns = list(cfg.get("json") or [])
    if "description" not in columns:
        columns.append("description")
    return columns


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name")
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--updated", default=os.environ.get("UPDATED"))
    ap.add_argument("--trans", default=os.environ.get("TRANS", "trans"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tz", default="Asia/Shanghai",
                    help="timezone for createdAt/pushedAt/updatedAt columns (IANA name)")
    args = ap.parse_args()

    try:
        tz = ZoneInfo(args.tz)
    except Exception as e:
        raise SystemExit(f"invalid --tz '{args.tz}': {e}")

    if not os.path.isfile(args.trans):
        raise SystemExit(f"translate-shell not found at {args.trans} (set TRANS)")

    records = run_search(args.config, args.updated)
    columns = load_columns(args.config)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(columns + ["description_zh"])
        if not records:
            print(f"{args.name}: 0 results -> {args.out} (header only)")
            return
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(_translate, args.trans, r.get("description", "")): r
                for r in records
            }
            for fut in as_completed(futures):
                futures[fut]["_zh"] = fut.result()
        ok = 0
        for r in records:
            zh = r.get("_zh", "")
            if zh:
                ok += 1
            row = [r.get(c, "") for c in columns]
            for i, c in enumerate(columns):
                if c in DATE_FIELDS:
                    row[i] = _to_local(row[i], tz)
            # Empty description -> keep the cell empty; only mark real failures.
            mark = FAIL_MARK if (r.get("description", "") and not zh) else ""
            wr.writerow(row + [(zh or mark)])
        print(f"{args.name}: {len(records)} results, {ok} translated -> {args.out}")


if __name__ == "__main__":
    main()
