#!/usr/bin/env python3
"""
memory_ingest.py — Lumen Memory Ingestion Tool
Processes ChatGPT export JSON into structured .md memory files.
Uses Featherless API (OpenAI-compatible) for extraction.

Usage:
    python memory_ingest.py --input conversations.json
    python memory_ingest.py --input conversations.json --model Qwen/Qwen2.5-14B-Instruct
    python memory_ingest.py --input conversations.json --limit 10  # test run
"""

import os
import re
import json
import time
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Config -----------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent.parent / "memory" / "ingested"
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"

EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction assistant. Your job is to analyse a conversation and extract structured metadata from it.

You will output ONLY valid JSON with no preamble, no markdown fences, no explanation. Just the raw JSON object.

Schema:
{
  "dates": ["YYYY-MM-DD"],
  "source": "chatgpt",
  "gpt_model": "GPT 5.2",
  "projects": [],
  "tags": [],
  "confidence": "high|medium|low",
  "summary": "One concise paragraph describing what this conversation was actually about and any key outcomes."
}

Guidelines:
- dates: use the conversation's actual date(s), ISO format
- source: always "chatgpt" for this pipeline
- gpt_model: the GPT model used, provided in the prompt header — use it exactly as given
- projects: only include if clearly relevant: lumen, dexter, mulligan, eridu, hydra, meridian, remy
- tags: 3-8 tags that best describe topics, tools, and themes. Use kebab-case.
- confidence: your confidence in the extraction quality. "low" if conversation is too short or ambiguous.
- summary: focus on what was decided, built, or learned — not just what was discussed."""

EXTRACTION_USER_TEMPLATE = """Extract structured metadata from this conversation:

TITLE: {title}
DATE: {date}
MODEL: {gpt_model}
{attachments_note}
MESSAGES:
{messages}"""


# --- Logging ----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("memory_ingest")


# --- Featherless Client -----------------------------------------------

def get_client() -> OpenAI:
    api_key = os.environ.get("FEATHERLESS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FEATHERLESS_API_KEY not set. Export it before running:\n"
            "  export FEATHERLESS_API_KEY=your_key_here"
        )
    return OpenAI(base_url=FEATHERLESS_BASE_URL, api_key=api_key)


# --- GPT Export Parsing -----------------------------------------------

def load_conversations(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    log.info(f"Loaded {len(data)} conversations from {path.name}")
    return data


def format_model_slug(slug: str) -> str:
    """Convert default_model_slug to a readable label."""
    if not slug or slug == "auto":
        return "GPT"
    # gpt-5-2 -> GPT 5.2, gpt-4o -> GPT 4o, etc.
    name = slug.replace("gpt-", "GPT ").replace("-", ".")
    return name


def extract_messages(conversation: dict) -> tuple[str, str, bool]:
    """
    Flatten conversation messages into a readable transcript.
    Returns (llm_transcript, formatted_transcript, had_attachments).
    llm_transcript: plain text for the extraction LLM (capped)
    formatted_transcript: dated, labelled lines for the MD file (full)
    Sorts by message create_time for correct chronological order.
    Dict parts (images/files) are skipped but flagged.
    """
    entries = []
    had_attachments = False
    mapping = conversation.get("mapping", {})
    model_label = format_model_slug(conversation.get("default_model_slug", ""))

    for node in mapping.values():
        msg = node.get("message")
        if not msg:
            continue
        role = msg.get("author", {}).get("role", "unknown")
        if role not in ("user", "assistant"):
            continue

        ts = msg.get("create_time") or 0
        content = msg.get("content", {})
        parts = content.get("parts", [])

        text_parts = []
        for p in parts:
            if isinstance(p, str):
                text_parts.append(p)
            elif isinstance(p, dict):
                had_attachments = True

        text = " ".join(text_parts).strip()
        if not text:
            continue

        if role == "user":
            label = "User"
        else:
            label = model_label

        entries.append((ts, role, label, text))

    # Sort chronologically
    entries.sort(key=lambda x: x[0])

    # LLM transcript — capped, plain
    llm_lines = []
    for ts, role, label, text in entries[:40]:
        llm_lines.append(f"{label}: {text[:1500]}")
    llm_transcript = "\n\n".join(llm_lines)

    # Formatted transcript — full, dated
    fmt_lines = []
    for ts, role, label, text in entries:
        if ts:
            try:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            except (ValueError, OSError):
                dt = None
            timestamp = dt.strftime("%d/%m/%y %H:%M") if dt else "unknown"
            fmt_lines.append(f"**{label} ({timestamp}):** {text}\n")
        else:
            fmt_lines.append(f"**{label}:** {text}\n")

    formatted_transcript = "\n".join(fmt_lines)

    return llm_transcript, formatted_transcript, had_attachments


def get_conversation_date(conversation: dict) -> str:
    ts = conversation.get("create_time")
    if ts:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    return datetime.now().strftime("%Y-%m-%d")


def get_conversation_title(conversation: dict) -> str:
    return conversation.get("title", "Untitled Conversation").strip()


# --- LLM Extraction ---------------------------------------------------

def extract_metadata(client: OpenAI, conversation: dict, model: str) -> tuple[dict, str] | tuple[None, None]:
    title = get_conversation_title(conversation)
    date = get_conversation_date(conversation)
    messages_text, formatted_transcript, had_attachments = extract_messages(conversation)

    if len(messages_text) < 100:
        log.warning(f"Skipping '{title}' — conversation too short")
        return None, None

    gpt_model = format_model_slug(conversation.get("default_model_slug", ""))
    attachments_note = "NOTE: This conversation contained image or file attachments (not included in transcript).\n" if had_attachments else ""

    user_prompt = EXTRACTION_USER_TEMPLATE.format(
        title=title,
        date=date,
        gpt_model=gpt_model,
        attachments_note=attachments_note,
        messages=messages_text
    )

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=1000,
            temperature=0.1,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw), formatted_transcript

    except json.JSONDecodeError as e:
        log.error(f"JSON parse failed for '{title}': {e}")
        return None, None
    except Exception as e:
        log.error(f"API error for '{title}': {e}")
        return None, None


# --- MD File Generation -----------------------------------------------

def build_filename(title: str, date: str) -> str:
    safe = re.sub(r"[^\w\s-]", "", title.lower())
    safe = re.sub(r"[\s_]+", "-", safe).strip("-")[:60]
    return f"{date}_{safe}.md"


def build_markdown(title: str, metadata: dict, formatted_transcript: str) -> str:
    dates_yaml = json.dumps(metadata.get("dates", []))
    projects = metadata.get("projects", [])
    tags = metadata.get("tags", [])
    confidence = metadata.get("confidence", "medium")
    summary = metadata.get("summary", "No summary extracted.")
    source = metadata.get("source", "chatgpt")
    gpt_model = metadata.get("gpt_model", "GPT")

    return f"""---
dates: {dates_yaml}
model: {gpt_model}
parsing_model: {DEFAULT_MODEL}
source: {source}
projects: {json.dumps(projects)}
tags: {json.dumps(tags)}
confidence: {confidence}
summary: {summary}
---

# {title}

## Summary

{summary}

## Transcript

{formatted_transcript}"""


def write_output(filename: str, content: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    out_path.write_text(content, encoding="utf-8")
    return out_path


# --- Progress Tracking ------------------------------------------------

def load_progress(progress_file: Path) -> set:
    if progress_file.exists():
        with open(progress_file) as f:
            return set(json.load(f))
    return set()


def save_progress(progress_file: Path, processed_ids: set):
    with open(progress_file, "w") as f:
        json.dump(list(processed_ids), f)


# --- Main -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest ChatGPT export into Lumen memory")
    parser.add_argument("--input", required=True, help="Path to conversations.json")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Featherless model to use")
    parser.add_argument("--limit", type=int, default=None, help="Max conversations to process (for testing)")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between API calls")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        return

    progress_file = OUTPUT_DIR / ".progress.json"
    processed_ids = load_progress(progress_file)

    client = get_client()
    conversations = load_conversations(input_path)

    remaining = [c for c in conversations if c.get("id") not in processed_ids]
    if args.limit:
        remaining = remaining[:args.limit]

    log.info(f"Processing {len(remaining)} conversations (skipping {len(processed_ids)} already done)")
    log.info(f"Model: {args.model} | Output: {OUTPUT_DIR}")

    success = 0
    skipped = 0
    failed = 0

    for i, conv in enumerate(remaining):
        title = get_conversation_title(conv)
        date = get_conversation_date(conv)
        conv_id = conv.get("id", f"unknown_{i}")

        log.info(f"[{i+1}/{len(remaining)}] {title[:60]}")

        metadata, formatted_transcript = extract_metadata(client, conv, args.model)

        if metadata is None:
            skipped += 1
            processed_ids.add(conv_id)
            save_progress(progress_file, processed_ids)
            continue

        filename = build_filename(title, date)
        content = build_markdown(title, metadata, formatted_transcript)

        try:
            out_path = write_output(filename, content)
            log.info(f"  -> {out_path.name}")
            success += 1
        except Exception as e:
            log.error(f"  Write failed: {e}")
            failed += 1

        processed_ids.add(conv_id)
        save_progress(progress_file, processed_ids)

        if i < len(remaining) - 1:
            time.sleep(args.delay)

    log.info(f"\nDone. Success: {success} | Skipped: {skipped} | Failed: {failed}")
    log.info(f"Files written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()