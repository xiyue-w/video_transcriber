import pandas as pd
from typing import List, Tuple, Any
import json
import gradio as gr
import os

model: Any = None

# -----------------------------------------
# CENTRAL SCHEMA (define once here)
# -----------------------------------------
USER_FIELDS = {
    "speaker": "",
    "turn_type": "",
    "theme_code": "",
    "notes": "",
}

# All columns in order:
DF_COLUMNS = [
    "segment_id", "start", "end", "text",
    *USER_FIELDS.keys()   # expands to speaker, turn_type, theme_code, notes
]

DF_TYPES = [
    "number", "number", "number", "str",
    *["str"] * len(USER_FIELDS)         # one str for each user field
]

# -----------------------------------------
# Core functions
# -----------------------------------------

def transcribe_video(video_path: str) -> Tuple[str, pd.DataFrame]:
    """
    Run Whisper on the video and return:
      - full transcript text
      - DataFrame of segments with editable columns (speaker, tags)
    """
    global model
    if video_path is None:
        return "", pd.DataFrame(
            columns=["segment_id", "start", "end", "text", "speaker", "tag"]
        )

    print(f"Transcribing: {video_path}")
    result = model.transcribe(video_path, verbose=False)

    full_text = result.get("text", "").strip()
    segments = result.get("segments", [])
    full_text_with_punctuation = ""

    rows = []
    for seg in segments:
        seg_text = seg.get("text", "").strip(),
        seg_text = ensure_sentence_punctuation_and_space(seg_text)
        rows.append(
            {
                "segment_id": seg.get("id"),
                "start": round(seg.get("start", 0.0), 3),
                "end": round(seg.get("end", 0.0), 3),
                "text": seg_text,
                ## User-editable fields:
                # "speaker": "",
                # "tag": "",
                **USER_FIELDS
            }
        )
        full_text_with_punctuation += seg_text

    df = pd.DataFrame(rows, columns=DF_COLUMNS)
    save_msg = save_dataframe(df)
    return full_text_with_punctuation, df, save_msg

def dataframe_to_srt(df: pd.DataFrame) -> str:
    """
    Convert a segments DataFrame to an SRT string.
    Uses edited 'text' and ignores empty rows.
    """
    if df is None or df.empty:
        return ""

    def format_ts(t: float) -> str:
        # Format seconds as "HH:MM:SS,mmm"
        hours = int(t // 3600)
        minutes = int((t % 3600) // 60)
        seconds = int(t % 60)
        millis = int(round((t - int(t)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    srt_lines = []
    # Ensure sorted by start time (optional)
    df_sorted = df.sort_values(by="start")

    index = 1
    for _, row in df_sorted.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue

        start = float(row.get("start", 0))
        end = float(row.get("end", 0))

        # Optionally include speaker/tag in the SRT text
        # speaker = str(row.get("speaker", "")).strip()
        # tag = str(row.get("tag", "")).strip()
        labels = {field: str(row.get(field, "")).strip() for field in USER_FIELDS.keys()}

        header_parts = []
        for field, value in labels.items():
            header_parts.append(f"[{value}]")

        if header_parts:
            final_text = " ".join(header_parts) + " " + text
        else:
            final_text = text

        srt_lines.append(str(index))
        srt_lines.append(f"{format_ts(start)} --> {format_ts(end)}")
        srt_lines.append(final_text)
        srt_lines.append("")  # blank line between entries
        index += 1

    return "\n".join(srt_lines)

def export_outputs(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Export the current segments DataFrame as:
      - SRT text
      - JSON string (including labels)
    """
    if df is None or df.empty:
        return "", ""

    srt_text = dataframe_to_srt(df)

    # Convert DataFrame to JSON list of dicts
    json_records = df.to_dict(orient="records")
    json_text = json.dumps(json_records, indent=2, ensure_ascii=False)

    return srt_text, json_text

# -----------------------------------------
# Helper: apply dropdown labels to selected row
# -----------------------------------------
NONE_OPTION: Any = ""

def apply_labels_to_row(df, row_idx, text, speaker, turn_type, theme_code, notes):
    if df is None or isinstance(df, str) or len(df) == 0:
        return df
    print("speaker, turn_type, theme_code")
    print(speaker, turn_type, theme_code)

    try:
        idx = int(row_idx)
    except (TypeError, ValueError):
        return df

    if idx < 0 or idx >= len(df):
        return df

    df = pd.DataFrame(df).copy()
    # Update text if provided
    if text is not None:
        df.at[idx, "text"] = text
    if speaker and speaker != NONE_OPTION:
        df.at[idx, "speaker"] = speaker
    if turn_type and turn_type != NONE_OPTION:
        df.at[idx, "turn_type"] = turn_type
    if theme_code and theme_code != NONE_OPTION:
        df.at[idx, "theme_code"] = theme_code
    if notes is not None:
        df.at[idx, "notes"] = notes

    save_msg = save_dataframe(df)

    return df, save_msg

# -----------------------------------------
# Combine multiple segments into one and 
# split one to many
# -----------------------------------------
def reindex_segment_ids(df):
    df = df.sort_values(by="start").reset_index(drop=True)
    df["segment_id"] = df.index  # 0,1,2,...
    return df

def parse_indices(indices_str, max_len=None):
    """
    Parse index expressions like:
        "3,4,5"
        "3-5"
        "1,2,4-7,10"
    
    Returns a sorted list of unique integers.
    If max_len is provided, filters out invalid indices.
    """
    if not indices_str:
        return []

    parts = indices_str.replace(" ", "").split(",")
    indices = set()

    for p in parts:
        if "-" in p:
            # Handle ranges like 3-5
            try:
                start, end = p.split("-")
                start, end = int(start), int(end)
                if start <= end:
                    for i in range(start, end + 1):
                        indices.add(i)
                else:
                    for i in range(end, start + 1):  # allow reversed 5-3
                        indices.add(i)
            except:
                continue
        else:
            # Handle single numbers
            try:
                indices.add(int(p))
            except:
                continue

    indices = sorted(indices)

    # Optional: filter out-of-range indices
    if max_len is not None:
        indices = [i for i in indices if 0 <= i < max_len]

    return indices

def combine_segments(df, indices_str):
    """
    Combine multiple segment rows into a single row.
    indices_str: e.g., "3,4,5", "3-5", "1,2,4-7,10"
    """
    if df is None or len(df) == 0:
        return df
    
    # Parse indices
    try:
        indices = parse_indices(indices_str, max_len=len(df))
    #     indices = sorted([int(x) for x in indices_str.split(",")])
    except:
        return df

    df = pd.DataFrame(df).copy()

    # Validate indices
    if any(i < 0 or i >= len(df) for i in indices):
        return df
    
    # Pull rows
    segs = df.iloc[indices]

    # Build merged row
    merged = {}
    merged["segment_id"] = segs["segment_id"].iloc[0]      # keep the first ID
    merged["start"] = segs["start"].min()
    merged["end"] = segs["end"].max()

    # Merge text (you can adjust delimiter)
    merged["text"] = " ".join(segs["text"])

    # For labels — choose the first non-empty or leave blank
    for field in USER_FIELDS.keys():
        nonempty = segs[field][segs[field] != ""]
        merged[field] = nonempty.iloc[0] if len(nonempty) else ""

    # Remove old rows
    df = df.drop(indices).reset_index(drop=True)

    # Insert merged row
    df.loc[len(df)] = merged
    df = df.sort_values(by="start").reset_index(drop=True)
    df = reindex_segment_ids(df)
    return df

def split_segment(df, row_idx, split_after_text):
    """
    Split a segment into two based on text index.
    Example:
        row_idx = 7
        split_char_index = 35
    """
    df = pd.DataFrame(df).copy()

    # Coerce start/end columns to numeric (handles strings from Gradio)
    if "start" in df.columns:
        df["start"] = pd.to_numeric(df["start"], errors="coerce")
    if "end" in df.columns:
        df["end"] = pd.to_numeric(df["end"], errors="coerce")

    try:
        row_idx = int(row_idx)
        # split_char_index = int(split_char_index)
    except:
        return df

    if row_idx < 0 or row_idx >= len(df):
        return df

    row = df.loc[row_idx].copy()
    text = str(row.get("text", ""))

    split_after_text = (split_after_text or "").strip()
    print(f"Splitting after text: '{split_after_text}'")
    if not split_after_text:
        # nothing provided → don't split
        return df
    # Find where the left side ends
    pos = text.find(split_after_text)
    print(f"Found split text at position: {pos}")
    if pos == -1:
        # If not found → do nothing
        return df
    
    split_char_index = pos + len(split_after_text)
    print(f"Calculated split_char_index: {split_char_index}")
    if split_char_index <= 0 or split_char_index >= len(text):
        return df

    # Create two new text parts
    text1 = text[:split_char_index].strip()
    text2 = text[split_char_index:].strip()
    print(f"Text part 1: '{text1}'")
    print(f"Text part 2: '{text2}'")

    # Safely get numeric times
    start_val = row.get("start", 0)
    end_val = row.get("end", start_val)

    try:
        start = float(start_val)
    except (TypeError, ValueError):
        start = 0.0

    try:
        end = float(end_val)
    except (TypeError, ValueError):
        end = start

    # Calculate midpoint timestamp heuristically (optional)
    mid_time = start + (end - start) / 2.0

    row1 = row.copy()
    row1["end"] = mid_time
    row1["text"] = text1

    row2 = row.copy()
    row2["start"] = mid_time
    row2["text"] = text2

    # Replace the selected row with these two
    df = df.drop(row_idx).reset_index(drop=True)
    df.loc[len(df)] = row1
    df.loc[len(df)] = row2

    df = df.sort_values(by="start").reset_index(drop=True)
    df = reindex_segment_ids(df)
    return df

# -----------------------------------------
# Click on a segment row → 
# label panel + (optional) video seek to that time
# -----------------------------------------

def load_row(df, evt: gr.SelectData):
    """
    When a row in segments_df is clicked, load that row into the label panel.
    """
    df = pd.DataFrame(df)
    idx = evt.index  # for DataFrame this should be the row index (0-based)
    if isinstance(idx, (list, tuple)):
        # for something like (row_idx, col_idx)
        row_idx = int(idx[0])
    else:
        row_idx = int(idx)

    row = df.loc[row_idx]

    return (
        row_idx,             # row_index
        row["text"],         # text_tb
        row.get("speaker", ""),
        row.get("turn_type", ""),
        row.get("theme_code", ""),
        row.get("notes", ""),
        float(row["start"])  # current_time for video seek
    )

# -----------------------------------------
# Save and load dataframe
# -----------------------------------------
import time
def get_save_path():
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_path = f"UserData/Saved/{timestamp}_saved_segments.json"
    return save_path

def save_dataframe(df):
    """Save DataFrame to JSON every time."""
    save_path = get_save_path()
    df = pd.DataFrame(df)
    df.to_json(save_path, orient="records", force_ascii=False, indent=2)
    return f"💾 Saved DataFrame to {save_path} ({len(df)} rows)."

def load_df_from_file(path):
    if not path:
        return pd.DataFrame(), "⚠️ No file selected."

    try:
        df = pd.read_json(path, orient="records")
        msg = f"📂 Loaded {path} ({len(df)} rows)."
    except Exception as e:
        df = pd.DataFrame()
        msg = f"❌ Failed to load {path}: {e}"

    return df, msg

import re

def ensure_sentence_punctuation_and_space(text: str) -> str:
    """
    Add a '.' to the end of the string *only if* it ends with:
      - no punctuation
      - no whitespace
    """
    if not text:
        return text
    text = normalize_text(text)
    text = text.strip()

    # Check if final character is already punctuation
    if re.match(r".*[.!?。！？]$", text):
        return text + " "  # already has punctuation

    # Otherwise add period
    return text + ". "

def normalize_text(x):
    """
    Whisper sometimes outputs ('text',) as a tuple.
    This function normalizes to a clean string.
    """
    if isinstance(x, tuple):
        x = " ".join(str(y) for y in x if y)
    return str(x)