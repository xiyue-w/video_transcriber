import pandas as pd
from typing import List, Tuple, Any
import json
import gradio as gr

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
    if video_path is None:
        return "", pd.DataFrame(
            columns=["segment_id", "start", "end", "text", "speaker", "tag"]
        )

    print(f"Transcribing: {video_path}")
    result = model.transcribe(video_path, verbose=False)

    full_text = result.get("text", "").strip()
    segments = result.get("segments", [])

    rows = []
    for seg in segments:
        rows.append(
            {
                "segment_id": seg.get("id"),
                "start": round(seg.get("start", 0.0), 3),
                "end": round(seg.get("end", 0.0), 3),
                "text": seg.get("text", "").strip(),
                ## User-editable fields:
                # "speaker": "",
                # "tag": "",
                **USER_FIELDS
            }
        )

    df = pd.DataFrame(rows, columns=DF_COLUMNS)
    return full_text, df

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
def apply_labels_to_row(df, row_idx, text, speaker, turn_type, theme_code, notes):
    if df is None or isinstance(df, str) or len(df) == 0:
        return df

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
    if speaker:
        df.at[idx, "speaker"] = speaker
    if turn_type:
        df.at[idx, "turn_type"] = turn_type
    if theme_code:
        df.at[idx, "theme_code"] = theme_code
    if notes is not None:
        df.at[idx, "notes"] = notes

    return df

# -----------------------------------------
# Combine multiple segments into one and 
# split one to many
# -----------------------------------------
def reindex_segment_ids(df):
    df = df.sort_values(by="start").reset_index(drop=True)
    df["segment_id"] = df.index  # 0,1,2,...
    return df

def combine_segments(df, indices_str):
    """
    Combine multiple segment rows into a single row.
    indices_str: e.g., "3,4,5"
    """
    if df is None or len(df) == 0:
        return df
    
    # Parse indices
    try:
        indices = sorted([int(x) for x in indices_str.split(",")])
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

def split_segment(df, row_idx, split_char_index):
    """
    Split a segment into two based on text index.
    Example:
        row_idx = 7
        split_char_index = 35
    """
    df = pd.DataFrame(df).copy()

    try:
        row_idx = int(row_idx)
        split_char_index = int(split_char_index)
    except:
        return df

    if row_idx < 0 or row_idx >= len(df):
        return df

    row = df.loc[row_idx].copy()
    text = row["text"]

    if split_char_index <= 0 or split_char_index >= len(text):
        return df

    # Create two new text parts
    text1 = text[:split_char_index].strip()
    text2 = text[split_char_index:].strip()

    # Calculate midpoint timestamp heuristically (optional)
    mid_time = (row["start"] + row["end"]) / 2

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
