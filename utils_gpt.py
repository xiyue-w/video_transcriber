import os
import json
from openai import OpenAI


def get_openai():
    with open("UserData/my_api_key.txt") as f:
        api_key = f.read().strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)
    return client

# client = OpenAI()  # uses OPENAI_API_KEY env var
client = get_openai()  # alternative: read from file

def gpt_suggest_labels(
    segment_text: str,
    turn_options: list[str],
    theme_options: list[str],
) -> tuple[str, str, str]:
    """
    Use GPT to suggest:
      - turn_type: one of turn_options
      - theme_code: one of theme_options
      - reasoning: short explanation for log/debug

    Returns: (turn_type, theme_code, reasoning)
    """
    segment_text = (segment_text or "").strip()
    if not segment_text:
        print("No segment text provided for GPT suggestion.")
        return "", "", "No text provided for GPT suggestion."

    # Safety: if lists are empty, GPT can't choose
    if not turn_options:
        turn_options = ["Question", "Answer", "Other"]
    if not theme_options:
        theme_options = ["Misc"]

    # Build a compact prompt
    system_msg = {
        "role": "system",
        "content": (
            "You are an assistant helping to label interview transcript segments. "
            "Based on the provided segment text, suggest the most appropriate turn type and theme code."
            "If your answer is similar to one of the provided options in turn_type or theme_code, pick it exactly. "
            "Otherwise, create a new label. "
            "Respond in JSON format with keys: turn_type, theme_code, reasoning."
        ),
    }

    user_msg = {
        "role": "user",
        "content": json.dumps(
            {
                "segment_text": segment_text,
                "turn_type_options": turn_options,
                "theme_code_options": theme_options,
            },
            ensure_ascii=False,
        ),
    }

    # Ask for a JSON object back
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # or gpt-4.1, gpt-4o-mini, etc.
        response_format={"type": "json_object"},
        messages=[system_msg, user_msg],
        temperature=0.2,
    )

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if something weird happens
        return "", "", f"Failed to parse GPT response: {raw}"

    turn_type = data.get("turn_type", "")
    theme_code = data.get("theme_code", "")
    reasoning = data.get("reasoning", "")

    # Basic sanity: enforce membership in provided lists
    if turn_type not in turn_options:
        turn_type = ""
    if theme_code not in theme_options:
        theme_code = ""

    return turn_type, theme_code, reasoning

def gradio_suggest_labels_with_display(segment_text, turn_options, theme_options):
    """
    Wrapper for Gradio:
      - segment_text: str from text_tb
      - turn_options: list from turns_state
      - theme_options: list from themes_state

    Returns:
      - update for turn_dd
      - update for theme_dd
      - log text (append or replace)
    """

    turn_type, theme_code, reasoning = gpt_suggest_labels(
        segment_text=segment_text,
        turn_options=turn_options,
        theme_options=theme_options,
    )

    ## debug: mock response
    # turn_type, theme_code, reasoning = "turn_type", "theme_code", "GPT suggestion not implemented yet."

    if not turn_type and not theme_code:
        msg = "GPT could not suggest labels. " + (reasoning or "")
    else:
        msg = (
            f"GPT suggestion:\n"
            f"  turn_type: {turn_type or '(none)'}\n"
            f"  theme_code: {theme_code or '(none)'}\n"
            f"Reasoning: {reasoning}"
        )

    return (
        turn_type,          # suggested_turn Textbox
        theme_code,         # suggested_theme Textbox
        reasoning,          # reasoning Textbox
        msg                 # log_box
    )