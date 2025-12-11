import os

import gradio as gr
import whisper
import pandas as pd

# -------------------------------
# 1. Load Whisper model. 
# TODO: Change model selection to a Gradio UI.
# -------------------------------
# You can change "small" to "base", "medium", "large", etc.
MODEL_NAME = "large"
print(f"Loading Whisper model: {MODEL_NAME}")
model = whisper.load_model(MODEL_NAME)

# -------------------------------
# 2. Core functions
# -------------------------------
from utils_core import transcribe_video, dataframe_to_srt,export_outputs
from utils_core import apply_labels_to_row, load_row
from utils_core import combine_segments, split_segment
from utils_core import USER_FIELDS, DF_COLUMNS, DF_TYPES
import utils_core
utils_core.model = model

# ----- helper: generic "add new label" function -----
def add_label(label_list, new_label):
    new_label = (new_label or "").strip()
    if new_label and new_label not in label_list:
        label_list = label_list + [new_label]
    # update dropdown choices and keep current value = new_label
    return gr.update(choices=label_list, value=new_label), label_list, ""

# -------------------------------
# 3. Build Gradio UI
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown(
        """
    # 🎧 Video Transcriber & Labeler (Whisper + Gradio)

    1. Upload a video  
    2. Click **Transcribe**  
    3. Edit transcript and add `speaker` / `tag` labels  
    4. Export as SRT or JSON  
    """
    )

    # Original states to keep lists of choices for dropdowns
    speakers_state = gr.State(["Interviewer", "P1", "P2"])
    turns_state = gr.State(["Answer", "Question", "Other"])
    themes_state = gr.State(["NA", "Intro", "Motivation", "Obstacle", "Design feedback"])

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload video", sources=["upload"])

            transcribe_btn = gr.Button("🔁 Transcribe with Whisper")

            gr.Markdown("### Export")
            export_btn = gr.Button("📤 Export SRT & JSON")

        with gr.Column(scale=2):
            full_transcript = gr.Textbox(
                label="Full transcript (editable)",
                lines=5,
                interactive=True,
                placeholder="Full transcript will appear here after transcription.",
            )

            segments_df = gr.DataFrame(
                headers=DF_COLUMNS,
                datatype=DF_TYPES,
                col_count=(len(DF_COLUMNS), "fixed"),
                wrap=True,
                interactive=True,
                label="Segments (edit & label)",
            )

    # Right panel: labeling UI
    gr.Markdown("## 🏷 Label selected segment")

    with gr.Row():
        with gr.Column(scale=1):
            row_index = gr.Number(
                label="Row index (0-based)",
                value=0,
                precision=0
            )

            text_tb = gr.Textbox(
                label="Transcribed text",
                lines=3,
                interactive=True
            )

            speaker_dd = gr.Dropdown(
                label="Speaker",
                choices=speakers_state.value,
                interactive=True
            )
            turn_dd = gr.Dropdown(
                label="Turn type",
                choices=turns_state.value,
                interactive=True
            )
            theme_dd = gr.Dropdown(
                label="Theme code",
                choices=themes_state.value,
                interactive=True
            )
            notes_tb = gr.Textbox(
                label="Notes",
                lines=3,
                interactive=True
            )

            apply_btn = gr.Button("✅ Apply labels to this row")

        with gr.Column(scale=1):
            gr.Markdown("### Manage label sets")

            new_speaker = gr.Textbox(label="New speaker")
            add_speaker_btn = gr.Button("Add speaker")

            new_turn = gr.Textbox(label="New turn type")
            add_turn_btn = gr.Button("Add turn type")

            new_theme = gr.Textbox(label="New theme code")
            add_theme_btn = gr.Button("Add theme code")

    with gr.Row():
        srt_output = gr.Textbox(
            label="SRT output",
            lines=10,
            interactive=False,
            placeholder="Click 'Export' to generate SRT from the edited segments."
        )

        json_output = gr.Textbox(
            label="JSON output",
            lines=10,
            interactive=False,
            placeholder="Click 'Export' to generate JSON from the edited segments."
        )

    # --- Wire callbacks ---

    # 1. Transcription
    transcribe_btn.click(
        fn=transcribe_video,
        inputs=[video_input],
        outputs=[full_transcript, segments_df],
    )

    # 2. Export SRT & JSON based on current DataFrame (user-edited)
    export_btn.click(
        fn=export_outputs,
        inputs=segments_df,
        outputs=[srt_output, json_output],
    )

    # 3. apply labels to selected row in DataFrame
    apply_btn.click(
        fn=apply_labels_to_row,
        inputs=[segments_df, row_index, text_tb, speaker_dd, turn_dd, theme_dd, notes_tb],
        outputs=segments_df
    )

    # 4) add new speaker
    add_speaker_btn.click(
        fn=add_label,
        inputs=[speakers_state, new_speaker],
        outputs=[speaker_dd, speakers_state, new_speaker]
    )

    # 5) add new turn type
    add_turn_btn.click(
        fn=add_label,
        inputs=[turns_state, new_turn],
        outputs=[turn_dd, turns_state, new_turn]
    )

    # 6) add new theme
    add_theme_btn.click(
        fn=add_label,
        inputs=[themes_state, new_theme],
        outputs=[theme_dd, themes_state, new_theme]
    )

    # 6) Combine / Split segments
    combine_indices = gr.Textbox(label="Rows to combine (comma-separated)")
    combine_button = gr.Button("Combine segments")

    split_index = gr.Number(label="Row to split (index)", precision=0)
    split_char = gr.Number(label="Split at character index", precision=0)
    split_button = gr.Button("Split segment")

    combine_button.click(
    fn=combine_segments,
    inputs=[segments_df, combine_indices],
    outputs=segments_df
    )

    split_button.click(
        fn=split_segment,
        inputs=[segments_df, split_index, split_char],
        outputs=segments_df
    )

    # 7) Correlate dataframe selection with label panel and video
    current_time = gr.Number(visible=False)
    segments_df.select(
        fn=load_row,
        inputs=segments_df,
        outputs=[row_index, text_tb, speaker_dd, turn_dd, theme_dd, notes_tb, current_time]
    )

    # Click segment: also seek the video
    current_time.change(
        fn=lambda t: None,  # no Python work
        inputs=current_time,
        outputs=None,
        js="""
    (value) => {
    const video = document.querySelector("video");
    if (video && typeof value === "number") {
        video.currentTime = value;
    }
    }
    """
    )

# -------------------------------
# 4. Launch
# -------------------------------

if __name__ == "__main__":
    # share=True if you want a public link
    demo.launch()
