import os

import gradio as gr
import whisper
import pandas as pd

# -------------------------------
# 01. Load Whisper model. 
# TODO: Change model selection to a Gradio UI.
# -------------------------------
# You can change "small" to "base", "medium", "large", etc.
MODEL_NAME_DEFAULT = "tiny"
current_model_name = MODEL_NAME_DEFAULT
model = whisper.load_model(MODEL_NAME_DEFAULT)

# -------------------------------
# 02. Core functions
# -------------------------------
from utils_core import transcribe_video, dataframe_to_srt,export_outputs
from utils_core import apply_labels_to_row, load_row
from utils_core import combine_segments, split_segment
from utils_core import USER_FIELDS, DF_COLUMNS, DF_TYPES
import utils_core
utils_core.model = model
from utils_gpt import gradio_suggest_labels_with_display
from utils_core import load_df_from_file

# ----- helper: generic "add new label" function -----
def add_label(label_list, new_label):
    new_label = (new_label or "").strip()
    if new_label and new_label not in label_list:
        label_list = label_list + [new_label]
    # update dropdown choices and keep current value = new_label
    return gr.update(choices=label_list, value=new_label), label_list, ""

# ----- helper: reload model -----
def reload_model(selected_name: str):
    """
    Load a new Whisper model when the user chooses from dropdown.
    """
    global model, current_model_name

    selected_name = selected_name.strip()
    if not selected_name:
        return "No model selected."

    if selected_name == current_model_name and model is not None:
        return f"Model '{selected_name}' is already loaded."

    # (Optional) free old model here, if you want to be explicit:
    # del model
    # torch.cuda.empty_cache()  # if you're using CUDA

    model = whisper.load_model(selected_name)
    utils_core.model = model
    current_model_name = selected_name
    return f"✅ Loaded Whisper model: {selected_name}"

# ----- helper: NONE_OPTION for the drop down menu -----
NONE_OPTION = ""
utils_core.NONE_OPTION = NONE_OPTION
# -------------------------------
# 03. Build Gradio UI
# -------------------------------
with gr.Blocks() as demo:
    # States for dropdown choices (you can tune defaults)
    speakers_state = gr.State(["Interviewer", "P1", "P2"])
    turns_state = gr.State(["Answer", "Question", "Other"])
    themes_state = gr.State(["NA"])
    # themes_state = gr.State(["NA", "Intro", "Motivation", "Obstacle", "Design feedback"])

    gr.Markdown(
        """
    # 🎧 Video Transcriber & Labeler (Whisper + Gradio)

    1. Upload a video  
    2. Click **Transcribe**  
    3. Edit transcript and add `speaker` / `tag` labels  
    4. Export as SRT or JSON  
    """
    )

    # =====================================================
    # 1. TOP ROW
    # =====================================================
    with gr.Row():
        # -----------------------------
        # 1.1 Left column: Video + basic controls
        # -----------------------------
        with gr.Column(scale=1):
            gr.Markdown("### 🎥 Video")
            video_input = gr.Video(
                label="Upload video", 
                sources=["upload"],
                interactive=True)

            transcribe_btn = gr.Button("🔁 Transcribe with Whisper")
            with gr.Row():
                current_time = gr.Number(
                    label="Current segment start (sec)",
                    value=0.0,
                    interactive=False,
                    precision=3,
                    min_width=0
                )
                # # originally:
                # current_time = gr.Number(visible=False)

                # model selection
                model_dropdown = gr.Dropdown(
                    label="Whisper model (load by selection)",
                    choices=["tiny", "base", "small", "medium", "large"],
                    value=MODEL_NAME_DEFAULT,
                    interactive=True,
                    min_width=0
                )

        # -----------------------------
        # 1.2 Middle column: Label & Edit Panel
        # -----------------------------
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Edit")
            with gr.Row():
                row_index = gr.Number(
                    label="Row index (0-based)",
                    value=0,
                    precision=0,
                    min_width=0
                )
                speaker_dd = gr.Dropdown(
                    label="Speaker",
                    choices=[NONE_OPTION] + speakers_state.value,
                    value=NONE_OPTION,
                    interactive=True,
                    min_width=0
                )
                turn_dd = gr.Dropdown(
                    label="Turn type",
                    choices=[NONE_OPTION] + turns_state.value,
                    value=NONE_OPTION,
                    interactive=True,
                    min_width=0
                )
                theme_dd = gr.Dropdown(
                    label="Theme code",
                    choices=[NONE_OPTION] + themes_state.value,
                    value=NONE_OPTION,
                    interactive=True,
                    min_width=0
                )

            text_tb = gr.Textbox(
                label="Transcribed text",
                lines=3,
                interactive=True
            )

            notes_tb = gr.Textbox(
                label="Notes",
                lines=1,
                interactive=True
            )

            apply_btn = gr.Button("✅ Apply labels to this row")
        # -----------------------------
        # 1.3 Right column: Segmentation / Manage labels / GPT (later)
        # -----------------------------
        with gr.Column(scale=1):
            # 1.3.2 Manage label sets
            gr.Markdown("### 🧱 Manage label sets")
            with gr.Row():
                with gr.Column(scale=1, min_width=0):
                    # new_speaker = gr.Textbox(label="New speaker")
                    new_speaker = gr.Textbox(show_label=False, placeholder="New speaker")
                    add_speaker_btn = gr.Button("Add speaker")
                with gr.Column(scale=1, min_width=0):
                    # new_turn = gr.Textbox(label="New turn type")
                    new_turn = gr.Textbox(show_label=False, placeholder="New turn type")
                    add_turn_btn = gr.Button("Add turn type")
                with gr.Column(scale=1, min_width=0):
                    # new_theme = gr.Textbox(label="New theme code")
                    new_theme = gr.Textbox(show_label=False, placeholder="New code")
                    add_theme_btn = gr.Button("Add theme code")

            # 1.3.3 GPT Suggested Labels (placeholder for later)
            gr.Markdown("### 🤖 GPT Suggested Labels")
            with gr.Row():
                suggested_turn = gr.Textbox(
                    label="Turn type",
                    interactive=False,
                    placeholder="Fill by GPT",
                    min_width=0
                )
                suggested_theme = gr.Textbox(
                    label="Theme code",
                    interactive=False,
                    placeholder="Fill by GPT",
                    min_width=0
                )
            reasoning = gr.Textbox(
                show_label=False,
                interactive=False,
                placeholder="Show GPT reasoning",)
            run_gpt_btn = gr.Button("Run GPT Suggestion", interactive=True)
    # =====================================================
    # 2. MIDDLE ROW: Segments DataFrame + full transcript
    # =====================================================
    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 🧩 Segments (click a row to load into label panel)")
            # segments_df = gr.DataFrame(
            #     label="Segments (click a row to load into label panel)",
            #     interactive=True,
            #     wrap=True
            # )

            # originally:
            segments_df = gr.DataFrame(
                headers=DF_COLUMNS,
                datatype=DF_TYPES,
                col_count=(len(DF_COLUMNS), "fixed"),
                wrap=True,
                interactive=True,
                # label="Segments (edit & label)",
                )
            
            load_file = gr.File(
                label="Load DataFrame from JSON",
                file_types=[".json"],  # or [".json", ".csv"]
                type="filepath"        # important: we want a path string
            )

            load_df_btn = gr.Button("📂 Load saved DataFrame")

        with gr.Column(scale=1):
        # 1.3.1 Segmentation
            gr.Markdown("### ✂️ Segmentation")
            # with gr.Row():
            #     with gr.Column(scale=1, min_width=0):
            combine_indices = gr.Textbox(
                label="Rows to combine \n(comma or dash-separated)",
                placeholder="e.g., 3,4,5 or 3-5 or 1,2,4-7,10"
            )
            combine_btn = gr.Button("🔗 Combine segments")
                # with gr.Column(scale=1, min_width=0):
            with gr.Row():
                split_index = gr.Number(
                    label="Row to split\n(index)",
                    precision=0,
                    min_width=0
                )
                # split_char = gr.Number(
                #     label="At character index",
                #     precision=0,
                #     min_width=0
                # )
                split_text = gr.Textbox(
                label="Split after text",
                placeholder="Paste the end part of the text that should become segment 1",
                min_width=0
                )
            split_btn = gr.Button("🔪 Split segment")
    with gr.Row():
        with gr.Column(scale=1):
            full_transcript = gr.Textbox(
                label="All transcribed text",
                lines=3,
                interactive=True
            )

    # =====================================================
    # 3. BOTTOM ROW: Export + Log
    # =====================================================
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📤 Export")
            export_btn = gr.Button("Export SRT + JSON")
            with gr.Row():
                srt_output = gr.Textbox(
                    label="SRT output",
                    lines=5,
                    interactive=False,
                    placeholder="Click 'Export' to generate SRT from the edited segments.",
                    min_width=0
                )
                json_output = gr.Textbox(
                    label="JSON output",
                    lines=5,
                    interactive=False,
                    placeholder="Click 'Export' to generate JSON from the edited segments.",
                    min_width=0
                )

        with gr.Column(scale=1):
            gr.Markdown("### 🪵 Log / Status")
            log_box = gr.Textbox(
                label="Log",
                lines=8,
                interactive=False
            )
    # =====================================================
    # Wire callbacks
    # =====================================================

    # 1.1 Transcription
    transcribe_btn.click(
        fn=transcribe_video,
        inputs=[video_input],
        outputs=[full_transcript, segments_df, log_box],
    )
    model_dropdown.change(
        fn=reload_model,
        inputs=model_dropdown,
        outputs=log_box,   # your status/log Textbox
    )

    # 1.2 apply labels to selected row in DataFrame
    apply_btn.click(
        fn=apply_labels_to_row,
        inputs=[segments_df, row_index, text_tb, speaker_dd, turn_dd, theme_dd, notes_tb],
        outputs=[segments_df, log_box]
    )

    # 1.3.1 Combine / Split segments
    combine_btn.click(
    fn=combine_segments,
    inputs=[segments_df, combine_indices],
    outputs=segments_df
    )
    split_btn.click(
        fn=split_segment,
        inputs=[segments_df, split_index, split_text],
        outputs=segments_df
    )

    # 1.3.2 add labels 
    # 1.3.2.1 add new speaker
    add_speaker_btn.click(
        fn=add_label,
        inputs=[speakers_state, new_speaker],
        outputs=[speaker_dd, speakers_state, new_speaker]
    )
    # 1.3.2.2 add new turn type
    add_turn_btn.click(
        fn=add_label,
        inputs=[turns_state, new_turn],
        outputs=[turn_dd, turns_state, new_turn]
    )
    # 1.3.2.3 add new theme
    add_theme_btn.click(
        fn=add_label,
        inputs=[themes_state, new_theme],
        outputs=[theme_dd, themes_state, new_theme]
    )

    # 1.3.3 GPT Suggested Labels
    run_gpt_btn.click(
        fn=gradio_suggest_labels_with_display,
        inputs=[text_tb, turns_state, themes_state],
        outputs=[suggested_turn, suggested_theme, reasoning, log_box],
    )

    # 2. Correlate dataframe selection with label panel and video
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
    load_df_btn.click(
        fn=load_df_from_file,
        inputs=load_file,
        outputs=[segments_df, log_box],
    )

    # 3. Export SRT & JSON based on current DataFrame (user-edited)
    export_btn.click(
        fn=export_outputs,
        inputs=segments_df,
        outputs=[srt_output, json_output],
    )

# -------------------------------
# 4. Launch
# -------------------------------

if __name__ == "__main__":
    # share=True if you want a public link
    demo.launch()
