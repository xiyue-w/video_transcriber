```{=html}
<p align="center">
```
`<img src="https://img.shields.io/badge/Python-3.11-blue.svg" />`{=html}
`<img src="https://img.shields.io/badge/Gradio-4.x-orange.svg" />`{=html}
`<img src="https://img.shields.io/badge/Whisper-Transcription-green" />`{=html}
`<img src="https://img.shields.io/badge/License-MIT-green" />`{=html}
`<img src="https://img.shields.io/github/last-commit/yourusername/video-transcriber" />`{=html}
```{=html}
</p>
```
```{=html}
<h1 align="center">
```
🎥 Video Transcriber & Annotation Tool
```{=html}
</h1>
```
```{=html}
<p align="center">
```
A Gradio-based Whisper transcription and qualitative coding interface
with GPT-assisted labeling.
```{=html}
</p>
```

------------------------------------------------------------------------

## 🚀 Overview

This project is an interactive **video transcription and annotation
tool** designed for research workflows such as:

-   Interview speech transcription and analysis
-   Qualitative coding & thematic analysis

It combines **Whisper** for transcription and **GPT** for assistive
labeling in a clean Gradio web interface.

------------------------------------------------------------------------

## ✨ Key Features

### 🔊 Whisper Transcription

-   Supports Whisper models: `tiny`, `base`, `small`, `medium`, `large`
-   Model hot‑swapping via dropdown
-   Automatic punctuation normalization

### 🧩 Segment Editing

-   Click‑to‑edit transcript segments
-   Merge segments (`3,4,5` or `3‑5`)
-   Split segments by **pasting text to split after**
-   Automatic `segment_id` reindexing

### 🏷 Manual Labeling

-   Speaker
-   Turn type (Question / Answer / etc.)
-   Theme codes
-   Notes

### 🤖 GPT‑Assisted Suggestions

-   Suggests turn type & theme code
-   Explains reasoning
-   Now it remains advisory. Future work if needed: auto‑fill dropdowns

### 💾 Autosave & Load

-   Autosaves on transcription & label edits
-   Load sessions from JSON files

### 📤 Export

-   `.srt` subtitle file
-   `.json` structured annotation file

------------------------------------------------------------------------

## 🛠 Installation

- Note: ffmpeg needs to be installed on your system.

``` bash
git clone https://github.com/yourusername/video-transcriber.git
cd video-transcriber

conda create -n video_transcriber python=3.11
conda activate video_transcriber

pip install -r requirements.txt
```
------------------------------------------------------------------------

## ▶️ Usage

``` bash
python run_transcriber.py
```

Open your browser at:

    http://127.0.0.1:7860

------------------------------------------------------------------------

## 📁 Project Structure

    video-transcriber/
    ├── run_transcriber.py
    ├── utils_core.py
    ├── requirements.txt
    └── UserData/
        ├── my_api_key.txt (‼️Not included in the repository. You have to create one with GPT api key inside by yourself.)
        └── Saved/
            └──saved_segments.json

------------------------------------------------------------------------

## 📌 Notes

-   Designed for qualitative research workflows
-   Fully manual + AI‑assisted hybrid coding
-   Safe autosave prevents annotation loss

------------------------------------------------------------------------

## 🙏 Acknowledgments

- Whisper (OpenAI)
- Gradio
- OpenAI GPT APIs
- Contributions from HCI research workflows

------------------------------------------------------------------------


## 📝 License

MIT License
