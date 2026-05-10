"""
Sub Vocal Synthesizer - Gradio Web UI
=======================================

Interactive web interface for testing the trained lip-to-speech models.
"""

import os
import gradio as gr
import logging
from sub_vocal.pipeline import LipToSpeechPipeline, PipelineConfig

# Auto-set mediapipe model path relative to project root
import pathlib
project_root = pathlib.Path(__file__).parent.parent
model_path = project_root / 'checkpoints' / 'mediapipe' / 'face_landmarker.task'
if model_path.exists() and 'MEDIAPIPE_MODEL_PATH' not in os.environ:
    os.environ['MEDIAPIPE_MODEL_PATH'] = str(model_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video(input_video, target_lang, tts_backend):
    if input_video is None:
        return None, "Please upload a video.", ""
        
    try:
        config = PipelineConfig(
            language=target_lang,
            output_path="synthesized_output.wav",
            tts_backend=tts_backend,
        )
        
        pipeline = LipToSpeechPipeline(config)
        result = pipeline.run(input_video)
        
        if result.success:
            return result.output_path, "Synthesis complete!", result.decoded_text
        else:
            return None, f"Synthesis failed: {', '.join(result.errors)}", ""
            
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return None, f"Error: {str(e)}", ""

# Define Gradio Interface
with gr.Blocks(title="Sub Vocal Speech Synthesizer") as app:
    gr.Markdown("# 🗣️ Sub Vocal Speech Synthesizer")
    gr.Markdown("Upload a silent video of someone speaking, and the AI will synthesize what they are saying based entirely on their lip movements! Uses ResNet18 + TCN Feature extraction.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Data")
            input_video = gr.Video(label="Input Silent Video (MP4)")
            
            gr.Markdown("### Options")
            target_lang = gr.Dropdown(choices=["en", "hi", "fr", "es", "de"], value="en", label="Target Language")
            tts_backend = gr.Dropdown(choices=["auto", "mms", "xtts"], value="auto", label="TTS Backend")
            
            synth_btn = gr.Button("Synthesize Speech", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Output")
            output_audio = gr.Audio(label="Synthesized Output Audio")
            output_text = gr.Textbox(label="Predicted/Translated Text", interactive=False)
            synth_status = gr.Textbox(label="Status", interactive=False)

    # Event handlers
    synth_btn.click(
        fn=process_video, 
        inputs=[input_video, target_lang, tts_backend], 
        outputs=[output_audio, synth_status, output_text]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
