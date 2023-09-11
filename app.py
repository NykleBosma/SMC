from faster_whisper import WhisperModel
import datetime
import csv
import chardet
import gradio as gr
import pandas as pd
import time
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

from gpuinfo import GPUInfo

import wave
import contextlib
import psutil


"""
The HuggingFace Gradio interface is based on the following example:
https://huggingface.co/spaces/vumichien/Whisper_speaker_diarization

The Whisper implementation is found here:
https://huggingface.co/openai/whisper-large-v2

The speaker diarization model and pipeline are found:
https://github.com/pyannote/pyannote-audio

"""

whisper_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2"]
source_languages = {
    "en": "English",
    "nl": "Dutch",
    "da": "Danish",
}
source_language_list = [key[0] for key in source_languages.items()]

#Loading the diarization model
embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def convert_time(secs):
    """
    Convert seconds to a datetime object.
    """
    return datetime.timedelta(seconds=round(secs))

def get_duration(audio_file):
    """
    Get the duration of an audio file.
    """
    with contextlib.closing(wave.open(audio_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration

def get_wav_files(folder_path=None):
    """
    Get all the .wav files in a folder.
    """
    if(folder_path == None):
        folder_path = os.path.dirname(os.path.abspath(__file__))
    wav_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            wav_files.append(file)
    return wav_files

def transcribe_audio(audio_file_path, selected_source_lang, whisper_model):
    """
    Transcribe an audio file using Whisper.
    """
    model = WhisperModel(whisper_model, device="cuda", compute_type="int8_float16")
    if(audio_file_path == None):
        # Take the sample file if none is selected
        audio_file = "sample1.wav"
    else:
        audio_file = audio_file_path

    try:
        options = dict(language=selected_source_lang, beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        segments_raw, info = model.transcribe(audio_file, **transcribe_options)

        segments = []
        for segment_chunk in segments_raw:
            chunk = {}
            chunk["start"] = segment_chunk.start
            chunk["end"] = segment_chunk.end
            chunk["text"] = segment_chunk.text
            segments.append(chunk)

        return segments
    except Exception as e:
        raise RuntimeError("Error while transcribing the audio")

def create_embeddings(audio_file_path, segments):
    """
    Create embeddings for the segments of an audio file.
    """
    #embedding_model = SpeechEmbedding()
    duration = get_duration(audio_file_path)

    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        audio = Audio()
        start = segment["start"]
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(audio_file_path, clip)
        embeddings[i] = embedding_model(waveform[None])
    embeddings = np.nan_to_num(embeddings)
    return embeddings

def assign_speaker_labels(embeddings, segments, num_speakers=0):
    """
    Assign speaker labels to the segments of an audio file based on the embeddings.
    """
    if num_speakers == 0:
        score_num_speakers = {}
        for num_speakers in range(2, 10+1):
            clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
            score = silhouette_score(embeddings, clustering.labels_, metric='euclidean')
            score_num_speakers[num_speakers] = score
        best_num_speaker = max(score_num_speakers, key=lambda x:score_num_speakers[x])
    else:
        best_num_speaker = num_speakers

    try:
        clustering = AgglomerativeClustering(best_num_speaker).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
    except:
        print(f'Could not assign labels, number of speakers: {best_num_speaker}')
    print(segments)
    return segments

def save_segments(segments):
    """
    Save the segments of an audio file to a CSV file.
    Use the Text Wizard in Excel when importing!
    """
    with open('output/segments.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Start', 'End', 'Speaker', 'Text'])
        for i, segment in enumerate(segments):
            writer.writerow([segment["start"], segment["end"], segment["speaker"], segment["text"]])

def read_segments(file_path):
    """
    Read the segments from a CSV file and return them as a dictionary.
    """
    print(type(file_path))
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    print(f"Encoding of {file_path} is {result['encoding']}")

    with open(file_path, 'r', newline='', encoding=result['encoding']) as f:
        reader = csv.reader(f)
        next(reader)  # skip the header row
        segments = []
        for row in reader:
            chunk = {}
            print(row)
            start, end, speaker, text = row
            chunk["start"] = float(start)
            chunk["end"] = float(end)
            chunk["speaker"] = speaker
            chunk["text"] = text
            segments.append(chunk)
    print(segments)
    return segments
def create_output(segments):
    """
    Create the final output from the segments of an audio file.
    """
    objects = {
        'Start': [],
        'End': [],
        'Speaker': [],
        'Text': []
    }
    text = ''
    # This function basically groups the segments by speaker
    # The objects dictionary is used to create the final output in the next step
    for (i, segment) in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            objects['Start'].append(str(convert_time(segment["start"])))
            objects['Speaker'].append(segment["speaker"])
            if i != 0:
                objects['End'].append(str(convert_time(segments[i - 1]["end"])))
                objects['Text'].append(text)
                text = ''
        text += segment["text"] + ' '
    objects['End'].append(str(convert_time(segments[i - 1]["end"])))
    objects['Text'].append(text)

    return objects

def speech_to_text(audio_file_path, selected_source_lang, whisper_model, num_speakers):
    """
    Speech Recognition is based on models from OpenAI Whisper https://github.com/openai/whisper
    Speaker diarization model and pipeline from https://github.com/pyannote/pyannote-audio
    """
    time_start = time.time()
    segments = transcribe_audio(audio_file_path, selected_source_lang, whisper_model)
    embeddings = create_embeddings(audio_file_path, segments)
    segments = assign_speaker_labels(embeddings, segments, num_speakers=2)

    save_segments(segments)
    objects = create_output(segments)

    #Some descriptive system usage statistics
    time_end = time.time()
    time_diff = time_end - time_start
    memory = psutil.virtual_memory()
    gpu_utilization, gpu_memory = GPUInfo.gpu_usage()
    gpu_utilization = gpu_utilization[0] if len(gpu_utilization) > 0 else 0
    gpu_memory = gpu_memory[0] if len(gpu_memory) > 0 else 0
    system_info = f"""
    *Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB.* 
    *Processing time: {time_diff:.5} seconds.*
    *GPU Utilization: {gpu_utilization}%, GPU Memory: {gpu_memory}MiB.*
    """

    # Possibility to save the output as a .csv file
    # And return the output as a pandas dataframe (which is used in the gradio interface)
    save_path = "output/transcript_result.csv"
    json_path = "output/segments.csv"
    df_results = pd.DataFrame(objects)
    df_results.to_csv(save_path)
    return df_results, system_info, save_path, json_path

    #
    #    raise RuntimeError("Error Running inference with local model", e)

def read_and_create(file_path):
    """
    # The goal is to read the segments from the CSV file, and create a final output in second Gradio tab
    """
    print(type(file_path))
    print(file_path)
    segments = read_segments(file_path.name)
    objects = create_output(segments)
    df_results = pd.DataFrame(objects)
    save_path = "output/transcript_result.csv"
    df_results.to_csv(save_path)

    return df_results, save_path

# ---- Gradio Layout -----
# Inspiration from https://huggingface.co/spaces/RASMUS/Whisper-youtube-crosslingual-subtitles
audio_in = gr.Video(label="Audio file", mirror_webcam=False)
youtube_url_in = gr.Textbox(label="Select file", lines=1, interactive=True)
df_init = pd.DataFrame(columns=['Start', 'End', 'Speaker', 'Text'])
memory = psutil.virtual_memory()
selected_source_lang = gr.Dropdown(choices=source_language_list, type="value", value="en", label="Spoken language in audio", interactive=True)
selected_whisper_model = gr.Dropdown(choices=whisper_models, type="value", value="base", label="Selected Whisper model", interactive=True)
number_speakers = gr.Number(precision=0, value=0, label="Input number of speakers for better results. If value=0, model will automatic find the best number of speakers", interactive=True)
system_info = gr.Markdown(f"*Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB*")
download_transcript = gr.File(label="Download transcript")
download_processed_transcript = gr.File(label="Download processed transcript")
download_json = gr.File(label="Download json")
transcription_df = gr.DataFrame(value=df_init,label="Transcription dataframe", row_count=(0, "dynamic"), max_rows = 10, wrap=True, overflow_row_behaviour='paginate')
trancription_df_processed = gr.DataFrame(value=df_init,label="Processed transcription dataframe", row_count=(0, "dynamic"), max_rows = 10, wrap=True, overflow_row_behaviour='paginate')
title = "Whisper speaker diarization"
demo = gr.Blocks(title=title)
demo.encrypt = False


with demo:
    with gr.Tab("Whisper speaker diarization"):
        gr.Markdown('''
            <div>
            <h1 style='text-align: center'>Whisper speaker diarization</h1>
            This space uses Whisper models from <a href='https://github.com/openai/whisper' target='_blank'><b>OpenAI</b></a> with <a href='https://github.com/guillaumekln/faster-whisper' target='_blank'><b>CTranslate2</b></a> which is a fast inference engine for Transformer models to recognize the speech (4 times faster than original openai model with same accuracy)
            and ECAPA-TDNN model from <a href='https://github.com/speechbrain/speechbrain' target='_blank'><b>SpeechBrain</b></a> to encode and clasify speakers
            </div>
        ''')

        with gr.Row():
            gr.Markdown('''
            ''')
            
        with gr.Row():         
            gr.Markdown('''
                ### Select wav file to be transcribed:
                ###
                ''')
            wav_files = get_wav_files()
            examples = gr.Radio(examples=wav_files, label="Select audio file", inputs=[audio_in])

        with gr.Row():
            with gr.Column():
                audio_in.render()
                with gr.Column():
                    gr.Markdown('''
                    ##### Here you can start the transcription process.
                    ##### Please select the source language for transcription.
                    ##### You can select a range of assumed numbers of speakers.
                    ''')
                selected_source_lang.render()
                selected_whisper_model.render()
                number_speakers.render()
                transcribe_btn = gr.Button("Transcribe audio and diarization")
                transcribe_btn.click(speech_to_text, 
                                     [audio_in, selected_source_lang, selected_whisper_model, number_speakers],
                                     [transcription_df, system_info, download_transcript, download_json]
                                    )
                
        with gr.Row():
            gr.Markdown('''
            ##### Here you will get transcription  output
            ##### ''')
            

        with gr.Row():
            with gr.Column():
                download_transcript.render()
                download_json.render()
                transcription_df.render()
                system_info.render()
                gr.Markdown('''<center><a href="https://opensource.org/licenses/Apache-2.0"><img src='https://img.shields.io/badge/License-Apache_2.0-blue.svg' alt='License: Apache 2.0'></center>''')


    with gr.Tab("Read CSV"):
        gr.Markdown('''<h1 style='text-align: center'>Upload the CSV and create Output</h1>''')
        with gr.Column():
            #file_upload = gr.File()
            #upload_button = gr.UploadButton("Click to Upload a File", file_types=["text"], file_count="single")
            #upload_button.upload( upload_button, file_output)

            csv_upload = gr.inputs.File(label="Upload file", type="file", optional=True)
            #csv_upload.render()

            read_btn = gr.Button("Read CSV and create output")
            read_btn.click(read_and_create, [csv_upload], [trancription_df_processed, download_processed_transcript])
        with gr.Column():
            download_processed_transcript.render()
            trancription_df_processed.render()


demo.launch(debug=True)