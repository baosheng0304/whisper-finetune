import argparse
import os
from pathlib import Path
from transformers import pipeline

parser = argparse.ArgumentParser(description='Script to transcribe a custom audio file of any length using Whisper Models of various sizes.')
parser.add_argument(
    "--hf_model",
    type=str,
    required=False,
    # default="openai/whisper-small",
    default="./models_small",
    help="Huggingface model name. Example: openai/whisper-tiny",
)
parser.add_argument(
    "--ckpt_dir",
    type=str,
    required=False,
    default="./models",
    help="Folder with the pytorch_model.bin file",
)
parser.add_argument(
    "--temp_ckpt_folder",
    type=str,
    required=False,
    default="temp_dir",
    help="Path to create a temporary folder containing the model and related files needed for inference",
)
parser.add_argument(
    "--path_to_audio",
    type=str,
    required=False,
    default="C:/zRHM/dataset/audio/001286_4.wav",
    help="Path to the audio file to be transcribed.",
)
parser.add_argument(
    "--language",
    type=str,
    required=False,
    default="ko",
    help="Two letter language code for the transcription language, e.g. use 'hi' for Hindi. This helps initialize the tokenizer.",
)
parser.add_argument(
    "--device",
    type=int,
    required=False,
    default=0,
    help="The device to run the pipeline on. -1 for CPU, 0 for the first GPU (default) and so on.",
)

args = parser.parse_args()

model_id = args.hf_model

transcribe = pipeline(
    task="automatic-speech-recognition",
    model=model_id,
    chunk_length_s=30,
    device=args.device,
)

transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language=args.language, task="transcribe")

print('Transcription: ')

ret = transcribe(args.path_to_audio)
print(ret["text"])
