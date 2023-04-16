import logging

import urllib.request
import wave
from pathlib import Path
import os
import urllib.request
from huggingface_hub import HfApi, Repository
import speech_recognition as sr
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pydub.utils import make_chunks

logger = logging.getLogger(__name__)

#Globel Variables
pyannote_url = "https://huggingface.co/pyannote/"
speechbrain_url = "https://huggingface.co/speechbrain/"
pyannote_dir = "pyannote_models"
speechbrain_dir = "speechbrain_models"


class SpeechDiarization:
    def __init__(self):
        self.pipeline = None
        self.verification = None

    def diarization(self):
        self._load_model("pyannote/speaker-diarization@2.1", pyannote_dir)

    def load_verify_model(self):
        self._load_model("speechbrain/spkrec-ecapa-voxceleb", speechbrain_dir)

    def _load_model(self, model_name, cache_dir):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        model_path = os.path.join(cache_dir, model_name.split("/")[-1])
        if not os.path.exists(model_path):
            try:
                urllib.request.urlopen(f"{pyannote_url}/{model_name}")
            except urllib.error.URLError:
                print(
                    f"Error: Unable to connect to the {model_name.split('/')[0]} model server. Please check your internet connection and try again.")
                exit()
            print(model_name)
            self.pipeline = Pipeline.from_pretrained(model_name,
                                                     use_auth_token="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",
                                                     cache_dir=cache_dir)
        else:
            self.pipeline = Pipeline.from_pretrained(model_path)

class SpeechRecognizer:
    def __init__(self, speech_diarization):
        self.profileSpeech = Path("Ege.wav")
        self.r = sr.Recognizer()
        self.pipeline = speech_diarization.pipeline
        self.verification = speech_diarization.verification

    def recognize(self, filename):
        input_file = Path(filename)
        output_file = Path("outputfile.wav")
        combined_file = Path("combined.wav")
        with wave.open(str(input_file), "rb") as infile:
            nchannels = infile.getnchannels()
            sampwidth = infile.getsampwidth()
            framerate = infile.getframerate()

        try:
            combine = AudioSegment.silent(duration=0)
            # Recognize speech
            output = self.pipeline(input_file, min_speakers=1, max_speakers=2)
            for turn, _, speaker in output.itertracks(yield_label=True):
                print("annen")
                start = turn.start
                end = turn.end
                infile.setpos(int(start * framerate))
                data = infile.readframes(int((end - start) * framerate))
                with wave.open(str(output_file), "wb") as outfile:
                    outfile.setnchannels(nchannels)
                    outfile.setsampwidth(sampwidth)
                    outfile.setframerate(framerate)
                    outfile.setnframes(int(len(data) / sampwidth))
                    outfile.writeframes(data)
                #verify the temperorary file
                try:
                    score, prediction = self.verification.verify_files(output_file,self.profileSpeech)
                    print(prediction)
                except RuntimeError as k:
                    continue
                if prediction is True:
                    the_result_audio_file = AudioSegment.from_wav(str(output_file))
                    combine += the_result_audio_file
                else:
                    continue
                print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            combine.export(str(combined_file), format="wav")
            chunks = make_chunks(combine, 1000)
            combined_audio = AudioSegment.empty()
            for chunk in chunks:
                combined_audio += chunk
            combined_audio.export('C:/Users/serha/PycharmProjects/pythonProject/combined.wav', format='wav')
            with sr.AudioFile("output-example4.wav") as the_combined_data:
                audio = self.r.record(the_combined_data)
            text = self.r.recognize_google(audio, language="tr-tr")
            logger.info(f"Recognized: {text}")
            return text
        except sr.UnknownValueError as e:
            return e
        except sr.RequestError as e:
            return e
        except ValueError as e:
            return e

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # create an instance of SpeechDiarization
    speech_diarization = SpeechDiarization()
    speech_diarization.diarization()
    speech_diarization.load_verify_model()

    recognizer = SpeechRecognizer(speech_diarization)
    filename = "Recording.wav"

    text = recognizer.recognize(filename)
    print(f"Recognized: {text}")
