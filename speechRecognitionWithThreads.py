from threading import Thread

import noisereduce as nr
import soundfile as sf
import speech_recognition as sr
from pyannote.audio import Pipeline
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pyannote.database import get_protocol, FileFinder
from pyannote.audio.pipelines import VoiceActivityDetection
import matplotlib.pyplot as plt
import numpy as np
import csv

try:
    from queue import Queue  # Python 3 import
except ImportError:
    from Queue import Queue  # Python 2 import


with open('SignList_ClassId_TR_EN.csv') as f:
    turkish = [row[1] for row in csv.reader(f)]

with open('SignList_ClassId_TR_EN.csv') as f:
    english = [row[2] for row in csv.reader(f)]

print(turkish)
print(english)

pipeline = Pipeline.from_pretrained("pyannote/speaker-segmentation",
                                    use_auth_token="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")

r = sr.Recognizer()
audio_queue = Queue()
sampleRate = 22500

print("minimum enerji eşiği belirleniyor {}".format(r.energy_threshold))


def recognize_worker():
    # this runs in a background thread
    while True:
        audio = audio_queue.get()  # retrieve the next audio processing job from the main thread
        if audio is None: break  # stop processing if the main thread is done

        # received audio data, now we'll recognize it using Google Speech Recognition
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            sentence = r.recognize_google(audio, language="tr-tr")

            print("Google Speech Recognition thinks you said " + r.recognize_google(audio, language="tr-tr"))
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

        audio_queue.task_done()  # mark the audio processing job as completed in the queue


# start a new thread to recognize audio, while this thread focuses on listening
recognize_thread = Thread(target=recognize_worker)
recognize_thread.daemon = True
recognize_thread.start()
with sr.Microphone() as source:
    try:
        while True:  # repeatedly listen for phrases and put the resulting audio on the audio processing job queue
            audio = r.listen(source, timeout=5, phrase_time_limit=5)  # It takes the microphones audio data
            with open("output-example1.flac", "wb") as f:
                f.write(audio.get_flac_data())
            # this segment will be taking the audio data and process through diarization and noise reduction
            # --------------------------------------------------------------------------------------
            # read audio data from file
            data, sample_rate = sf.read("output-example1.flac")
            reduce_noise = nr.reduce_noise(y=data, sr=sample_rate)

            # --------------------------------------------------------------------------------------

            sf.write("output-example4.wav", reduce_noise, samplerate=sample_rate)
            sf.write("output-example1.flac", reduce_noise, samplerate=sample_rate)
            #try:
            #    diarization = pipeline("output-example4.wav")
            #except ValueError:
            #    pass

            source.audio = audio
            #for turn, _, speaker in diarization.itertracks(yield_label=True):
            #    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            try:
                output = pipeline("output-example4.wav")
            except ValueError:
                pass

            for turn, _, speaker in output.itertracks(yield_label=True):
                print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

            print(output.itertracks(yield_label=True))

            audio_queue.put(audio)
    except KeyboardInterrupt:  # allow Ctrl + C to shut down the program
        pass

audio_queue.join()  # block until all current audio processing jobs are done
audio_queue.put(None)  # tell the recognize_thread to stop
recognize_thread.join()  # wait for the recognize_thread to actually stop
