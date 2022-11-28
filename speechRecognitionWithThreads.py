import scipy.io.wavfile
import soundfile
import speech_recognition as sr
from scipy.io.wavfile import write
import numpy as np
import scipy as sc
from threading import Thread
from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO
import numpy as np
import plotly.graph_objs as go
import plotly
import IPython
import subprocess

try:
    from queue import Queue  # Python 3 import
except ImportError:
    from Queue import Queue  # Python 2 import

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
            with open("outputexample.wav", "wb") as f:
                f.write(audio.get_wav_data())
            sound = np.frombuffer(audio.frame_data, dtype=np.int16)
            # this segment will be taking the audio data and process through diarization and noise reduction
            # --------------------------------------------------------------------------------------
            # read audio data from file
            # (returns sampling freq and signal as a numpy array)
            fs, s = aIO.read_audio_file("outputexample.wav")
            print(s)
            print(fs)
            # play the initial and the generated files in notebook:
            IPython.display.display(IPython.display.Audio("outputexample.wav"))

            # print duration in seconds:
            duration = len(s) / float(fs)
            print(f'duration = {duration} seconds')

            # extract short-term features using a 50msec non-overlapping windows
            win, step = 0.050, 0.050
            [f, fn] = aF.feature_extraction(s, fs, int(fs * win),
                                            int(fs * step))
            print(f'{f.shape[1]} frames, {f.shape[0]} short-term features')
            print('Feature names:')
            for i, nam in enumerate(fn):
                print(f'{i}:{nam}')
            # plot short-term energy
            # create time axis in seconds
            time = np.arange(0, duration - step, win)
            # get the feature whose name is 'energy'
            energy = f[fn.index('energy'), :]
            mylayout = go.Layout(yaxis=dict(title="frame energy value"),
                                 xaxis=dict(title="time (sec)"))
            plotly.offline.iplot(go.Figure(data=[go.Scatter(x=time,
                                                            y=energy)],
                                           layout=mylayout))

            # --------------------------------------------------------------------------------------
            audio.frame_rate, audio.sample_data = scipy.io.wavfile.read("outputexample.wav")
            audio_queue.put(r.listen(source, timeout=5, phrase_time_limit=5))
    except KeyboardInterrupt:  # allow Ctrl + C to shut down the program
        pass

audio_queue.join()  # block until all current audio processing jobs are done
audio_queue.put(None)  # tell the recognize_thread to stop
recognize_thread.join()  # wait for the recognize_thread to actually stop
