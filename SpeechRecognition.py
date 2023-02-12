import csv
import warnings
import wave
from threading import Thread
import soundfile as sf
import speech_recognition as sr

import torchaudio
import noisereduce as nr
from pyannote.audio import Pipeline
from pydub import AudioSegment
from speechbrain.pretrained import EncoderDecoderASR, SpeakerRecognition

"""warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')"""
from scipy.io import wavfile

from queue import Queue  # Python 3 import

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="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")

verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                               savedir="pretrained_models/spkrec-ecapa-voxceleb")

record = sr.Recognizer()
record_for_another_recognizer = sr.Recognizer()
audio_queue = Queue()
sampleRate = 32000
accountSpeakerFile = "C:/Users/serha/PycharmProjects/pythonProject/Ege.wav"

print("minimum enerji eşiği belirleniyor {}".format(record.energy_threshold))


def recognize_worker():
    # this runs in a background thread
    while True:
        audiosentence = audio_queue.get()  # retrieve the next audio processing job from the main thread
        if audio is None: break  # stop processing if the main thread is done

        # received audio data, now we'll recognize it using Google Speech Recognition
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            sentence = record.recognize_google(audiosentence, language="tr-tr")
            print(
                "Google Speech Recognition şunu söylediğini düşünüyor: " + sentence)
        except sr.UnknownValueError:
            print("Google Speech Recognition söylediğini anlayamadı")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        except AssertionError as e:
            print(format(e))

        audio_queue.task_done()  # mark the audio processing job as completed in the queue


# start a new thread to recognize audio, while this thread focuses on listening
recognize_thread = Thread(target=recognize_worker)
recognize_thread.daemon = True
recognize_thread.start()
with sr.Microphone() as source:
    try:
        while True:  # repeatedly listen for phrases and put the resulting audio on the audio processing job queue
            audio = record.listen(source, phrase_time_limit=10)  # It takes the microphones audio data
            with open("output-example1.flac", "wb") as f:
                f.write(audio.get_flac_data())
            # this segment will be taking the audio data and process through diarization and noise reduction
            # --------------------------------------------------------------------------------------
            # read audio data from file
            data, sample_rate = sf.read("output-example1.flac")
            reduce_noise = nr.reduce_noise(y=data, sr=sample_rate)  # apply the noise reduction
            sf.write("output-example4.wav", reduce_noise, samplerate=sample_rate)
            # sf.write("output-example1.flac", reduce_noise, samplerate=sample_rate)

            # --------------------------------------------------------------------------------------

            try:
                output = pipeline("output-example4.wav", min_speakers=1, max_speakers=2)
            except ValueError :
                break
            except NameError:
                break

            combine = 0
            for turn, _, speaker in output.itertracks(yield_label=True):

                # times between which to extract the wave from
                start = turn.start  # seconds
                end = turn.end  # seconds
                # file to extract the snippet from
                if accountSpeakerFile is None:
                    if speaker[9] == "0":
                        # file to extract the snippet from
                        try:
                            with wave.open('output-example4.wav', "rb") as infile:
                                # get file data
                                nchannels = infile.getnchannels()
                                sampwidth = infile.getsampwidth()
                                framerate = infile.getframerate()
                                # set position in wave to start of segment
                                infile.setpos(int(start * framerate))
                                # extract data
                                data = infile.readframes(int((end - start) * framerate))
                        except wave.Error:
                            break

                        try:
                            # write the extracted data to a new file
                            with wave.open('outputfile.wav', 'wb') as outfile:
                                outfile.setnchannels(nchannels)
                                outfile.setsampwidth(sampwidth)
                                outfile.setframerate(framerate)
                                outfile.setnframes(int(len(data) / sampwidth))
                                outfile.writeframes(data)
                        except wave.Error:
                            break
                        infile.close()
                        outfile.close()

                        the_result_audio_file = AudioSegment.from_wav("outputfile.wav")

                        combine = combine + the_result_audio_file
                        combine.export("C:/Users/serha/PycharmProjects/pythonProject/combined.wav", format='wav')
                else:
                    try:
                        with wave.open('output-example4.wav', "rb") as infile:
                            # get file data
                            nchannels = infile.getnchannels()
                            sampwidth = infile.getsampwidth()
                            framerate = infile.getframerate()
                            # set position in wave to start of segment
                            infile.setpos(int(start * framerate))
                            # extract data
                            data = infile.readframes(int((end - start) * framerate))
                    except wave.Error:
                        break
                    try:
                        # write the extracted data to a new file
                        with wave.open('outputfile.wav', 'wb') as outfile:
                            outfile.setnchannels(nchannels)
                            outfile.setsampwidth(sampwidth)
                            outfile.setframerate(framerate)
                            outfile.setnframes(int(len(data) / sampwidth))
                            outfile.writeframes(data)
                    except wave.Error:
                        break
                    try:
                        score, prediction = verification.verify_files("outputfile.wav", accountSpeakerFile)
                    except RuntimeError as k:
                        break
                    if prediction is True:
                        the_result_audio_file = AudioSegment.from_wav("outputfile.wav")
                        combine = combine + the_result_audio_file
                        combine.export("C:/Users/serha/PycharmProjects/pythonProject/combined.wav", format='wav')
                    else:
                        break

                    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            with sr.AudioFile("output-example4.wav") as the_combined_data:
                audio = record.record(the_combined_data)
            audio_queue.put(audio)
    except KeyboardInterrupt:  # allow Ctrl + C to shut down the program
        pass

audio_queue.join()  # block until all current audio processing jobs are done
audio_queue.put(None)  # tell the recognize_thread to stop
recognize_thread.join()  # wait for the recognize_thread to actually stop
