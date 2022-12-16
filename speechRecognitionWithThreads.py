import csv
import os
import wave
from threading import Thread

import noisereduce as nr
import soundfile as sf
import speech_recognition as sr
from pyannote.audio import Pipeline
from pydub import AudioSegment

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

record = sr.Recognizer()
record_for_another_recognizer = sr.Recognizer()
audio_queue = Queue()
sampleRate = 32000

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

            print("Google Speech Recognition thinks you said " + record.recognize_google(audiosentence, language="tr-tr"))
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
            audio = record.listen(source, timeout=5, phrase_time_limit=10)  # It takes the microphones audio data
            with open("output-example1.flac", "wb") as f:
                f.write(audio.get_flac_data())
            # this segment will be taking the audio data and process through diarization and noise reduction
            # --------------------------------------------------------------------------------------
            # read audio data from file
            data, sample_rate = sf.read("output-example1.flac")
            reduce_noise = nr.reduce_noise(y=data, sr=sample_rate)  # apply the noise reduction

            # --------------------------------------------------------------------------------------

            sf.write("output-example4.wav", reduce_noise, samplerate=sample_rate)
            sf.write("output-example1.flac", reduce_noise, samplerate=sample_rate)
            # try:
            #    diarization = pipeline("output-example4.wav")
            # except ValueError:
            #    pass
            with sr.AudioFile("output-example4.wav") as source_from_file:
                audio = record.record(source_from_file)

            # for turn, _, speaker in diarization.itertracks(yield_label=True):
            #    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            try:
                output = pipeline("output-example4.wav")
            except ValueError:
                pass
            combine = 0
            for turn, _, speaker in output.itertracks(yield_label=True):
                # times between which to extract the wave from
                start = turn.start  # seconds
                end = turn.end  # seconds

                if speaker[9] == "0":
                    # file to extract the snippet from
                    with wave.open('output-example4.wav', "rb") as infile:
                        # get file data
                        nchannels = infile.getnchannels()
                        sampwidth = infile.getsampwidth()
                        framerate = infile.getframerate()
                        # set position in wave to start of segment
                        infile.setpos(int(start * framerate))
                        # extract data
                        data = infile.readframes(int((end - start) * framerate))

                    # write the extracted data to a new file
                    with wave.open('outputfile.wav', 'w') as outfile:
                        outfile.setnchannels(nchannels)
                        outfile.setsampwidth(sampwidth)
                        outfile.setframerate(framerate)
                        outfile.setnframes(int(len(data) / sampwidth))
                        outfile.writeframes(data)
                    infile.close()
                    outfile.close()

                    the_result_audio_file = AudioSegment.from_wav("outputfile.wav")

                    combine = combine + the_result_audio_file
                    combine.export("C:/Users/serha/PycharmProjects/pythonProject/combined.wav", format='wav')
                print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            with sr.AudioFile("combined.wav") as the_combined_data:
                audio = record.record(the_combined_data)
            audio_queue.put(audio)
    except KeyboardInterrupt:  # allow Ctrl + C to shut down the program
        pass

audio_queue.join()  # block until all current audio processing jobs are done
audio_queue.put(None)  # tell the recognize_thread to stop
recognize_thread.join()  # wait for the recognize_thread to actually stop


# a function that splits the audio file into chunks
# and applies speech recognition
def get_large_audio_transcription(path):
    whole_text = ''
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)
    # recognize the chunk
    with sr.AudioFile(path) as source:
        audio_listened = record.record(source)
        # try converting it to text
        try:
            text = record.recognize_google(audio_listened)
        except sr.UnknownValueError as e:
            print("Error:", str(e))
        else:
            text = f"{text.capitalize()}. "
            whole_text += text
    # return the text for all chunks detected
    return whole_text
