import csv
import warnings
from threading import Thread

import malaya_speech
import soundfile as sf
import speech_recognition as sr
from malaya_speech import Pipeline
import malaya_speech.diarization as diarization
from sklearn.model_selection import GridSearchCV
import CustomTransformer
import CustomClassifier

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')
from scipy.io import wavfile

try:
    from queue import Queue  # Python 3 import
except ImportError:
    from Queue import Queue  # Python 2 import
# creating the parameters grid

param_grid = {
    'scaler__appendEnergy': [True, False],
    'scaler__winlen': [0.020, 0.025, 0.015, 0.01],
    'scaler__preemph': [0.95, 0.97, 1, 0.90, 0.5, 0.1],
    'scaler__numcep': [20, 13, 16],
    'scaler__nfft': [1024, 1200, 512],
    'scaler__ceplifter': [15, 22, 0],
    'scaler__highfreq': [6000],
    'scaler__nfilt':[55, 0, 22],
    'svc__n_components': [2 * i for i in range(0, 12, 1)],
    'svc__max_iter': list(range(50, 400, 50)),
    'svc__covariance_type': ['full', 'tied', 'diag', 'spherical'],
    'svc__n_init': list(range(1, 4, 1)),
    'svc__init_params': ['kmeans', 'random']
}



with open('SignList_ClassId_TR_EN.csv') as f:
    turkish = [row[1] for row in csv.reader(f)]

with open('SignList_ClassId_TR_EN.csv') as f:
    english = [row[2] for row in csv.reader(f)]

print(turkish)
print(english)

model = malaya_speech.speaker_vector.deep_model('speakernet')

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

            print(
                "Google Speech Recognition thinks you said " + record.recognize_google(audiosentence, language="tr-tr"))
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
    def load_wav(file):
        return malaya_speech.load(file)[0]

    # load model
    def deep_model(model: str = 'speakernet', quantized: bool = False, **kwargs):
        """
        Load Speaker2Vec model.

        Parameters
        ----------
        model : str, optional (default='speakernet')
            Model architecture supported. Allowed values:

            * ``'vggvox-v1'`` - VGGVox V1, embedding size 1024
            * ``'vggvox-v2'`` - VGGVox V2, embedding size 512
            * ``'deep-speaker'`` - Deep Speaker, embedding size 512
            * ``'speakernet'`` - SpeakerNet, embedding size 7205

        quantized : bool, optional (default=False)
            if True, will load 8-bit quantized model.
            The quantized model isn’t necessarily faster, it totally depends on the machine.

        Returns
        -------

        result : malaya_speech.supervised.classification.load function
        """


    try:
        while True:  # repeatedly listen for phrases and put the resulting audio on the audio processing job queue
            audio = record.listen(source, timeout=5, phrase_time_limit=10)  # It takes the microphones audio data
            with open("output-example1.flac", "wb") as f:
                f.write(audio.get_flac_data())
            # this segment will be taking the audio data and process through diarization and noise reduction
            # --------------------------------------------------------------------------------------
            # read audio data from file
            data, sample_rate = sf.read("output-example1.flac")
            # reduce_noise = nr.reduce_noise(y=data, sr=sample_rate)  # apply the noise reduction
            sf.write("output-example4.wav", data, samplerate=sample_rate)
            # sf.write("output-example1.flac", reduce_noise, samplerate=sample_rate)

            sample_rate, data = wavfile.read('output-example4.wav')

            # --------------------------------------------------------------------------------------

            with sr.AudioFile("output-example4.wav") as source_from_file:
                audio = record.record(source_from_file)

            # try:
            #    output = pipeline("output-example4.wav", min_speakers=1, max_speakers=2)
            # except ValueError:
            #    pass
            speakers = ['Ege.wav', 'output-example4.wav']

            p = Pipeline()
            frame = p.foreach_map(load_wav).map(model)

            print("**************************************************")
            r = p.emit(speakers)

            print("**************************************************")
            print(diarization.speaker_similarity(data, r['speaker-vector']))
            # calculate similarity
            from scipy.spatial.distance import cdist

            1 - cdist(r['speaker-vector'], r['speaker-vector'], metric='cosine')
            print(1 - cdist(r['speaker-vector'], r['speaker-vector'], metric='cosine'))
            print(data)
            #x, y = CustomClassifier.load_data('output-example4.wav')

            ## creating pipeline of transformer and classifier
            #pipe = Pipeline([('scaler', CustomTransformer()), ('svc', CustomClassifier())])

            #search = GridSearchCV(pipe, param_grid, n_jobs=-1)
            ## searching for appropriate parameters
            #search.fit(x, y)

            combine = 0
            # for turn, _, speaker in output.itertracks(yield_label=True):
            #    # times between which to extract the wave from
            #    start = turn.start  # seconds
            #    end = turn.end  # seconds
            #
            #    if speaker[9] == "0":
            #        # file to extract the snippet from
            #        try:
            #            with wave.open('output-example4.wav', "rb") as infile:
            #                # get file data
            #                nchannels = infile.getnchannels()
            #                sampwidth = infile.getsampwidth()
            #                framerate = infile.getframerate()
            #                # set position in wave to start of segment
            #                infile.setpos(int(start * framerate))
            #                # extract data
            #                data = infile.readframes(int((end - start) * framerate))
            #        except wave.Error:
            #            pass
            #
            #        try:
            #            # write the extracted data to a new file
            #            with wave.open('outputfile.wav', 'w') as outfile:
            #                outfile.setnchannels(nchannels)
            #                outfile.setsampwidth(sampwidth)
            #                outfile.setframerate(framerate)
            #                outfile.setnframes(int(len(data) / sampwidth))
            #                outfile.writeframes(data)
            #        except wave.Error:
            #            pass
            #        infile.close()
            #        outfile.close()
            #
            #       the_result_audio_file = AudioSegment.from_wav("outputfile.wav")
            #
            #        combine = combine + the_result_audio_file
            #        combine.export("C:/Users/serha/PycharmProjects/pythonProject/combined.wav", format='wav')
            #    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            with sr.AudioFile("output-example4.wav") as the_combined_data:
                audio = record.record(the_combined_data)
            audio_queue.put(audio)
    except KeyboardInterrupt:  # allow Ctrl + C to shut down the program
        pass

audio_queue.join()  # block until all current audio processing jobs are done
audio_queue.put(None)  # tell the recognize_thread to stop
recognize_thread.join()  # wait for the recognize_thread to actually stop
