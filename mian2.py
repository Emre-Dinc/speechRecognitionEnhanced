import speech_recognition as sr
import sounddevice as sd
import numpy as np
import os
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 15  # Duration of recording
sd.default.dtype = 'int32', 'int32'
print("Start recording the answer.....")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording.astype(np.int16))  # Save as WAV file in 16-bit format
recognizer = sr.Recognizer()
sound = "output.wav"
while True:
    with sr.AudioFile(sound) as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Converting the answer to text...")
        audio = recognizer.listen(source)
        print(audio)

        try:
            text = recognizer.recognize_google(audio, language="tr-tr")
            print("The converted text:" + text)


        except sr.UnknownValueError:

            print("Söylediğinizi algılayamadım")

        except sr.RequestError as e:

            print("Google API'a ulaşılamadı dolayısıyla program kapatılıyor {0}".format(e))
