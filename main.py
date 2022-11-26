import speech_recognition as sr
import scipy as sc


r = sr.Recognizer()
m = sr.Microphone()

try:
    print("Lütfen sessizliği sağlayınız:")
    with m as source:
        r.adjust_for_ambient_noise(source, duration=0)
    print("minimum enerji eşiği belirleniyor {}".format(r.energy_threshold))
    while True:
        print("Bir şey söyleyiniz!")
        with m as source:
            audio = r.listen(source, timeout=5, phrase_time_limit= 5)
            print(audio)
        print("Tamamdır! Şimdi söylediklerinizi algılıyorum...")
        try:
            value = r.recognize_google(audio, language="tr-tr")

            if str is bytes:
                print(u"Söylediğiniz cümle: {}".format(value).encode("utf-8"))
            else:
                print("Söylediğiniz cümle {}".format(value))
        except sr.UnknownValueError:
            print("Söylediğinizi algılayamadım")
        except sr.RequestError as e:
            print("Google API'a ulaşılamadı dolayısıyla program kapatılıyor {0}".format(e))
except KeyboardInterrupt:
    pass
