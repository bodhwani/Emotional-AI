import speech_recognition as sr
r = sr.Recognizer()
harvard = sr.AudioFile('output.wav')
with harvard as source:
    audio = r.record(source)
print("text is ",r.recognize_google(audio))