import whisper
import ollama
import os
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import time
import subprocess
import re

FS = 44100
WHISPER_MODEL = "small"
AUDIO_FILE = "input.wav"
VOICE_NAME = "Majed"
VOICE_RATE = "235"
MAX_HISTORY = 10

chat_history = []

def fix_print(text, color="white"):
    reshaped = get_display(reshape(text))
    colors = {"green": "\033[92m", "blue": "\033[94m", "yellow": "\033[93m", "white": "\033[0m"}
    print(f"{colors.get(color, '')}{reshaped}\033[0m")

def speak(text):
    if not text.strip(): return
    clean_text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text).strip()
    subprocess.run(['say', '-v', VOICE_NAME, '-r', VOICE_RATE, clean_text])

def record_audio(duration=4):
    fix_print("جاري التسجيل...", "yellow")
    recording = sd.rec(int(duration * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    write(AUDIO_FILE, FS, (recording * 32767).astype(np.int16))

def main():
    fix_print("جاري تحميل النظام...", "white")
    stt_model = whisper.load_model(WHISPER_MODEL)
    
    system_prompt = {
        "role": "system", 
        "content": (
            "IDENTITY: You are a highly intelligent, friendly AI Voice Assistant for a graduation project. "
            "STRICT LANGUAGE RULE: Respond ONLY in Arabic (Jordanian/Levantine dialect is preferred) or English. "
            "ABSOLUTELY NO CHINESE CHARACTERS. If you use any Chinese, the system will fail. "
            "STYLE: Be concise, helpful, and natural. Keep answers short (1-3 sentences) because they will be spoken out loud."
        )
    }
    chat_history.append(system_prompt)

    fix_print("النظام جاهز للعمل", "green")

    while True:
        user_choice = input("\nاضغط Enter للتحدث أو p للخروج: ").strip().lower()

        if user_choice == "p":
            fix_print("تم إنهاء البرنامج بالتوفيق", "yellow")
            break

        record_audio()

        try:
            result = stt_model.transcribe(AUDIO_FILE, fp16=False, language='ar')
            user_text = result["text"].strip()
            if not user_text: continue
            
            fix_print(f"أنت: {user_text}", "green")
            chat_history.append({"role": "user", "content": user_text})
        except Exception as e:
            continue

        if len(chat_history) > MAX_HISTORY:
            chat_history[:] = [chat_history[0]] + chat_history[-(MAX_HISTORY-1):]

        try:
            fix_print("الرد: ", "blue")
            full_response = ""
            sentence_buffer = ""
            
            response_stream = ollama.chat(
                model="qwen2.5:14b", 
                messages=chat_history,
                stream=True 
            )

            for chunk in response_stream:
                content = chunk['message']['content']
                if any('\u4e00' <= char <= '\u9fff' for char in content):
                    continue 
                
                full_response += content
                sentence_buffer += content
                
                if any(p in content for p in ['.', '؟', '!', '\n']):
                    print(get_display(reshape(sentence_buffer)), end='', flush=True)
                    speak(sentence_buffer)
                    sentence_buffer = ""

            if sentence_buffer:
                print(get_display(reshape(sentence_buffer)), flush=True)
                speak(sentence_buffer)

            chat_history.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            fix_print("خطأ في معالجة الرد", "red")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fix_print("\nتم الإيقاف", "yellow")