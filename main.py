import asyncio
from vad_handler import VADHandler
from llm_handler import OpenAIHandler
import tts_handler
from stt_handler import InferenceHandler
import soundfile as sf
import queue
import threading
from queue import Queue 
import pyttsx3
from threading import Event


output_queue = queue.Queue()
openai_handler = OpenAIHandler()
listen_control = Event()
generating = Event()


async def process_audio(audio_data):
    inference_handler = InferenceHandler("http://127.0.0.1:8080/inference")
    sf.write("output.wav", audio_data, 16000)
    user_text = inference_handler.send_inference_request("output.wav")
    # Simulate sending inference request and getting user text
    print(user_text)
    if user_text != "":
        print("AI:")
        generating.set()
        openai_handler.get_openai_response(user_text, tts_handler.sentence_queue)
        generating.clear()
        print("")

        # await text_to_speech(ai_response)
        # pass
    else:
        print("Error Empyt text.")
        listen_control.set()
        print("listening...")
    
    print("")
    print("USER:")



async def main():
    vad_handler = VADHandler(audio_enhancement=True)

    listen_control.set()

    print("USER:")
    await vad_handler.record_and_process(listen_control, process_audio)



if __name__ == "__main__":

    threading.Thread(target=tts_handler.generate_audio).start()
    threading.Thread(target=tts_handler.play_audio, args=(listen_control,generating,)).start()
    # threading.Thread(target=tts_handler_2.generate_local_audio, args=(listen_control,generating,)).start()

    asyncio.run(main())
