from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import base64
import os
import subprocess
import json

from deepface import DeepFace

from .whisper_service import transcribe_audio_file
from .llm_service import get_emora_response, generate_response, generate_english_response
from .tts_service import generate_telugu_audio


def home(request):
    return render(request, "index.html")


@csrf_exempt
def voice_input(request):
    """Telugu voice recording → Whisper transcribe → LLM → TTS audio"""
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"})

    input_file = "input.webm"
    wav_file = "input.wav"

    try:
        audio_file = request.FILES.get("audio")
        if not audio_file:
            return JsonResponse({"error": "No audio received"})

        with open(input_file, "wb+") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_file, "-ar", "16000", "-ac", "1", wav_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            return JsonResponse({"error": "Audio conversion failed"})

        text = transcribe_audio_file(wav_file)
        if not text:
            return JsonResponse({
                "text": "",
                "response": "నేను వినలేకపోయాను. మళ్ళీ చెప్పండి.",
                "audio": ""
            })

        response_text = generate_response(text)
        audio_path = generate_telugu_audio(response_text)

        return JsonResponse({"text": text, "response": response_text, "audio": audio_path})

    except Exception as e:
        return JsonResponse({"error": str(e)})
    finally:
        for f in [input_file, wav_file]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass


@csrf_exempt
def english_voice_input(request):
    """English text from browser STT → LLM → text back for browser TTS"""
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"})
    try:
        body = json.loads(request.body)
        text = body.get("text", "").strip()
        if not text:
            return JsonResponse({"error": "No text received"})

        response_text = generate_english_response(text)
        return JsonResponse({"text": text, "response": response_text})

    except Exception as e:
        return JsonResponse({"error": str(e)})


@csrf_exempt
def receive_image(request):
    """Webcam frame → DeepFace emotion → LLM comment → audio (Telugu) or text (English)"""
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"})

    image_file = None
    try:
        image_data = request.POST.get("image")
        lang = request.POST.get("lang", "telugu")

        if not image_data:
            return JsonResponse({"error": "No image received"})
        if "base64," not in image_data:
            return JsonResponse({"error": "Invalid image format"})

        header, imgstr = image_data.split("base64,")
        ext = header.split("/")[-1].replace(";", "").replace("data:", "")
        ext = ext if ext in ["jpg", "jpeg", "png"] else "png"
        image_file = f"captured_image.{ext}"

        with open(image_file, "wb") as f:
            f.write(base64.b64decode(imgstr))

        result = DeepFace.analyze(
            img_path=image_file,
            actions=["emotion"],
            enforce_detection=False
        )
        emotion = result[0]["dominant_emotion"]
        print(f"Detected Emotion: {emotion}")

        if lang == "english":
            ai_response = generate_english_response(f"emotion:{emotion}")
            print(f"AI English Emotion Response: {ai_response}")
            return JsonResponse({
                "emotion": emotion,
                "response": ai_response,
                "audio": ""
            })
        else:
            ai_response = get_emora_response(emotion)
            print(f"AI Telugu Emotion Response: {ai_response}")
            audio_path = generate_telugu_audio(ai_response)
            return JsonResponse({
                "emotion": emotion,
                "response": ai_response,
                "audio": audio_path
            })

    except Exception as e:
        print("Emotion Detection Error:", e)
        return JsonResponse({"error": str(e)})
    finally:
        if image_file:
            try:
                if os.path.exists(image_file):
                    os.remove(image_file)
            except:
                pass