from transformers import VitsModel, AutoTokenizer
from django.conf import settings
import torch
import scipy.io.wavfile
import numpy as np
import uuid
import os
import re

# ─────────────────────────────────────────
# Load Telugu TTS Model
# ─────────────────────────────────────────
print("Loading Telugu TTS model...")
model     = VitsModel.from_pretrained("facebook/mms-tts-tel")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tel")
model     = model.to("cpu")
torch.set_num_threads(2)
print("Telugu TTS model loaded ✓")

SAMPLE_RATE         = 16000
MAX_CHARS_PER_CHUNK = 180

# ─────────────────────────────────────────
# English → Telugu replacements
# ─────────────────────────────────────────
ENGLISH_REPLACEMENTS = {
    r'\bhi\b':         'హాయ్',
    r'\bhello\b':      'నమస్కారం',
    r'\bok\b':         'సరే',
    r'\bokay\b':       'సరే',
    r'\byes\b':        'అవును',
    r'\bno\b':         'లేదు',
    r'\bai\b':         'కృత్రిమ మేధ',
    r'\bunique\b':     'ప్రత్యేకమైన',
    r'\bfeel\b':       'అనిపించు',
    r'\bgood\b':       'మంచి',
    r'\bbad\b':        'చెడు',
    r'\bfriend\b':     'స్నేహితుడు',
    r'\blife\b':       'జీవితం',
    r'\blove\b':       'ప్రేమ',
    r'\bhappy\b':      'సంతోషం',
    r'\bsad\b':        'దుఃఖం',
    r'\btime\b':       'సమయం',
    r'\bday\b':        'రోజు',
    r'\bnight\b':      'రాత్రి',
    r'\bwork\b':       'పని',
    r'\bhome\b':       'ఇల్లు',
    r'\bfamily\b':     'కుటుంబం',
    r'\brelax\b':      'విశ్రాంతి తీసుకో',
    r'\bstress\b':     'ఒత్తిడి',
    r'\benergy\b':     'శక్తి',
    r'\bpositive\b':   'సానుకూలంగా',
    r'\bnegative\b':   'ప్రతికూలంగా',
    r'\bstrong\b':     'బలంగా',
    r'\bsmile\b':      'చిరునవ్వు',
    r'\bhope\b':       'ఆశ',
    r'\bdream\b':      'కల',
    r'\bsuccess\b':    'విజయం',
    r'\bhelp\b':       'సహాయం',
    r'\bcare\b':       'శ్రద్ధ',
    r'\btrust\b':      'నమ్మకం',
    r'\bexcited\b':    'ఉత్సాహంగా',
    r'\bworry\b':      'ఆందోళన',
    r'\bfear\b':       'భయం',
    r'\bcalm\b':       'శాంతి',
    r'\bpeace\b':      'శాంతి',
    r'\bjoy\b':        'ఆనందం',
    r'\bpain\b':       'నొప్పి',
    r'\btears\b':      'కన్నీళ్ళు',
    r'\blaugh\b':      'నవ్వు',
    r'\bcry\b':        'ఏడ్చు',
    r'\bthink\b':      'ఆలోచించు',
    r'\blearn\b':      'నేర్చుకో',
    r'\bgrow\b':       'ఎదుగు',
    r'\bchange\b':     'మార్పు',
    r'\bstart\b':      'ప్రారంభించు',
    r'\bstop\b':       'ఆపు',
    r'\btry\b':        'ప్రయత్నించు',
    r'\bwait\b':       'వేచి ఉండు',
    r'\bcome\b':       'రా',
    r'\bgo\b':         'వెళ్ళు',
    r'\bsee\b':        'చూడు',
    r'\bhear\b':       'వినండి',
    r'\bknow\b':       'తెలుసు',
    r'\bunderstand\b': 'అర్థం చేసుకో',
    r'\bremember\b':   'గుర్తుంచుకో',
    r'\bforget\b':     'మర్చిపో',
    r'\btalk\b':       'మాట్లాడు',
    r'\blisten\b':     'వినండి',
    r'\bshare\b':      'పంచుకో',
    r'\bgive\b':       'ఇవ్వు',
    r'\btake\b':       'తీసుకో',
    r'\bmake\b':       'చేయి',
    r'\bneed\b':       'అవసరం',
    r'\bwant\b':       'కావాలి',
    r'\blike\b':       'ఇష్టం',
    r'\bsorry\b':      'క్షమించండి',
    r'\bplease\b':     'దయచేసి',
    r'\bthank\b':      'ధన్యవాదాలు',
    r'\bthanks\b':     'ధన్యవాదాలు',
    r'\bwhy\b':        'ఎందుకు',
    r'\bwhat\b':       'ఏమి',
    r'\bhow\b':        'ఎలా',
    r'\bwhen\b':       'ఎప్పుడు',
    r'\bwhere\b':      'ఎక్కడ',
    r'\bwho\b':        'ఎవరు',
}


# ─────────────────────────────────────────
# Clean text — remove ALL TTS-breaking chars
# ─────────────────────────────────────────
def clean_for_tts(text: str) -> str:

    # 1. Replace known English words with Telugu
    for pattern, replacement in ENGLISH_REPLACEMENTS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 2. Remove remaining ASCII letters
    text = re.sub(r'[a-zA-Z]+', '', text)

    # 3. Fix ellipsis — THE cause of "ayo chela chelai" mispronunciation
    text = re.sub(r'\.{2,}', '。', text)   # temp placeholder
    text = re.sub(r'。', ' ', text)         # replace with space

    # 4. Remove dashes that TTS reads aloud
    text = re.sub(r'[—–\-]{1,}', ' ', text)

    # 5. Remove brackets, quotes, special symbols
    text = re.sub(r'[(){}\[\]""\"\'`~@#$%^&*_+=<>|\\\/]', '', text)

    # 6. Remove numbers (TTS reads them as gibberish in Telugu)
    text = re.sub(r'[0-9]+', '', text)

    # 7. Keep only Telugu unicode chars + basic sentence punctuation
    #    Telugu unicode range: \u0C00-\u0C7F
    #    Keep:  Telugu chars, spaces, . ! ? ।
    text = re.sub(r'[^\u0C00-\u0C7F\s.!?।౦-౯]', '', text)

    # 8. Fix repeated punctuation left behind
    text = re.sub(r'[.!?।]{2,}', '.', text)

    # 9. Fix multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)

    # 10. Fix space before punctuation
    text = re.sub(r'\s([.!?।])', r'\1', text)

    return text.strip()


# ─────────────────────────────────────────
# Split text into chunks
# ─────────────────────────────────────────
def split_into_chunks(text: str) -> list:

    sentences = re.split(r'(?<=[.!?।])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks  = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= MAX_CHARS_PER_CHUNK:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            if len(sentence) > MAX_CHARS_PER_CHUNK:
                parts = re.split(r'(?<=[,])\s*', sentence)
                sub   = ""
                for part in parts:
                    if len(sub) + len(part) + 1 <= MAX_CHARS_PER_CHUNK:
                        sub = (sub + " " + part).strip()
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = part.strip()
                if sub:
                    chunks.append(sub)
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks if chunks else [text.strip()]


# ─────────────────────────────────────────
# Synthesize one chunk
# ─────────────────────────────────────────
def synthesize_chunk(text: str) -> np.ndarray:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length
    )
    with torch.no_grad():
        output = model(**inputs).waveform
    return output.squeeze().cpu().numpy()


# ─────────────────────────────────────────
# Main TTS entry point
# ─────────────────────────────────────────
def generate_telugu_audio(text: str) -> str:

    if not text or not text.strip():
        return ""

    # ── Clean first ──
    cleaned = clean_for_tts(text)
    print(f"TTS original  : {text[:100]}")
    print(f"TTS cleaned   : {cleaned[:100]}")

    if not cleaned.strip():
        print("TTS: nothing left after cleaning")
        return ""

    chunks = split_into_chunks(cleaned)
    print(f"TTS: {len(chunks)} chunk(s) for {len(cleaned)} chars")

    waveforms     = []
    silence_short = np.zeros(int(SAMPLE_RATE * 0.20))
    silence_long  = np.zeros(int(SAMPLE_RATE * 0.40))

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        print(f"  [{i+1}/{len(chunks)}] {chunk[:80]}{'...' if len(chunk)>80 else ''}")
        try:
            wav = synthesize_chunk(chunk)
            waveforms.append(wav)
            if i < len(chunks) - 1:
                ends_sentence = chunk[-1] in '.!?।'
                waveforms.append(silence_long if ends_sentence else silence_short)
        except Exception as e:
            print(f"  Chunk {i+1} failed: {e}")
            continue

    if not waveforms:
        print("TTS failed: no waveforms generated")
        return ""

    full_waveform = np.concatenate(waveforms).astype(np.float32)

    max_val = np.max(np.abs(full_waveform))
    if max_val > 0:
        full_waveform = (full_waveform / max_val * 0.92).astype(np.float32)

    media_root = settings.MEDIA_ROOT
    os.makedirs(media_root, exist_ok=True)
    filename = f"audio_{uuid.uuid4().hex}.wav"
    filepath = os.path.join(media_root, filename)

    print("Saving audio...")
    scipy.io.wavfile.write(filepath, rate=SAMPLE_RATE, data=full_waveform)

    duration = len(full_waveform) / SAMPLE_RATE
    print(f"TTS saved → {filepath} ({duration:.2f}s)")

    return f"{settings.MEDIA_URL.rstrip('/')}/{filename}"