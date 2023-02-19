"""
    Speach to Text transcription
        Link to Huggingface example: https://huggingface.co/facebook/wav2vec2-base-960h
        Link to the code: https://www.analyticsvidhya.com/blog/2021/02/hugging-face-introduces-the-first-automatic-speech-recognition-model-wav2vec2/
"""
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import librosa
import project_functions as pf
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


# load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

fname = "The magic of truth and lies (and iPods) ｜ Marco Tempest.wav"; duration = 307

# Read the sound file
# speech, rate = librosa.load(fname, sr=16000, offset=duration / 2 - 10, duration=70)
segment_length = 70; start_at = 12  # Load "segment_length" seconds of a file, starting at "start_at" seconds
speech, rate = librosa.load(fname, sr=16000, offset=start_at, duration=segment_length)

# Tokenize the waveform
input_values = tokenizer(speech, return_tensors='pt').input_values

# Retrieve logits from the model
logits = model(input_values).logits

# Take argmax value and decode into transcription
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)

# Print the output
#print(transcription)

text = ''.join(transcription)

spaCy_summarizer = pf.SummarizeSpaCy()
freq_words = spaCy_summarizer.tokenize_text(text)  # For the entire Document
sent_strength = spaCy_summarizer.find_sent_strength(text)  # For each Section
summary_spaCy = spaCy_summarizer.summarize(7)  # For each Section
print(freq_words)
print(sent_strength)
print(summary_spaCy)

print("End")

