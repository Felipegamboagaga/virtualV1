import whisper
import gradio as gr 
import time
from pyChatGPT import ChatGPT
import warnings
import openai
from gtts import gTTS
import torch
torch.cuda.empty_cache()
openai.api_key='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'
###Defining Variables

warnings.filterwarnings("ignore")
model = whisper.load_model("large-v2")

def transcribe(audio):

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    result_text = result.text

    resultGPT = openai.Completion.create(
                model="text-davinci-003",
                prompt=result_text,
                temperature=0.6,
                max_tokens=3000
                )
    out_result = resultGPT.choices[0]['text']

    out_audio = gTTS(text=out_result, lang='es-us',slow=False)
    out_audio.save("output.mp3")

    
    return [result_text, out_result, 'output.mp3']

output_1 = gr.Textbox(label="Speech to Text")
output_2 = gr.Textbox(label="ChatGPT Output")
output_3 = gr.Audio(label='tu voz', placeholder='name here')



gr.Interface(
    title = 'Vainito OpenAi', 
    fn=transcribe, 
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath")
    ],

    outputs=[
        output_1,  output_2, output_3,
    ],
    live=True).launch()