from transformers import pipeline
import urllib.request
import openai
import pickle
from opensearchpy import OpenSearch

def parseVoiceOpenAI(file_extention, has_audio = False):
    if has_audio:
        file = open(f"./voice1.mpeg", "rb")
    else:
        file = open(f"./voice.{file_extention}", "rb")
    transcription = openai.Audio.transcribe("whisper-1", file)
    return transcription.text

def parseVoice(file_extention):
    model = pipeline("automatic-speech-recognition", model="openai/whisper-medium")
    out = model(f"voice1.{file_extention}")
    return out["text"]

def parseData(data):
    utt = data.get('utterance')
    file = data.get('file')
    audio_file = data.get("audio")
    has_audio = False
    file_type = None
    file_extention = None
    if file != None:
        file_type = "unknown" # default value to return
        file_info = file.split(",/")[0]
        file_extention = file_info.split(";")[0].split("/")[1].strip()
        extracted_file_type = file_info.split("/")[0].split(":")[1].strip()
        if extracted_file_type == "image":
            file_type = "image"
            urllib.request.urlretrieve(file, f"img.{file_extention}")
        elif extracted_file_type == "audio":
            has_audio = True
            audio_file_extention = file_extention
            urllib.request.urlretrieve(file, f"voice.{file_extention}")
        else:
            extracted_file_type = "unknown"
            urllib.request.urlretrieve(file, f"unknown.{file_extention}")

    if(audio_file != None):    
        try:
            has_audio = True
            audio_file_info = audio_file.split(",/")[0]
            audio_file_extention = audio_file_info.split(";")[0].split("/")[1].strip()
            audio_file_type = audio_file_info.split("/")[0].split(":")[1].strip()
            print("Audio file info: ", audio_file_info)
            print("Audio file extention: ", audio_file_extention)
            print("Audio file type: ", audio_file_type)
            urllib.request.urlretrieve(audio_file, f"voice1.{audio_file_extention}")
        except:
            has_audio=False
            print("File retrieval failed:")
        
    return utt, file_type, file_extention, has_audio, audio_file_extention

def saveState(state):
    with open('state.pkl', 'wb') as f:
        pickle.dump(state, f)

def loadState(state):
    try:
        with open('state.pkl', 'rb') as f:
            state = pickle.load(f)
    except:
        print("Failed to load state")
        
    return state

def createConn():
    host = 'api.novasearch.org'
    port = 443

    index_name = "farfetch_images"

    user = 'ifetch' # Add your user name here.
    password = 'S48YdnMQ' # Add your user password here. For testing only. Don't store credentials in code.
    client = OpenSearch(
        hosts = [{'host': host, 'port': port}],
        http_compress = True,
        http_auth = (user, password),
        url_prefix = 'opensearch',
        use_ssl = True,
        verify_certs = False,
        ssl_assert_hostname = False,
        ssl_show_warn = False
    )
    return client, index_name