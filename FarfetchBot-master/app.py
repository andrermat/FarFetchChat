from flask import Flask, request
import json
from PIL import Image
import numpy as np
import torch
from langdetect import detect


from models.model_utils import get_model
from transformers import AutoTokenizer, pipeline
import config
import transformers

import openai
openai.api_key = open("./key.txt", "r").read()


import urllib.request

import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

from flask_cors import CORS

import pickle

from utils.SearchSlots import SearchSlots, SlotManager
from utils.History import init_GPT_History, History
from utils.LoadModels import load_translators, load_language_detection, load_clip_model, load_voice_recognition, load_slot_classifier, load_intent_classifier
from utils.Utils import parseVoice, parseData, saveState, loadState, createConn, parseProductResponse, extract_id, summarize
from utils.Translation import Translator
from utils.ModelReply import msg_history_to_davinci, GPTrequest, davinciRequest, chatReply
from utils.IntentDetection import mainIntentDetector, intentDetectorModel, loadIntentDetector,intentDetectorModelv2



def load_translators(flag):
    if flag:
        print("Loading translators...")
        print("Loading pt translator...")
        pt_tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/translation-pt-en-t5")
        pt_model = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/translation-pt-en-t5")
        print("Loading de translator...")
        de_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        de_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        print("Loading es translator...")
        es_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
        es_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")
        print("Loading fr translator...")
        fr_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        fr_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        print("Loading ru translator...")
        ru_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
        ru_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
        return pt_tokenizer, pt_model, de_tokenizer, de_model, es_tokenizer, es_model, fr_tokenizer, fr_model, ru_tokenizer, ru_model
    else:
        print("Translators not preloaded, will load on demand...")
        return None, None, None, None, None, None, None, None, None, None
    
def load_language_detection():
    print("Loading language detector ...")
    lang_detect_tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
    lang_detect_model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")
    return lang_detect_tokenizer, lang_detect_model

LANGUAGE_SUPPORT = False
VOICE_SUPPORT = False
PRE_LOAD_TRANSLATORS = False
LOAD_PREVIOUS_STATE = False
SAVE_STATE = False
SKIP_NEGATIVE_SLOTS = True

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)
if __name__ == '__main__':
      app.run(host='127.0.0.1', port=4000)


state = {
    "GPT_history" : init_GPT_History(),
    "search_slots" : SearchSlots(),
    "previous_results" : []
}
slotManager = SlotManager()

if LOAD_PREVIOUS_STATE:
    state = loadState(state)
else:
    # Clear state
    saveState(state)



if LANGUAGE_SUPPORT:
    lang_detect_tokenizer, lang_detect_model = load_language_detection()
    pt_tokenizer, pt_model, de_tokenizer, de_model, es_tokenizer, es_model, fr_tokenizer, fr_model, ru_tokenizer, ru_model = load_translators(PRE_LOAD_TRANSLATORS)

#translator = Translator(PRE_LOAD_TRANSLATORS)

slot_FTmodel, bart_FTtokenizer = load_slot_classifier()
intent_FTmodel = load_intent_classifier()

clip_model, clip_processor, clip_tokenizer = load_clip_model()

intent_tokenizer, intent_model, intent_input_function = loadIntentDetector()
voice_recognition_model = load_voice_recognition()

print("--Ready!--")

def sendMessage(message, recomendations = ""):
    global state
    responseDict = { "has_response": True, "recommendations":recomendations,
    "response":message, "system_action":""}
    jsonString = json.dumps(responseDict)
    saveState(state)
    return jsonString

@app.route('/set/', methods=['GET','POST'])
def set_key():
    key = request.args.get('key')
    openai.api_key = key
    return "Key was set"

def test(query, size=10):
    print("Executing text query...")
    global clip_model, clip_tokenizer, state
    client, index_name = createConn()

    #inputs = processor(text=[search_query], images=[], return_tensors="pt", padding=True)

    inputs = clip_tokenizer([query], padding=True, return_tensors="pt")
    text_features = F.normalize(clip_model.get_text_features(**inputs))
    comb_embeds = text_features[0].detach().numpy().tolist()


    query_denc = {
    'size': size,
    '_source': ['product_id', 'product_family', 'product_category', 'product_sub_category', 'product_gender', 
                'product_main_colour', 'product_second_color', 'product_brand', 'product_materials', 
                'product_short_description', 'product_attributes', 'product_image_path', 
                'product_highlights', 'outfits_ids', 'outfits_products', 'image_embedding'],
    "query": {
            "knn": {
                "combined_embedding": {
                    "vector": comb_embeds,
                    "k": 2
                }
            }
        },
        "post_filter": {
            "term": {
            "product_brand": "nike"
            }
        }
        
    }

    response = client.search(
        body = query_denc,
        index = index_name
    )

    results = [r['_source'] for r in response['hits']['hits']]
    
    return results



@app.route('/', methods=['GET','POST'])
def main():
    global state
    loadState(state)
    if request.is_json:

        mainMenuText = '''
        Hello! My name is iFetch bot and i will be your assistant.
        I can talk to you in english, portuguese, spanish, french, german, russian.
        Also feel free to talk to me using voice or send me a picture of something you're looking for.
        How can i help you today?
        
        '''
        data = request.json       

        utt, file_type, file_extention, has_audio, audio_file_extention = parseData(data)

        try:
            if has_audio and VOICE_SUPPORT :
                utt = parseVoice(audio_file_extention, voice_recognition_model)
                print("Voice transcription: ", utt)
            if file_type != None:
                if file_type == "image":
                    print("Processing image...")
                    state["search_slots"].clean()
                    query, lang = translate(utt)
                    state["GPT_history"].add(f"Your next reply should be in {lang}", "system")
                    state["GPT_history"].add(utt, "user")
                    intent, slots = mainIntentDetector(query,intent_tokenizer, intent_model, intent_input_function)
                    state["search_slots"].clean()
                    state = slotManager.parseSlots(query, slots,state)
                    
                    state["GPT_history"].add(f"The user uploaded an image and the system is looking for matches", "system")
                    results = imageSearch(file_extention=file_extention)
                    state["previous_results"] = results
                    recomendations, system_prompt = parseProductResponse(results)
                    state["GPT_history"].add(system_prompt, "system")
                    reply = chatReply(state)
                    state["GPT_history"].add(reply, "assistant")
                    i = 1
                    for item in recomendations:
                        reply += f"Item {i} id: " + str(item["id"]) + "\n"
                        i+= 1
                    return sendMessage(reply, recomendations)
                elif file_type == "unknown":
                    return sendMessage("I'm sorry, it seems like you uploaded a file format that is not supported.")

            if "TEST" in utt:
                utt = utt.split(":")[1].strip()
                state["GPT_history"].add(utt, "user")
                summarized = summarize(utt)
                results = text_query(summarized)
                recomendations, system_prompt = parseProductResponse(results)
                state["GPT_history"].add(system_prompt, "system")
                reply = chatReply(state)
                state["GPT_history"].add(reply, "assistant")
                return sendMessage(reply, recomendations)
            if utt == "Hi!" or utt == "":
                return sendMessage(mainMenuText)
            else:
                query, lang = translate(utt)
                state["GPT_history"].add(f"Your next reply should be in {lang}", "system")
                intent, slots = mainIntentDetector(query,intent_tokenizer, intent_model, intent_input_function)
                return intentMenu(intent, query, slots)
        except Exception as e:
            print(e)
            print(e.with_traceback())
            return sendMessage("Something went wrong while processing your request.")

    return sendMessage("Something went wrong while processing your request.")

def intentMenu(intent, utt, slots):
    global state
    # user_neutral_are_you_a_bot, user_neutral_do_you_have_pets, user_neutral_fun_fact, "user_neutral_goodbye, user_neutral_greeting, user_neutral_how_old_are_you, user_neutral_meaning_of_life
    # user_neutral_tell_joke, user_neutral_what_are_your_hobbies, user_neutral_what_is_your_name, user_neutral_where_are_you_from, user_neutral_who_do_you_work_for
    # user_neutral_who_made_you,user_neutral_oos, user_neutral_what_can_i_ask_you
    state["GPT_history"].add(utt, "user")
    shop_intents = ["user_inform_product_attribute", "user_inform_product_id", "user_qa_check_information", "user_qa_product_composition", 
                    "user_qa_product_description", "user_qa_product_information", "user_qa_product_measurement", "user_request_get_products"]
    
    if intent in shop_intents:
        shop_intent = intentDetectorModel(utt)
        print("Shop intent v2: ", intentDetectorModelv2(utt, intent_FTmodel, bart_FTtokenizer))
        if "search" in shop_intent:
            state["search_slots"].clean()
            state = slotManager.parseSlots(utt, slots, state)
            slotManager.parseSlotsv2(utt, slots, slot_FTmodel, bart_FTtokenizer)
            results = text_query(utt)

        elif "more" in shop_intent:
            state = slotManager.parseSlots(utt, slots, state)
            print("previous results: ", state["previous_results"])
            if len(state["previous_results"]) > 0:
                results = imageSearch(size = 3)
                results.pop(0)
            results = text_query(utt, size = 3)
        elif "information" in shop_intent:
            state["search_slots"].clean()
            if len(state["previous_results"]) == 0:
                state["GPT_history"].add("item id is required for more information", "system")
                reply = chatReply(state)
                return sendMessage(reply)
            else:
                state["GPT_history"].add("User wants a detailed explanation of the following items", "system")
                results = state["previous_results"]
        elif "id" in shop_intent:
            state["search_slots"].clean()
            id = extract_id(utt)
            results = id_search(id)
        elif "continue" in shop_intent:
            state = slotManager.parseSlots(utt, slots, state)
            results = text_query(utt)
        elif "buy" in shop_intent:
            state["GPT_history"].add("Buying products through the chat bot is not yet possible", "system")
            reply = chatReply(state)
            return sendMessage(reply)
        state["previous_results"] = results
        recomendations, system_prompt = parseProductResponse(results)
        state["GPT_history"].add(system_prompt, "system")
        reply = chatReply(state)
        state["GPT_history"].add(reply, "assistant")
        i = 1
        for item in recomendations:
            reply += f"Item {i} id: " + str(item["id"]) + "\n"
            i+= 1
        return sendMessage(reply, recomendations)
    else:
        reply = chatReply(state)
        state["GPT_history"].add(reply, "assistant")
        return sendMessage(reply)




def id_search(qtxt):
    print("Executing id search...")
    client, index_name = createConn()

    query_denc = {
    'size': 3,
    '_source': ['product_id', 'product_family', 'product_category', 'product_sub_category', 'product_gender', 
                'product_main_colour', 'product_second_color', 'product_brand', 'product_materials', 
                'product_short_description', 'product_attributes', 'product_image_path', 
                'product_highlights', 'outfits_ids', 'outfits_products'],
    'query': {
        'multi_match': {
          'query': qtxt,
          'fields': ['product_id']
          }
        }
    }

    response = client.search(
        body = query_denc,
        index = index_name
    )

    results = [r['_source'] for r in response['hits']['hits']]

    return results

# Searches for items that match the query
def text_query(search_query, size=1):
    print("Executing text query...")
    global clip_model, clip_tokenizer, state
    client, index_name = createConn()

    #inputs = processor(text=[search_query], images=[], return_tensors="pt", padding=True)
    positive_slots, negative_slots = state["search_slots"].get()
    if len(positive_slots) > 0:
        positive_str = ""
        for key in positive_slots:
            positive_str += positive_slots[key] + " "
        print("Positive: ", positive_str)
        pos = clip_tokenizer([positive_str], padding=True, return_tensors="pt")
        pos_features = F.normalize(clip_model.get_text_features(**pos))
        pos_embeds = pos_features[0].detach().numpy().tolist()
    else:
        query = summarize(search_query)
        if "none" in query.lower():
            query = search_query
        print("Positive: ", query)
        inputs = clip_tokenizer([query], padding=True, return_tensors="pt")
        text_features = F.normalize(clip_model.get_text_features(**inputs))
        pos_embeds = text_features[0].detach().numpy().tolist()

    if len(negative_slots) > 0:
        negative_str = ""
        for key in negative_slots:
            negative_str += negative_slots[key] + " "
        print("Negative: ", negative_str)
        neg = clip_tokenizer([negative_str], padding=True, return_tensors="pt")
        neg_features = F.normalize(clip_model.get_text_features(**neg))
        neg_embeds = neg_features[0].detach().numpy().tolist()

        embeds = torch.tensor(np.array(pos_embeds) - np.array(neg_embeds))
        comb_embeds = F.normalize(embeds, dim=0).to(torch.device('cpu')).numpy()
    else:
        print("Negative: None")
        comb_embeds = pos_embeds

    query_denc = {
    'size': size,
    '_source': ['product_id', 'product_family', 'product_category', 'product_sub_category', 'product_gender', 
                'product_main_colour', 'product_second_color', 'product_brand', 'product_materials', 
                'product_short_description', 'product_attributes', 'product_image_path', 
                'product_highlights', 'outfits_ids', 'outfits_products', 'image_embedding'],
    "query": {
            "knn": {
            "combined_embedding": {
                "vector": comb_embeds,
                "k": 2
            }
            }
        }
    }

    response = client.search(
        body = query_denc,
        index = index_name
    )

    results = [r['_source'] for r in response['hits']['hits']]
    return results

def imageSearch(file_extension = None, size = 1):
    print("Executing image search...")
    global clip_model, clip_tokenizer, clip_processor, state
    client, index_name = createConn()
    if file_extension == None:
        img_embeds = state["previous_results"][0]["image_embedding"]
    else:
        img = Image.open(f"img.{file_extension}")
        input_img = clip_processor(images=img, return_tensors="pt")
        embeddings_img = F.normalize(clip_model.get_image_features(**input_img))
        img_embeds = embeddings_img[0].detach().numpy()

    pos_embeds = []
    neg_embeds = []
    positive_slots, negative_slots = state["search_slots"].get()

    if len(positive_slots) > 0:
        positive_str = ""
        for key in positive_slots:
            positive_str += positive_slots[key] + " "
        print("Positive: ", positive_str)
        pos = clip_tokenizer([positive_str], padding=True, return_tensors="pt")
        pos_features = F.normalize(clip_model.get_text_features(**pos))
        pos_embeds = pos_features[0].detach().numpy().tolist()

    if len(negative_slots) > 0:
        negative_str = ""
        for key in negative_slots:
            negative_str += negative_slots[key] + " "
        print("Negative: ", negative_str)
        neg = clip_tokenizer([negative_str], padding=True, return_tensors="pt")
        neg_features = F.normalize(clip_model.get_text_features(**neg))
        neg_embeds = neg_features[0].detach().numpy().tolist()

    if pos_embeds == [] and neg_embeds == []:
        comb_embeds = img_embeds
    elif pos_embeds == []:
        embeds = torch.tensor(np.array(img_embeds) - np.array(neg_embeds))
        comb_embeds = F.normalize(embeds, dim=0).to(torch.device('cpu')).numpy()
    elif neg_embeds == []:
        embeds = torch.tensor(np.array(img_embeds) + np.array(pos_embeds))
        comb_embeds = F.normalize(embeds, dim=0).to(torch.device('cpu')).numpy()
    else:
        embeds = torch.tensor(np.array(img_embeds) + np.array(pos_embeds) - np.array(neg_embeds))
        comb_embeds = F.normalize(embeds, dim=0).to(torch.device('cpu')).numpy()


    query_denc = {
    'size': size,
    '_source': ['product_id', 'product_family', 'product_category', 'product_sub_category', 'product_gender', 
                'product_main_colour', 'product_second_color', 'product_brand', 'product_materials', 
                'product_short_description', 'product_attributes', 'product_image_path', 
                'product_highlights', 'outfits_ids', 'outfits_products', 'image_embedding'],
    "query": {
            "knn": {
            "combined_embedding": {
                "vector": comb_embeds,
                "k": 5
            }
            }
        }
    }

    response = client.search(
        body = query_denc,
        index = index_name
    )

    results = [r['_source'] for r in response['hits']['hits']]


    return results



def languageDetectionv2(text):
    lang = detect(text)
    return lang

def languageDetection(text):
    print("Detecting language...")
    global lang_detect_model, lang_detect_tokenizer

    inputs = lang_detect_tokenizer(text,padding=True, return_tensors="pt")
    logits = lang_detect_model(**inputs).logits
    id = logits.argmax().item()
    lang = lang_detect_model.config.id2label[id]
    print("Language detected: ", lang)
    return lang

def translate(text):
    if not LANGUAGE_SUPPORT:
        return text, "english"
    global pt_tokenizer, pt_model, de_tokenizer, de_model, es_tokenizer, es_model, fr_tokenizer, fr_model, ru_tokenizer, ru_model

    translated = text
    language = "english"

    #labels = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
    lang = languageDetection(text)

    if lang not in ["en", "pt", "de", "es", "fr", "ru"]:
        print("Language not supported, trying v2...")
        lang = languageDetectionv2(text)
        print("Language detected v2: ", lang)
    
    print("Translating...")
    if lang == "pt":
        from transformers import pipeline
        if pt_model == None:
            print("Loading pt model...")
            pt_tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/translation-pt-en-t5")
            pt_model = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/translation-pt-en-t5")

        tokenizer = pt_tokenizer
        model = pt_model

        pten_pipeline = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

        ans = pten_pipeline("translate Portuguese to English: "+ text)
        translated = ans[0]['generated_text']
        language = "portuguese"
    else:
        if lang == "de":
            if de_model == None:
                print("Loading de model...")
                de_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
                de_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")
            tokenizer = de_tokenizer
            model = de_model
            language = "german"
        elif lang == "es":
            if es_model == None:
                print("Loading es model...")
                es_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
                es_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")
            tokenizer = es_tokenizer
            model = es_model
            language = "spanish"
        elif lang == "fr":
            if fr_model == None:
                print("Loading fr model...")
                fr_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
                fr_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            tokenizer = fr_tokenizer
            model = fr_model
            language = "french"
        elif lang == "ru":
            if ru_model == None:
                print("Loading ru model...")
                ru_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
                ru_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
            tokenizer = ru_tokenizer
            model = ru_model
            language = "russian"
        else:
            return text, "english"

        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(**inputs)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Translated: ", translated)
    return translated, language

