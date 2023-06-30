from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel

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

def load_clip_model():
    print("Loading clip model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor, clip_tokenizer