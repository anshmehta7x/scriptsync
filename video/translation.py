from googletrans import Translator
from tqdm import tqdm

translator = Translator()
translation_cache = {}

def translate_text(text, dest='fr', max_retries=1):
    if not text or text.strip() == "":
        return ""
    if text in translation_cache:
        return translation_cache[text]
    
    for attempt in range(max_retries):
        try:
            translation = translator.translate(text, dest=dest).text
            translation_cache[text] = translation
            return translation
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error translating text: {text}. Retrying... Error: {str(e)}")
            else:
                print(f"Failed to translate text after {max_retries} attempts: {text}. Error: {str(e)}")
                return text  # Return original text if translation fails

def create_translate_cache(ocr_results, dest='fr'):
    global translation_cache
    texts = set()
    
    # Extract unique texts from OCR results
    for frame_result in ocr_results:
        for detection in frame_result:
            if isinstance(detection, dict) and 'text' in detection:
                texts.add(detection['text'])
    
    with tqdm(total=len(texts), desc="Creating Translation Cache") as pbar:
        for text in texts:
            translation = translate_text(text, dest=dest)
            translation_cache[text] = translation
            pbar.update(1)
    
    return translation_cache