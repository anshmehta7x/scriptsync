from googletrans import Translator
from tqdm import tqdm
import logging

translator = Translator()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

translation_cache = {}

def translate_text(text, dest='fr',src='en', max_retries=1):
    """
    Translate text to the specified destination language with caching and error handling.

    Args:
        text (str): Text to translate.
        dest (str): Destination language code (default is 'fr' for French).
        max_retries (int): Maximum retry attempts on error (default is 1).

    Returns:
        str: Translated text or original text if translation fails.
    """
    if not text or text.strip() == "":
        return ""

    if text in translation_cache:
        return translation_cache[text]

    for attempt in range(max_retries):
        try:
            translation = translator.translate(text,src=src, dest=dest).text
            translation_cache[text] = translation
            return translation
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Error translating text: {text}. Retrying... Error: {str(e)}")
            else:
                logger.error(f"Failed to translate text after {max_retries} attempts: {text}. Error: {str(e)}")
                return text  # Return original text if translation fails

def create_translate_cache(ocr_results,src='en', dest='fr'):
    """
    Create translation cache for unique texts extracted from OCR results.

    Args:
        ocr_results (list): List of dictionaries containing OCR results.
        dest (str): Destination language code (default is 'fr' for French).

    Returns:
        dict: Translation cache mapping original text to translated text.
    """
    global translation_cache
    texts = set()

    # Extract unique texts from OCR results
    for frame_result in ocr_results:
        for detection in frame_result:
            if isinstance(detection, dict) and 'text' in detection:
                texts.add(detection['text'])

    with tqdm(total=len(texts), desc="Creating Translation Cache") as pbar:
        for text in texts:
            translation = translate_text(text,src=src ,dest=dest)
            translation_cache[text] = translation
            pbar.update(1)

    return translation_cache
