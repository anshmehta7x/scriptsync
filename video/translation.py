import googletrans
from difflib import SequenceMatcher  # For fuzzy matching

class TranslationCache:
    def __init__(self):
        self.cache = {}
        self.translator = googletrans.Translator()
        self.similarity_threshold = 0.8  # for fuzzy mathching

    def _find_fuzzy_match(self, text):
        for cached_text in self.cache:
            similarity = SequenceMatcher(None, text, cached_text).ratio()
            if similarity >= self.similarity_threshold:
                return cached_text
        return None

    def translate(self, text, src, dest):
        # Check for an exact or fuzzy match in the cache
        match = self._find_fuzzy_match(text)
        if match:
            return self.cache[match]

        # If no match is found, translate the text and cache the result
        translation = self.translator.translate(text, src=src, dest=dest)
        self.cache[text] = translation.text
        return translation.text
