from deep_translator import GoogleTranslator
from thefuzz import fuzz


class TranslationCache:
    def __init__(self):
        self.cache = {}
        self.translator = GoogleTranslator()
        self.source = 'en'
        self.dest = 'fr'
        self.supported_sources = ['en', 'fr', 'es', 'de']
        self.supported_dest = ['en', 'fr', 'es', 'de']

    def set_source(self, source):
        if source in self.supported_sources:
            self.source = source
        else:
            raise ValueError(f"Unsupported source language: {source}")

    def set_dest(self, dest):
        if dest in self.supported_dest:
            self.dest = dest
        else:
            raise ValueError(f"Unsupported destination language: {dest}")

    def fuzzy_match_transaltions(self, text_to_match):
        for source in self.cache.keys():
            if fuzz.ratio(source, text_to_match) > 80:
                return self.cache[source]

        return False

    def translate(self, text):
        match = self.fuzzy_match_transaltions(text)
        if match:
            return match

        translation = self.translator.translate(text, source=self.source, target=self.dest)
        self.cache[text] = translation
        return translation
