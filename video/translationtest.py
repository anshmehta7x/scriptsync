from translator import TranslationCache
import sys
import time

def print_header(test_name):
    print("\n" + "=" * 50)
    print(f"TEST: {test_name}")
    print("=" * 50)

def print_result(description, success):
    status = "PASS" if success else "FAIL"
    color = "\033[92m" if success else "\033[91m"
    print(f"{color}{status}\033[0m - {description}")

def run_test_translation_cache_init():
    print_header("TranslationCache Initialization")

    cache = TranslationCache()
    print(f"Cache initialized: {cache.cache}")
    print(f"Source language: {cache.source}")
    print(f"Destination language: {cache.dest}")
    print(f"Supported source languages: {cache.supported_sources}")
    print(f"Supported destination languages: {cache.supported_dest}")

    print_result("Cache is empty", cache.cache == {})
    print_result("Default source is 'en'", cache.source == 'en')
    print_result("Default destination is 'fr'", cache.dest == 'fr')
    print_result("English is supported source", 'en' in cache.supported_sources)
    print_result("French is supported destination", 'fr' in cache.supported_dest)

def run_test_set_source_language():
    print_header("Set Source Language")

    cache = TranslationCache()
    original_source = cache.source
    print(f"Original source: {original_source}")

    try:
        cache.set_source('es')
        print(f"New source after setting to 'es': {cache.source}")
        print_result("Source changed to 'es'", cache.source == 'es')
    except Exception as e:
        print(f"ERROR: {e}")
        print_result("Should set valid source language", False)

    try:
        cache.set_source('invalid_lang')
        print_result("Should raise ValueError for invalid source", False)
    except ValueError as e:
        print(f"Expected error raised: {e}")
        print_result("Correctly rejected invalid source language", True)
    except Exception as e:
        print(f"Wrong error type: {e}")
        print_result("Should raise ValueError specifically", False)

def run_test_set_destination_language():
    print_header("Set Destination Language")

    cache = TranslationCache()
    original_dest = cache.dest
    print(f"Original destination: {original_dest}")

    try:
        cache.set_dest('de')
        print(f"New destination after setting to 'de': {cache.dest}")
        print_result("Destination changed to 'de'", cache.dest == 'de')
    except Exception as e:
        print(f"ERROR: {e}")
        print_result("Should set valid destination language", False)

    try:
        cache.set_dest('invalid_lang')
        print_result("Should raise ValueError for invalid destination", False)
    except ValueError as e:
        print(f"Expected error raised: {e}")
        print_result("Correctly rejected invalid destination language", True)
    except Exception as e:
        print(f"Wrong error type: {e}")
        print_result("Should raise ValueError specifically", False)

def run_test_fuzzy_matching():
    print_header("Fuzzy Matching")

    cache = TranslationCache()
    test_cache = {
        "Hello world": "Bonjour le monde",
        "How are you?": "Comment allez-vous?",
        "Good morning": "Bonjour",
        "Good night": "Bonne nuit",
        "Thank you very much": "Merci beaucoup"
    }
    cache.cache = test_cache
    print(f"Cache initialized with {len(cache.cache)} entries: {cache.cache}")

    exact_match = cache.fuzzy_match_transaltions("Hello world")
    print(f"Exact match result: {exact_match}")
    print_result("Exact match works", exact_match == "Bonjour le monde")

    fuzzy_match = cache.fuzzy_match_transaltions("Hello worl")
    print(f"Fuzzy match result: {fuzzy_match}")
    print_result("Fuzzy match works with similar text", fuzzy_match == "Bonjour le monde")

    fuzzy_match2 = cache.fuzzy_match_transaltions("How are you")
    print(f"Fuzzy match result 2: {fuzzy_match2}")
    print_result("Fuzzy match works with missing punctuation", fuzzy_match2 == "Comment allez-vous?")

    fuzzy_match3 = cache.fuzzy_match_transaltions("Good mornings")
    print(f"Fuzzy match result 3: {fuzzy_match3}")
    print_result("Fuzzy match works with added letter", fuzzy_match3 == "Bonjour")

    no_match = cache.fuzzy_match_transaltions("Something completely different")
    print(f"No match result: {no_match}")
    print_result("No match returns False for dissimilar text", no_match is False)

def run_test_translate_with_cache_building():
    print_header("Translation with Cache Building")

    cache = TranslationCache()

    test_texts = [
        "Hello world",
        "How are you?",
        "Good morning",
        "Good night",
        "Thank you very much",
        "What is your name?",
        "My name is John",
        "I live in Paris",
        "The weather is nice today",
        "I like to eat pizza",
        "The book is on the table",
        "Can you help me?",
        "Where is the train station?",
        "How much does this cost?",
        "I don't understand"
    ]

    print(f"Starting cache size: {len(cache.cache)}")

    for idx, text in enumerate(test_texts):
        print(f"\nTranslation {idx+1}/{len(test_texts)}: '{text}'")
        translation = cache.translate(text)
        print(f"Result: '{translation}'")

        if idx % 5 == 0 and idx > 0:
            print(f"Current cache size: {len(cache.cache)}")

    print(f"\nFinal cache size: {len(cache.cache)}")
    print_result("Cache contains all entries", len(cache.cache) == len(test_texts))

    print("\nFull cache contents:")
    for key, value in cache.cache.items():
        print(f"'{key}' -> '{value}'")

    print("\nTesting fuzzy matching on built cache...")
    fuzzy_tests = [
        ("Hello worl", "Hello world"),
        ("How are you", "How are you?"),
        ("Good morn", "Good morning"),
        ("My name is Johnny", "My name is John"),
        ("I live in Pari", "I live in Paris")
    ]

    for fuzzy_text, expected_match in fuzzy_tests:
        match = cache.fuzzy_match_transaltions(fuzzy_text)
        translation = cache.cache.get(expected_match, "Not found")
        print(f"Fuzzy '{fuzzy_text}' -> Expected original: '{expected_match}' -> Translation: '{translation}'")
        print_result(f"Fuzzy match for '{fuzzy_text}'", match == translation)

def run_test_different_language_pairs():
    print_header("Different Language Pair Tests")

    lang_pairs = [
        ("en", "fr"),
        ("en", "es"),
        ("en", "de"),
        ("fr", "en")
    ]

    test_text = "Hello, how are you?"

    for source, dest in lang_pairs:
        cache = TranslationCache()

        try:
            cache.set_source(source)
            cache.set_dest(dest)
            print(f"\nTranslating from {source} to {dest}: '{test_text}'")
            translation = cache.translate(test_text)
            print(f"Result: '{translation}'")
            print_result(f"Translation from {source} to {dest} succeeded", translation is not None)

        except Exception as e:
            print(f"ERROR: {e}")
            print_result(f"Translation from {source} to {dest} should work", False)

if __name__ == "__main__":
    print("Running translator tests...")

    try:
        run_test_translation_cache_init()
        run_test_set_source_language()
        run_test_set_destination_language()
        run_test_fuzzy_matching()
        run_test_translate_with_cache_building()
        run_test_different_language_pairs()

        print("\n" + "=" * 50)
        print("All tests completed!")
    except Exception as e:
        print(f"\nTest execution failed with error: {e}")
        import traceback
        print(traceback.format_exc())
