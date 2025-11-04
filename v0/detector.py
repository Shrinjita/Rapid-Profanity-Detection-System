# detector.py
import re
import unicodedata
from better_profanity import profanity
from rapidfuzz import process, fuzz

class ProfanityDetector:
    def __init__(self):
        # Initialize the profanity checker
        profanity.load_censor_words()
        # Load custom profanity list
        self.custom_list = self._load_custom_list()
        # Create our own combined profanity list
        self.profanity_list = self.custom_list
    
    def _load_custom_list(self):
        # Create a comprehensive list of English and Hindi profanity
        # English base list
        english_words = [
            "shit", "fuck", "asshole", "bitch", "bastard", "cunt", 
            "dick", "pussy", "whore", "slut", "cock", "damn", "hell"
        ]
        
        # Hindi words from the provided list
        hindi_words = [
            "kutte", "aulad", "zat", "suar", "chutiya", "harami", 
            "randi", "bhosdi", "gaandu", "lund", "chut", "gand", "madarchod"
        ]
        
        return english_words + hindi_words
    
    def normalize(self, text):
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Lowercase
        text = text.lower()
        
        # Replace common leet substitutions
        leet_replacements = {
            '0': 'o', '1': 'i', '3': 'e', '4': 'a', 
            '5': 's', '7': 't', '@': 'a', '$': 's', '!': 'i'
        }
        for char, replacement in leet_replacements.items():
            text = text.replace(char, replacement)
        
        # Remove zero-width and control characters
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _create_obfuscation_regex(self, word):
        # Create regex that allows non-word characters between letters
        pattern = r'[\W_]*'.join([re.escape(char) for char in word])
        return re.compile(pattern, re.IGNORECASE)
    
    def detect(self, text, threshold=2.5):
        normalized_text = self.normalize(text)
        
        matches = []
        total_score = 0
        
        # Check against our custom list with multiple methods
        for word in self.profanity_list:
            # Exact match
            if word in normalized_text:
                matches.append({
                    'type': 'exact',
                    'matched': word,
                    'original': text,
                    'weight': 3
                })
                total_score += 3
                continue
                
            # Obfuscation pattern match
            regex = self._create_obfuscation_regex(word)
            if regex.search(normalized_text):
                matches.append({
                    'type': 'obfuscation',
                    'matched': word,
                    'original': text,
                    'weight': 2
                })
                total_score += 2
        
        # Check fuzzy matches for tokens
        tokens = normalized_text.split()
        for token in tokens:
            result = process.extractOne(token, self.profanity_list, scorer=fuzz.ratio)
            if result and result[1] >= 75:
                matched_word, score, _ = result
                matches.append({
                    'type': 'fuzzy',
                    'matched': matched_word,
                    'original': token,
                    'weight': 1.5,
                    'similarity': score
                })
                total_score += 1.5
        
        return {
            'score': total_score,
            'flagged': total_score >= threshold,
            'matches': matches,
            'normalized_text': normalized_text
        }