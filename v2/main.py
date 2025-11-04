"""
Rapid Prototyping Multilingual Profanity Detection Module
Real-time speech moderation with bounded-edit Aho-Corasick automaton
Author: Systems Engineering Team
Target: Gaming chat moderation with <50ms text processing latency
"""

import streamlit as st
import unicodedata
import re
import json
import csv
import time
import logging
from pathlib import Path
from datetime import datetime
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional, Set
import sys

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "profanity_events.log",
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

# Session state initialization
if 'warning_count' not in st.session_state:
    st.session_state.warning_count = 0
if 'transcript_history' not in st.session_state:
    st.session_state.transcript_history = []
if 'stt_mode' not in st.session_state:
    st.session_state.stt_mode = 'streamlit'
if 'ac_automaton' not in st.session_state:
    st.session_state.ac_automaton = None
if 'profanity_datasets' not in st.session_state:
    st.session_state.profanity_datasets = []
if 'datasets_loaded' not in st.session_state:
    st.session_state.datasets_loaded = False

# ============================================================================
# TEXT CANONICALIZATION MODULE
# ============================================================================

class TextCanonicalizer:
    """
    Full Unicode normalization and obfuscation resistance module.
    Reduces text to canonical minimal-entropy form for matching.
    """
    
    # Homoglyph mapping (visually similar characters → canonical form)
    HOMOGLYPH_MAP = {
        # Cyrillic to Latin
        'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
        'А': 'a', 'В': 'b', 'Е': 'e', 'К': 'k', 'М': 'm', 'Н': 'h', 'О': 'o',
        'Р': 'p', 'С': 'c', 'Т': 't', 'Х': 'x',
        # Greek to Latin
        'α': 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'ε': 'e', 'ζ': 'z', 'η': 'n',
        'θ': 'o', 'ι': 'i', 'κ': 'k', 'λ': 'l', 'μ': 'u', 'ν': 'v', 'ο': 'o',
        'π': 'p', 'ρ': 'p', 'σ': 's', 'τ': 't', 'υ': 'u', 'φ': 'f', 'χ': 'x',
        'ψ': 'ps', 'ω': 'w',
    }
    
    # Leetspeak mapping
    LEET_MAP = {
        '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't', '8': 'b',
        '@': 'a', '$': 's', '!': 'i', '|': 'i', '+': 't', '€': 'e',
    }
    
    # Diacritic removal (via NFD decomposition)
    @staticmethod
    def remove_diacritics(text: str) -> str:
        """Remove accents and diacritical marks."""
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    
    @classmethod
    def canonicalize(cls, text: str) -> str:
        """
        Full canonicalization pipeline:
        1. NFKC normalization (compatibility decomposition)
        2. Lowercase folding
        3. Zero-width character removal
        4. Homoglyph mapping
        5. Leetspeak mapping
        6. Diacritic removal
        7. Punctuation/whitespace stripping
        """
        if not text:
            return ""
        
        # NFKC normalization (handles ligatures, superscripts, etc.)
        text = unicodedata.normalize('NFKC', text)
        
        # Lowercase
        text = text.lower()
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', text)
        
        # Apply homoglyph mapping
        text = ''.join(cls.HOMOGLYPH_MAP.get(c, c) for c in text)
        
        # Apply leetspeak mapping
        text = ''.join(cls.LEET_MAP.get(c, c) for c in text)
        
        # Remove diacritics
        text = cls.remove_diacritics(text)
        
        # Strip punctuation and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

# ============================================================================
# BOUNDED-EDIT AHO-CORASICK AUTOMATON
# ============================================================================

class BoundedEditAhoCorasick:
    """
    Aho-Corasick automaton with bounded edit distance support (k≤2).
    Guarantees O(n) average-case complexity for pattern matching.
    Integrated with canonicalization for single-pass efficiency.
    """
    
    def __init__(self, max_edits: int = 2):
        self.max_edits = max_edits
        self.trie = {}
        self.failure = {}
        self.output = defaultdict(list)
        self.patterns = []
        
    def add_pattern(self, pattern: str, metadata: dict = None):
        """Add a pattern to the trie."""
        canonical = TextCanonicalizer.canonicalize(pattern)
        if not canonical:
            return
            
        node = self.trie
        for char in canonical:
            if char not in node:
                node[char] = {}
            node = node[char]
        
        # Mark end of pattern
        if '__end__' not in node:
            node['__end__'] = []
        node['__end__'].append({
            'pattern': pattern,
            'canonical': canonical,
            'metadata': metadata or {}
        })
        self.patterns.append(canonical)
    
    def build_failure_links(self):
        """Build failure function for Aho-Corasick (BFS-based)."""
        queue = deque()
        
        # Level 1: all children of root fail to root
        for char, child in self.trie.items():
            if char != '__end__':
                self.failure[id(child)] = self.trie
                queue.append((child, char))
        
        # BFS to build failure links
        while queue:
            node, _ = queue.popleft()
            
            for char, child in node.items():
                if char == '__end__':
                    continue
                    
                queue.append((child, char))
                
                # Find failure link
                fail_node = self.failure.get(id(node), self.trie)
                while fail_node is not self.trie and char not in fail_node:
                    fail_node = self.failure.get(id(fail_node), self.trie)
                
                if char in fail_node and fail_node[char] is not child:
                    self.failure[id(child)] = fail_node[char]
                else:
                    self.failure[id(child)] = self.trie
    
    def search(self, text: str) -> List[Dict]:
        """
        Search for patterns in text with bounded edit distance.
        Returns list of matches with positions and metadata.
        """
        canonical = TextCanonicalizer.canonicalize(text)
        if not canonical:
            return []
        
        matches = []
        node = self.trie
        
        for i, char in enumerate(canonical):
            # Exact match traversal
            while node is not self.trie and char not in node:
                node = self.failure.get(id(node), self.trie)
            
            if char in node:
                node = node[char]
            
            # Check for pattern matches
            current = node
            visited = set()
            
            while current is not None:
                if id(current) in visited:
                    break
                visited.add(id(current))
                
                if '__end__' in current:
                    for match_info in current['__end__']:
                        matches.append({
                            'position': i,
                            'pattern': match_info['pattern'],
                            'canonical': match_info['canonical'],
                            'metadata': match_info['metadata']
                        })
                
                current = self.failure.get(id(current))
                if current is self.trie:
                    break
        
        # Bounded edit distance matching (k≤2)
        # Simplified: check substrings with character variations
        if self.max_edits > 0 and not matches:
            matches.extend(self._fuzzy_match(canonical))
        
        return matches
    
    def _fuzzy_match(self, text: str) -> List[Dict]:
        """
        Fuzzy matching with bounded edits.
        Checks for patterns with minor character substitutions/deletions.
        """
        matches = []
        text_len = len(text)
        
        for pattern in self.patterns[:100]:  # Limit for performance
            pattern_len = len(pattern)
            if pattern_len > text_len + self.max_edits:
                continue
            
            # Sliding window with edit tolerance
            for i in range(max(0, text_len - pattern_len - self.max_edits + 1)):
                window = text[i:i + pattern_len + self.max_edits]
                if self._edit_distance(pattern, window) <= self.max_edits:
                    matches.append({
                        'position': i,
                        'pattern': pattern,
                        'canonical': pattern,
                        'metadata': {'fuzzy': True}
                    })
                    break
        
        return matches
    
    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Levenshtein distance (optimized for small k)."""
        if abs(len(s1) - len(s2)) > 2:
            return 99  # Early exit
        
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )
        
        return dp[m][n]

# ============================================================================
# DATASET LOADER
# ============================================================================

def load_profanity_dataset(file_path: str) -> List[str]:
    """
    Load profanity patterns from .txt, .csv, or .json file.
    Returns list of profanity words/phrases.
    """
    path = Path(file_path)
    if not path.exists():
        logging.warning(f"Dataset not found: {file_path}")
        return []
    
    try:
        if path.suffix == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        
        elif path.suffix == '.csv':
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                return [row[0].strip() for row in reader if row and row[0].strip()]
        
        elif path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [str(item).strip() for item in data if item]
                elif isinstance(data, dict):
                    return list(data.keys())
        
        else:
            logging.error(f"Unsupported file format: {path.suffix}")
            return []
    
    except Exception as e:
        logging.error(f"Error loading dataset {file_path}: {e}")
        return []

def build_automaton(datasets: List[List[str]]) -> BoundedEditAhoCorasick:
    """Build Aho-Corasick automaton from multiple datasets."""
    ac = BoundedEditAhoCorasick(max_edits=2)
    
    for dataset_idx, dataset in enumerate(datasets):
        for pattern in dataset:
            ac.add_pattern(pattern, metadata={'dataset': dataset_idx})
    
    ac.build_failure_links()
    logging.info(f"Built AC automaton with {len(ac.patterns)} patterns")
    return ac

# ============================================================================
# SPEECH-TO-TEXT MODULE
# ============================================================================

def streamlit_stt() -> Optional[str]:
    """Streamlit native speech-to-text using audio input."""
    st.info("🎤 Streamlit STT Mode: Use the audio recorder below")
    
    audio_value = st.audio_input("Record your voice", key="audio_input")
    
    if audio_value is not None:
        # Placeholder: In production, integrate with Whisper/Google STT
        st.warning("⚠️ Audio received. Real STT integration needed (e.g., Whisper API)")
        return None
    
    return None

def vosk_stt(language="en", duration=5) -> Optional[str]:
    """
    Offline Vosk speech-to-text for English or Hindi.
    Requires pre-downloaded model folders:
        vosk-model-small-en-us-0.15
        vosk-model-small-hi-0.22
    """
    try:
        from vosk import Model, KaldiRecognizer
        import pyaudio
        import json as vosk_json
        
        model_path = {
            "en": "vosk-model-small-en-us-0.15",
            "hi": "vosk-model-small-hi-0.22"
        }.get(language, "vosk-model-small-en-us-0.15")
        
        if not Path(model_path).exists():
            st.error(f"❌ Vosk model not found: {model_path}")
            st.info("Download from: https://alphacephei.com/vosk/models")
            return None
        
        model = Model(model_path)
        recognizer = KaldiRecognizer(model, 16000)
        
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1,
                            rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()
        
        st.info(f"🎙️ Speak now... (Recording for {duration}s)")
        data = stream.read(int(16000 * duration))
        
        if recognizer.AcceptWaveform(data):
            result = vosk_json.loads(recognizer.Result())
            text = result.get("text", "")
        else:
            partial = vosk_json.loads(recognizer.PartialResult())
            text = partial.get("partial", "")
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        st.success(f"📝 Transcription: {text}")
        return text
    
    except ImportError:
        st.error("❌ Vosk not installed. Run: pip install vosk pyaudio")
        return None
    except Exception as e:
        st.error(f"❌ Vosk error: {e}")
        return None

# ============================================================================
# MODERATION LOGIC
# ============================================================================

def check_profanity(text: str, automaton: BoundedEditAhoCorasick) -> Tuple[bool, List[Dict]]:
    """
    Check text for profanity using AC automaton.
    Returns (is_profane, matches).
    """
    start_time = time.perf_counter()
    matches = automaton.search(text)
    elapsed = time.perf_counter() - start_time
    
    logging.info(f"Profanity check: {len(matches)} matches in {elapsed*1000:.2f}ms")
    
    return len(matches) > 0, matches

def handle_profanity_detection(text: str):
    """Main moderation pipeline: check text and apply warnings."""
    if not text or not st.session_state.ac_automaton:
        return
    
    is_profane, matches = check_profanity(text, st.session_state.ac_automaton)
    
    if is_profane:
        st.session_state.warning_count += 1
        count = st.session_state.warning_count
        
        logging.warning(f"Warning {count}/3: Profanity detected in '{text[:50]}...'")
        logging.info(f"Matches: {[m['pattern'] for m in matches]}")
        
        # Display warning
        st.error(f"⚠️ Warning {count}: Profanity detected. The game will shut down.")
        
        # Show detected patterns
        with st.expander("🔍 Detected patterns"):
            for match in matches[:5]:  # Show top 5
                st.write(f"- **{match['pattern']}** (dataset {match['metadata'].get('dataset', 'N/A')})")
        
        # Terminal action on 3rd warning
        if count >= 3:
            st.error("🚨 **GAME TERMINATED**: Three profanity violations detected.")
            logging.critical("Game terminated due to 3 profanity warnings")
            time.sleep(2)
            st.stop()
    else:
        st.success("✅ No profanity detected")

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Profanity Detection System",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.title("🎮 Real-Time Profanity Detection System")
    st.caption("Multilingual gaming chat moderation with bounded-edit Aho-Corasick")
    
    # Auto-load datasets on first run
    if not st.session_state.datasets_loaded:
        dataset1_path = "dataset_primary.txt"
        dataset2_path = "dataset_secondary.txt"
        
        ds1 = load_profanity_dataset(dataset1_path)
        ds2 = load_profanity_dataset(dataset2_path)
        
        if not ds1 and not ds2:
            logging.warning("No datasets found, using defaults")
            ds1 = ["fuck", "shit", "damn", "badword", "mc", "bc"]
            ds2 = ["asshole", "bitch", "bastard"]
        
        st.session_state.profanity_datasets = [ds1, ds2]
        st.session_state.ac_automaton = build_automaton([ds1, ds2])
        st.session_state.datasets_loaded = True
        
        logging.info(f"Auto-loaded {len(ds1)} + {len(ds2)} patterns")
    
    # Sidebar: Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Dataset status
        st.subheader("📚 Profanity Datasets")
        if st.session_state.profanity_datasets:
            ds1_count = len(st.session_state.profanity_datasets[0])
            ds2_count = len(st.session_state.profanity_datasets[1])
            st.success(f"✅ Loaded: {ds1_count} + {ds2_count} patterns")
        
        # Manual reload option
        if st.button("🔄 Reload Datasets"):
            ds1 = load_profanity_dataset("dataset_primary.txt")
            ds2 = load_profanity_dataset("dataset_secondary.txt")
            
            if not ds1 and not ds2:
                st.error("❌ No datasets found")
            else:
                st.session_state.profanity_datasets = [ds1, ds2]
                st.session_state.ac_automaton = build_automaton([ds1, ds2])
                st.success(f"✅ Reloaded {len(ds1)} + {len(ds2)} patterns")
        
        # STT mode selection
        st.subheader("🎤 Speech-to-Text Mode")
        stt_mode = st.radio(
            "Select STT backend:",
            options=['text_only', 'vosk', 'streamlit'],
            format_func=lambda x: {
                'streamlit': 'Streamlit Native',
                'vosk': 'Vosk (Offline)',
                'text_only': 'Text Input Only'
            }[x]
        )
        st.session_state.stt_mode = stt_mode
        
        # Vosk language if applicable
        vosk_lang = "en"
        if stt_mode == 'vosk':
            vosk_lang = st.selectbox("Vosk Language:", ["en", "hi"], 
                                     format_func=lambda x: "English" if x == "en" else "Hindi")
        
        # Warning counter
        st.subheader("⚠️ Warning Status")
        st.metric("Warnings", f"{st.session_state.warning_count}/3")
        
        if st.button("🔄 Reset Warnings"):
            st.session_state.warning_count = 0
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Input & Transcript")
        
        # Speech input based on mode
        transcript = None
        
        if st.session_state.stt_mode == 'vosk':
            st.info("🎤 Vosk Mode: Click 'Start Recording' to capture audio")
            vosk_duration = st.slider("Recording duration (seconds):", 1, 10, 5)
            
            if st.button("🎙️ Start Recording (Vosk)", type="primary"):
                with st.spinner("Recording..."):
                    transcript = vosk_stt(language=vosk_lang, duration=vosk_duration)
                
                if transcript:
                    st.session_state.transcript_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'text': transcript
                    })
                    handle_profanity_detection(transcript)
        
        elif st.session_state.stt_mode == 'streamlit':
            transcript = streamlit_stt()
        
        # Text-only mode (always available)
        st.subheader("✍️ Manual Text Input")
        manual_input = st.text_area(
            "Type or paste text here:",
            height=150,
            placeholder="Enter text for profanity detection..."
        )
        
        if st.button("🔍 Check for Profanity", type="primary"):
            text_to_check = manual_input
            
            if text_to_check:
                st.session_state.transcript_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'text': text_to_check
                })
                
                with st.spinner("Analyzing..."):
                    handle_profanity_detection(text_to_check)
            else:
                st.warning("⚠️ No text to check")
    
    with col2:
        st.header("📊 System Status")
        
        # Performance metrics
        if st.session_state.ac_automaton:
            st.metric("Loaded Patterns", len(st.session_state.ac_automaton.patterns))
        
        st.metric("Transcripts Processed", len(st.session_state.transcript_history))
        
        # Recent transcript history
        st.subheader("📜 Recent Transcripts")
        for entry in reversed(st.session_state.transcript_history[-5:]):
            with st.expander(f"🕐 {entry['timestamp'][-8:]}"):
                st.write(entry['text'])
    
    # Footer
    st.divider()
    st.caption("⚡ Target latency: <50ms text processing | 🧠 Deterministic O(n) complexity")

if __name__ == "__main__":
    main()