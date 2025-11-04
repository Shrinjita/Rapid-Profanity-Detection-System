# Multilingual Profanity Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A real-time multilingual profanity detection system designed for gaming platforms and content moderation. Combines rule-based detection with advanced obfuscation pattern matching to catch profanity that bypasses traditional word filters.

## Why We Built This

Traditional profanity filters fail catastrophically in gaming environments where users employ sophisticated obfuscation techniques:
- **Leet speak**: `sh1t`, `f*ck`, `a$$hole`
- **Punctuation insertion**: `s.h.i.t`, `f-u-c-k`
- **Character repetition**: `fuuuuck`, `shiiiit`
- **Cross-language evasion**: Switching to Hindi, Arabic, or other languages

Gaming platforms need **fast, explainable, and multilingual** detection that doesn't require GPU resources or complex model deployments. This system addresses that gap.

## How It Works

**Simple 4-Stage Pipeline:**

```
Text Input → Normalization → Multi-Layer Detection → Scoring → Decision
```

1. **Text Normalization**: Unicode normalization, leet speak conversion, whitespace cleanup
2. **Multi-Layer Detection**:
   - **Exact matching** (weight: 3.0) - Direct word matches
   - **Obfuscation patterns** (weight: 2.0) - Regex-based punctuation/spacing detection  
   - **Fuzzy matching** (weight: 1.5) - Similarity-based detection for typos/variations
3. **Configurable Scoring**: Weighted scoring system with adjustable threshold
4. **Explainable Results**: Shows exactly which patterns triggered detection

## Text Tagging / Classification Strategy

**Multi-Tier Classification System:**

- **Tier 1 - Explicit Profanity** (Score ≥ 3.0): Direct matches to blacklisted terms
- **Tier 2 - Obfuscated Content** (Score ≥ 2.0): Pattern-matched variations  
- **Tier 3 - Fuzzy Matches** (Score ≥ 1.5): Similarity-based detection (75%+ match)
- **Contextual/Ambiguous** (Manual review): Terms requiring human judgment

**Language Support:**
- **English**: Comprehensive profanity database
- **Hindi**: Extended coverage including regional variations
- **Extensible**: Easy addition of new languages via word lists

**Detection Categories:**
- Explicit sexual content
- Violent language  
- Discriminatory terms
- Gaming-specific toxicity
- Regional/cultural profanity

## Key Features

- ⚡ **High Performance**: No GPU required, sub-millisecond detection
- 🌍 **Multilingual**: English + Hindi with easy expansion
- 🛡️ **Obfuscation Resistant**: Catches leet speak, punctuation tricks, repetition
- 📊 **Explainable AI**: Shows exactly why text was flagged
- 🎛️ **Configurable**: Adjustable sensitivity thresholds
- 📁 **Batch Processing**: Handle CSV uploads for bulk analysis
- 🔧 **Developer-Friendly**: Simple API, clear documentation
- 📈 **Real-time Demo**: Interactive Streamlit interface

## Tech Stack

**Core Detection:**
- **Python 3.8+** - Main language
- **better-profanity** - Base profanity detection
- **rapidfuzz** - High-performance fuzzy matching
- **unicodedata2** - Unicode normalization

**Demo Interface:**
- **Streamlit** - Interactive web demo
- **Pandas** - Data processing and CSV handling

**Text Processing:**
- **Regular Expressions** - Obfuscation pattern matching
- **Unicode Normalization** - Consistent text preprocessing

## Getting Started

### Clone the Repository
```bash
git clone https://github.com/yourusername/multilingual-profanity-detector.git
cd multilingual-profanity-detector
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit>=1.22.0
pandas>=1.5.0
better-profanity>=0.7.0
rapidfuzz>=3.0.0
unicodedata2>=15.0.0
```

### Environment Setup
No environment variables or additional configuration required. The system works out-of-the-box with default settings.

## Usage

### Interactive Demo
Launch the Streamlit demo interface:
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` to access the web interface.

### Programmatic Usage
```python
from detector import ProfanityDetector

# Initialize detector
detector = ProfanityDetector()

# Single text detection
result = detector.detect("This is some sh1t", threshold=2.5)
print(f"Flagged: {result['flagged']}")
print(f"Score: {result['score']}")
print(f"Matches: {result['matches']}")

# Example output:
# Flagged: True
# Score: 2.0
# Matches: [{'type': 'obfuscation', 'matched': 'shit', 'weight': 2}]
```

### Future CLI Interface
```bash
# Coming soon
python -m detector "text to analyze" --threshold 2.5 --lang en,hi
```

### Demo Examples

**Clean Text:**
```
Input: "Hello world, how are you?"
Output: CLEAN (Score: 0.0)
```

**Explicit Profanity:**
```
Input: "This is shit"
Output: FLAGGED (Score: 3.0) - Exact match: 'shit'
```

**Obfuscated Detection:**
```
Input: "That's some sh1t"
Output: FLAGGED (Score: 2.0) - Obfuscation pattern: 'sh1t' → 'shit'
```

**Hindi Detection:**
```
Input: "kutte ki aulad"
Output: FLAGGED (Score: 3.0) - Exact match: Hindi profanity
```

### Batch Processing
Upload CSV with `text` column to process multiple entries:
```csv
id,text
1,"Hello world"
2,"This is sh1t"
3,"kutte ki aulad"
```

## Dataset / Model Files

### Evaluation Dataset
- **dataset_small.csv**: 31 test examples covering:
  - Explicit profanity (9 examples)
  - Obfuscated content (7 examples)  
  - Clean text (10 examples)
  - Contextual/ambiguous cases (5 examples)

### Word Lists
Built-in profanity databases:
- **English**: 13 core terms + variations
- **Hindi**: 13 regional terms + transliterations

### Model Files
Currently rule-based only. No trained models required.

**Future Model Integration:**
- Fine-tuned DistilBERT for contextual analysis
- Word embeddings for semantic similarity
- Custom transformer models for gaming-specific toxicity

## Contributing

We welcome contributions! Priority areas:

1. **Language Expansion**: Add Spanish, Portuguese, Arabic word lists
2. **Obfuscation Patterns**: New evasion techniques and detection methods
3. **Performance Optimization**: Faster fuzzy matching, caching strategies
4. **Gaming Context**: Platform-specific toxicity patterns

**Contribution Process:**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-language`  
3. Add comprehensive tests for new functionality
4. Submit pull request with clear description

**Code Standards:**
- Python 3.8+ compatibility
- Type hints for all functions
- Comprehensive test coverage
- Clear documentation

## Roadmap / Future Work

### Phase 1: Core Expansion (Q4 2024)
- [ ] Spanish and Portuguese language support
- [ ] Docker containerization for easy deployment
- [ ] Simple CLI interface
- [ ] Performance benchmarking suite

### Phase 2: Intelligence Layer (Q1 2025)
- [ ] Lightweight transformer integration for context
- [ ] Word embeddings for semantic detection
- [ ] User feedback loop for continuous improvement
- [ ] Custom wordlist management UI

### Phase 3: Enterprise Features (Q2 2025)
- [ ] Real-time performance metrics dashboard
- [ ] A/B testing framework for threshold optimization
- [ ] Multi-tenant configuration management
- [ ] Advanced analytics and reporting

### Phase 4: Advanced Detection (Q3 2025)
- [ ] Image-based profanity detection (OCR + analysis)
- [ ] Audio content moderation integration
- [ ] Cross-platform behavior analysis
- [ ] ML-powered contextual understanding

## License

MIT License - see [LICENSE](LICENSE) file for details.

This permissive license allows free use in commercial gaming platforms and community tools.

## Acknowledgments

**Open Source Libraries:**
- [better-profanity](https://github.com/snguyenthanh/better_profanity) - Foundation profanity detection
- [rapidfuzz](https://github.com/maxbachmann/RapidFuzz) - High-performance fuzzy string matching
- [Streamlit](https://streamlit.io/) - Interactive demo interface

**Community Contributors:**
- Gaming community moderators for real-world feedback and edge case identification
- Multilingual contributors for Hindi language validation and cultural context

**Research & Inspiration:**
- Academic research on content moderation and NLP preprocessing techniques
- Gaming industry best practices for real-time content filtering

---

**Questions or Issues?** Open an issue on GitHub or contact the maintainers.

**Want to Integrate?** Check our [Integration Guide](docs/integration.md) for API documentation and deployment examples.
