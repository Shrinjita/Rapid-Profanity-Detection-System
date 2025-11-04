# generate_dataset.py
import pandas as pd

# Create a small test dataset
data = []
id_counter = 1

# Explicit profanity examples
explicit_examples = [
    "This is shit",
    "Go to hell",
    "You're an asshole",
    "What the fuck",
    "Bloody hell",
    "kutte ki aulad",
    "chutiya",
    "harami",
    "randi"
]

for text in explicit_examples:
    data.append({
        'id': id_counter,
        'text': text,
        'label': 1,
        'notes': 'explicit profanity'
    })
    id_counter += 1

# Obfuscated examples
obfuscated_examples = [
    "This is sh1t",
    "What the f*ck",
    "s.h.i.t",
    "fuuuuck",
    "a$$hole",
    "kutt3 k1 @ul@d",
    "chut!ya"
]

for text in obfuscated_examples:
    data.append({
        'id': id_counter,
        'text': text,
        'label': 1,
        'notes': 'obfuscated profanity'
    })
    id_counter += 1

# Clean examples
clean_examples = [
    "Hello world",
    "How are you doing?",
    "This is a classroom",
    "I live in Scunthorpe",
    "The shell is hard",
    "Nice to meet you",
    "What is your name?",
    "I like to play games",
    "The weather is nice",
    "Have a good day"
]

for text in clean_examples:
    data.append({
        'id': id_counter,
        'text': text,
        'label': 0,
        'notes': 'clean text'
    })
    id_counter += 1

# Ambiguous examples
ambiguous_examples = [
    "You're a special person",
    "That's one hell of a game",
    "I'm feeling bloody tired",
    "That's badass",
    "This is the shit! (meaning excellent)"
]

for text in ambiguous_examples:
    data.append({
        'id': id_counter,
        'text': text,
        'label': 0.5,
        'notes': 'contextual/ambiguous'
    })
    id_counter += 1

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv('dataset_small.csv', index=False)
print("Dataset created with", len(df), "examples")