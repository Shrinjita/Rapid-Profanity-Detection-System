from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import re
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'dev-secret-key-123')
socketio = SocketIO(app, cors_allowed_origins="*")

HINDI_PROFANITY_LATIN = [
    "mc", "bc", "bhosdike", "chutiya", "gandu", "lund", 
    "kutta", "kamine", "behenchod", "madarchod", "sala", 
    "randi", "harami", "lauda", "chut", "gaand", "chod"
]

HINDI_PROFANITY_DEVANAGARI = [
    "मादरचोद", "बहनचोद", "रंडी", "रण्डी", "लौड़ा", "लंड", 
    "चूत", "गांड", "गांडू", "चोद", "चोदू", "चुतिया", 
    "कुत्ता", "कमीने", "साला", "हरामी", "भोसडीके"
]

users = {}
game_state = {
    'players': {},
    'enemies': []
}

def normalize_latin_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', '', text)
    return text

def normalize_devanagari_text(text):
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    text = re.sub(r'\s+', '', text)
    return text

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def is_similar(word, profanity, threshold=2):
    return levenshtein_distance(word, profanity) <= threshold

def detect_profanity(text):
    print(f"[DEBUG] Checking text: {text}")
    
    normalized_latin = normalize_latin_text(text)
    normalized_devanagari = normalize_devanagari_text(text)
    
    for profanity in HINDI_PROFANITY_LATIN:
        if profanity in normalized_latin:
            print(f"[DEBUG] Detected Latin profanity: {profanity}")
            return True
        
        words = text.lower().split()
        for word in words:
            word_normalized = normalize_latin_text(word)
            if is_similar(word_normalized, profanity):
                print(f"[DEBUG] Detected similar Latin profanity: {profanity} (from word: {word})")
                return True
    
    for profanity in HINDI_PROFANITY_DEVANAGARI:
        if profanity in text:
            print(f"[DEBUG] Detected Devanagari profanity: {profanity}")
            return True
        
        if profanity in normalized_devanagari:
            print(f"[DEBUG] Detected normalized Devanagari profanity: {profanity}")
            return True
    
    print(f"[DEBUG] No profanity detected")
    return False

@app.route('/')
def index():
    return render_template('game.html')

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('join_game')
def handle_join(data):
    username = data.get('username', 'Anonymous')
    session_id = request.sid
    
    users[session_id] = {
        'username': username,
        'warnings': 0,
        'blocked': False,
        'score': 0,
        'position': {'x': 400, 'y': 500}
    }
    
    game_state['players'][session_id] = users[session_id]
    
    emit('game_joined', {
        'session_id': session_id,
        'username': username,
        'players': {sid: {'username': u['username'], 'score': u['score'], 'warnings': u['warnings']} 
                   for sid, u in users.items() if not u['blocked']}
    })
    
    emit('player_joined', {
        'session_id': session_id,
        'username': username,
        'score': 0,
        'warnings': 0
    }, broadcast=True, include_self=False)

@socketio.on('voice_message')
def handle_voice_message(data):
    session_id = request.sid
    text = data.get('text', '')
    username = users.get(session_id, {}).get('username', 'Anonymous')
    
    if session_id not in users or users[session_id]['blocked']:
        return
    
    contains_profanity = detect_profanity(text)
    
    if contains_profanity:
        users[session_id]['warnings'] += 1
        warnings = users[session_id]['warnings']
        
        warning_data = {
            'username': username,
            'warnings': warnings,
            'text': text
        }
        
        if warnings >= 3:
            users[session_id]['blocked'] = True
            emit('user_blocked', {
                'username': username,
                'session_id': session_id
            }, broadcast=True)
        else:
            emit('profanity_warning', warning_data, broadcast=True)
    
    emit('chat_message', {
        'username': username,
        'text': text,
        'flagged': contains_profanity,
        'session_id': session_id
    }, broadcast=True)

@socketio.on('text_message')
def handle_text_message(data):
    session_id = request.sid
    text = data.get('text', '')
    username = users.get(session_id, {}).get('username', 'Anonymous')
    
    if session_id not in users or users[session_id]['blocked']:
        return
    
    contains_profanity = detect_profanity(text)
    
    if contains_profanity:
        users[session_id]['warnings'] += 1
        warnings = users[session_id]['warnings']
        
        warning_data = {
            'username': username,
            'warnings': warnings,
            'text': text
        }
        
        if warnings >= 3:
            users[session_id]['blocked'] = True
            emit('user_blocked', {
                'username': username,
                'session_id': session_id
            }, broadcast=True)
        else:
            emit('profanity_warning', warning_data, broadcast=True)
    
    emit('chat_message', {
        'username': username,
        'text': text,
        'flagged': contains_profanity,
        'session_id': session_id
    }, broadcast=True)

@socketio.on('player_move')
def handle_player_move(data):
    session_id = request.sid
    if session_id in users and not users[session_id]['blocked']:
        users[session_id]['position'] = data.get('position', users[session_id]['position'])
        emit('player_moved', {
            'session_id': session_id,
            'position': users[session_id]['position']
        }, broadcast=True, include_self=False)

@socketio.on('player_shoot')
def handle_player_shoot(data):
    session_id = request.sid
    if session_id in users and not users[session_id]['blocked']:
        emit('player_shot', {
            'session_id': session_id,
            'bullet': data.get('bullet')
        }, broadcast=True)

@socketio.on('enemy_hit')
def handle_enemy_hit(data):
    session_id = request.sid
    if session_id in users and not users[session_id]['blocked']:
        users[session_id]['score'] += 10
        emit('score_update', {
            'session_id': session_id,
            'username': users[session_id]['username'],
            'score': users[session_id]['score']
        }, broadcast=True)

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    if session_id in users:
        username = users[session_id]['username']
        del users[session_id]
        if session_id in game_state['players']:
            del game_state['players'][session_id]
        emit('player_left', {
            'session_id': session_id,
            'username': username
        }, broadcast=True)
    print(f"Client disconnected: {session_id}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
