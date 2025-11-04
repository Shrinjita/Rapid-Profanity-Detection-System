const socket = io();
let username = '';
let sessionId = '';
let myScore = 0;
let myWarnings = 0;
let isBlocked = false;
let gameStarted = false;

let recognition = null;
let isRecording = false;

const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

let player = {
    x: 400,
    y: 500,
    width: 40,
    height: 40,
    speed: 5,
    color: '#6366f1'
};

let otherPlayers = {};
let bullets = [];
let enemies = [];
let keys = {};

let enemySpawnTimer = 0;
const ENEMY_SPAWN_INTERVAL = 120;

document.getElementById('join-btn').addEventListener('click', joinGame);
document.getElementById('username-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') joinGame();
});

document.getElementById('voice-btn').addEventListener('click', toggleVoice);
document.getElementById('send-text').addEventListener('click', sendTextMessage);
document.getElementById('text-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendTextMessage();
});

document.getElementById('start-game').addEventListener('click', () => {
    if (!gameStarted) {
        gameStarted = true;
        document.getElementById('start-game').textContent = 'Game Running';
        document.getElementById('start-game').disabled = true;
        gameLoop();
    }
});

document.getElementById('close-alert').addEventListener('click', () => {
    document.getElementById('warning-alert').classList.remove('show');
});

document.addEventListener('keydown', (e) => {
    keys[e.key] = true;
    if (e.key === ' ' && gameStarted && !isBlocked) {
        e.preventDefault();
        shootBullet();
    }
});

document.addEventListener('keyup', (e) => {
    keys[e.key] = false;
});

function joinGame() {
    const usernameInput = document.getElementById('username-input');
    username = usernameInput.value.trim() || 'Player_' + Math.floor(Math.random() * 1000);
    
    socket.emit('join_game', { username });
    document.getElementById('join-modal').classList.add('hidden');
    document.getElementById('player-name').innerHTML = `Player: <strong>${username}</strong>`;
}

function toggleVoice() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        addChatMessage('System', 'Voice recognition not supported in this browser', true);
        return;
    }

    if (isRecording) {
        stopVoiceRecognition();
    } else {
        startVoiceRecognition();
    }
}

function startVoiceRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.lang = 'hi-IN';
    recognition.continuous = true;
    recognition.interimResults = false;

    recognition.onstart = () => {
        isRecording = true;
        document.getElementById('voice-btn').classList.add('recording');
        document.getElementById('voice-status').textContent = 'Recording...';
        document.getElementById('voice-indicator').classList.add('active');
    };

    recognition.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript;
        socket.emit('voice_message', { text: transcript });
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        stopVoiceRecognition();
    };

    recognition.onend = () => {
        if (isRecording) {
            recognition.start();
        }
    };

    recognition.start();
}

function stopVoiceRecognition() {
    if (recognition) {
        isRecording = false;
        recognition.stop();
        document.getElementById('voice-btn').classList.remove('recording');
        document.getElementById('voice-status').textContent = 'Start Voice Chat';
        document.getElementById('voice-indicator').classList.remove('active');
    }
}

function sendTextMessage() {
    const input = document.getElementById('text-input');
    const text = input.value.trim();
    
    if (text && !isBlocked) {
        socket.emit('text_message', { text });
        input.value = '';
    }
}

function addChatMessage(user, text, flagged = false) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message' + (flagged ? ' flagged' : '');
    
    messageDiv.innerHTML = `
        <span class="chat-username">${user}:</span>
        <span class="chat-text">${text}</span>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function updatePlayerList() {
    const playerList = document.getElementById('player-list');
    playerList.innerHTML = '';
    
    const players = [{ username, score: myScore, warnings: myWarnings, isCurrent: true }];
    
    Object.values(otherPlayers).forEach(p => {
        players.push({ ...p, isCurrent: false });
    });
    
    players.forEach(p => {
        const playerDiv = document.createElement('div');
        playerDiv.className = 'player-item' + (p.isCurrent ? ' current-player' : '');
        playerDiv.innerHTML = `
            <div class="player-info">
                <span class="player-name">${p.username}</span>
                <span class="player-score">Score: ${p.score} | Warnings: ${p.warnings || 0}/3</span>
            </div>
        `;
        playerList.appendChild(playerDiv);
    });
}

function updateWarningDisplay(warnings) {
    const warningValue = document.querySelector('.warning-value');
    warningValue.textContent = `${warnings}/3`;
    
    if (warnings === 1) {
        warningValue.style.color = '#eab308';
    } else if (warnings === 2) {
        warningValue.style.color = '#f97316';
    } else if (warnings >= 3) {
        warningValue.style.color = '#ef4444';
    }
}

function showWarningAlert(warnings, message) {
    const alertModal = document.getElementById('warning-alert');
    const alertContent = alertModal.querySelector('.alert-content');
    const alertTitle = document.getElementById('alert-title');
    const alertMessage = document.getElementById('alert-message');
    
    alertContent.className = 'alert-content warning-level-' + warnings;
    
    if (warnings === 1) {
        alertTitle.textContent = '⚠️ Warning 1: Yellow Alert';
        alertMessage.textContent = `Profanity detected: "${message}". First warning issued. Please maintain respectful communication.`;
    } else if (warnings === 2) {
        alertTitle.textContent = '⚠️ Warning 2: Orange Alert';
        alertMessage.textContent = `Profanity detected: "${message}". Second warning! One more violation will result in blocking.`;
    } else if (warnings >= 3) {
        alertTitle.textContent = '🚫 Warning 3: Red Alert';
        alertMessage.textContent = `Profanity detected: "${message}". You have been blocked from the game due to repeated violations.`;
    }
    
    alertModal.classList.add('show');
}

socket.on('game_joined', (data) => {
    sessionId = data.session_id;
    username = data.username;
    
    Object.entries(data.players).forEach(([sid, player]) => {
        if (sid !== sessionId) {
            otherPlayers[sid] = player;
        }
    });
    
    updatePlayerList();
});

socket.on('player_joined', (data) => {
    if (data.session_id !== sessionId) {
        otherPlayers[data.session_id] = {
            username: data.username,
            score: data.score,
            warnings: data.warnings
        };
        updatePlayerList();
        addChatMessage('System', `${data.username} joined the game`, false);
    }
});

socket.on('player_left', (data) => {
    if (data.session_id !== sessionId) {
        delete otherPlayers[data.session_id];
        updatePlayerList();
        addChatMessage('System', `${data.username} left the game`, false);
    }
});

socket.on('chat_message', (data) => {
    addChatMessage(data.username, data.text, data.flagged);
});

socket.on('profanity_warning', (data) => {
    if (data.username === username) {
        myWarnings = data.warnings;
        updateWarningDisplay(myWarnings);
        showWarningAlert(data.warnings, data.text);
    }
    addChatMessage('System', `⚠️ ${data.username} received warning ${data.warnings}/3`, true);
});

socket.on('user_blocked', (data) => {
    if (data.username === username) {
        isBlocked = true;
        showWarningAlert(3, 'Multiple violations');
        stopVoiceRecognition();
        addChatMessage('System', '🚫 You have been blocked from the game', true);
    } else {
        addChatMessage('System', `🚫 ${data.username} has been blocked for repeated violations`, true);
    }
});

socket.on('score_update', (data) => {
    if (data.session_id === sessionId) {
        myScore = data.score;
        document.getElementById('player-score').innerHTML = `Score: <strong>${myScore}</strong>`;
    } else if (otherPlayers[data.session_id]) {
        otherPlayers[data.session_id].score = data.score;
    }
    updatePlayerList();
});

function updatePlayer() {
    if (isBlocked) return;
    
    if (keys['ArrowLeft'] && player.x > 0) {
        player.x -= player.speed;
    }
    if (keys['ArrowRight'] && player.x < canvas.width - player.width) {
        player.x += player.speed;
    }
    if (keys['ArrowUp'] && player.y > 0) {
        player.y -= player.speed;
    }
    if (keys['ArrowDown'] && player.y < canvas.height - player.height) {
        player.y += player.speed;
    }
    
    socket.emit('player_move', { position: { x: player.x, y: player.y } });
}

function shootBullet() {
    bullets.push({
        x: player.x + player.width / 2 - 2,
        y: player.y,
        width: 4,
        height: 15,
        speed: 7,
        color: '#6366f1'
    });
    
    socket.emit('player_shoot', { bullet: { x: player.x, y: player.y } });
}

function updateBullets() {
    bullets = bullets.filter(bullet => {
        bullet.y -= bullet.speed;
        return bullet.y > 0;
    });
}

function spawnEnemy() {
    enemies.push({
        x: Math.random() * (canvas.width - 30),
        y: -30,
        width: 30,
        height: 30,
        speed: 2 + Math.random() * 2,
        color: '#ef4444'
    });
}

function updateEnemies() {
    enemySpawnTimer++;
    if (enemySpawnTimer >= ENEMY_SPAWN_INTERVAL) {
        spawnEnemy();
        enemySpawnTimer = 0;
    }
    
    enemies = enemies.filter(enemy => {
        enemy.y += enemy.speed;
        return enemy.y < canvas.height;
    });
    
    bullets.forEach((bullet, bIndex) => {
        enemies.forEach((enemy, eIndex) => {
            if (bullet.x < enemy.x + enemy.width &&
                bullet.x + bullet.width > enemy.x &&
                bullet.y < enemy.y + enemy.height &&
                bullet.y + bullet.height > enemy.y) {
                
                bullets.splice(bIndex, 1);
                enemies.splice(eIndex, 1);
                
                socket.emit('enemy_hit', {});
            }
        });
    });
}

function drawPlayer() {
    ctx.fillStyle = player.color;
    ctx.beginPath();
    ctx.moveTo(player.x + player.width / 2, player.y);
    ctx.lineTo(player.x, player.y + player.height);
    ctx.lineTo(player.x + player.width, player.y + player.height);
    ctx.closePath();
    ctx.fill();
    
    ctx.fillStyle = '#ffffff';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(username, player.x + player.width / 2, player.y + player.height + 15);
}

function drawBullets() {
    bullets.forEach(bullet => {
        ctx.fillStyle = bullet.color;
        ctx.fillRect(bullet.x, bullet.y, bullet.width, bullet.height);
    });
}

function drawEnemies() {
    enemies.forEach(enemy => {
        ctx.fillStyle = enemy.color;
        ctx.fillRect(enemy.x, enemy.y, enemy.width, enemy.height);
        
        ctx.strokeStyle = '#ffffff';
        ctx.strokeRect(enemy.x, enemy.y, enemy.width, enemy.height);
    });
}

function drawBackground() {
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    for (let i = 0; i < 50; i++) {
        const x = Math.random() * canvas.width;
        const y = Math.random() * canvas.height;
        ctx.fillRect(x, y, 1, 1);
    }
}

function gameLoop() {
    if (!gameStarted) return;
    
    drawBackground();
    updatePlayer();
    updateBullets();
    updateEnemies();
    drawBullets();
    drawEnemies();
    drawPlayer();
    
    requestAnimationFrame(gameLoop);
}

updatePlayerList();
