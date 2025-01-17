// frontend/script.js

document.getElementById('send-button').addEventListener('click', sendSymptoms);
document.getElementById('symptoms-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendSymptoms();
    }
});

function appendMessage(sender, message) {
    const chatBox = document.getElementById('chat-box');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(sender === 'user' ? 'user-message' : 'bot-message');

    const avatar = document.createElement('div');
    avatar.classList.add('avatar');
    avatar.innerHTML = sender === 'user' ? '👤' : '🤖'; // Using emojis as simple avatars

    const messageContent = document.createElement('div');
    messageContent.classList.add('message-content');
    messageContent.innerHTML = message;

    // Timestamp
    const timestamp = document.createElement('span');
    timestamp.classList.add('timestamp');
    const now = new Date();
    timestamp.innerText = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    messageContent.appendChild(timestamp);

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendSymptoms() {
    const input = document.getElementById('symptoms-input');
    const symptoms = input.value.trim();

    if (symptoms === '') {
        alert('Please enter at least one symptom.');
        return;
    }

    appendMessage('user', symptoms);
    input.value = '';

    try {
        // Display a typing indicator
        appendMessage('bot', '🤖 Typing...');

        // Call /predict endpoint
        const predictResponse = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symptoms: symptoms.split(',').map(s => s.trim()) })
        });

        // Remove the typing indicator
        removeLastBotMessage();

        if (!predictResponse.ok) {
            const errorData = await predictResponse.json();
            appendMessage('bot', `❌ Error predicting disease: ${errorData.detail}`);
            return;
        }

        const predictData = await predictResponse.json();
        const disease = predictData.disease;
        const confidence = predictData.confidence;

        appendMessage('bot', `✅ Predicted Disease: ${disease} (Confidence: ${(confidence * 100).toFixed(1)}%)`);

        // Display another typing indicator for explanation
        appendMessage('bot', '🤖 Fetching detailed explanation...');

        // Call /explain endpoint
        const explainResponse = await fetch('http://127.0.0.1:8000/explain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ disease: disease })
        });

        // Remove the typing explanation message
        removeLastBotMessage();

        if (!explainResponse.ok) {
            const errorData = await explainResponse.json();
            appendMessage('bot', `❌ Error explaining disease: ${errorData.detail}`);
            return;
        }

        const explainData = await explainResponse.json();
        const explanation = explainData.explanation;

        // Format the explanation with bullet points for better readability
        const formattedExplanation = explanation
            .replace(/■/g, '•') // Replace ■ with • for standard bullets
            .replace(/\n/g, '<br>'); // Replace newlines with line breaks

        appendMessage('bot', `📄 <strong>Disease Explanation:</strong><br>${formattedExplanation}`);
    } catch (error) {
        // Remove the typing messages in case of errors
        removeLastBotMessage();
        removeLastBotMessage();
        appendMessage('bot', `❗ An unexpected error occurred: ${error.message}`);
    }
}

function removeLastBotMessage() {
    const chatBox = document.getElementById('chat-box');
    const messages = chatBox.getElementsByClassName('message');
    if (messages.length > 0) {
        const lastMessage = messages[messages.length - 1];
        if (lastMessage.classList.contains('bot-message')) {
            chatBox.removeChild(lastMessage);
        }
    }
}
