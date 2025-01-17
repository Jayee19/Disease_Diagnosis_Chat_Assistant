document.addEventListener('DOMContentLoaded', function() {
    const messagesContainer = document.getElementById('messagesContainer');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');

    // Add welcome message
    addBotMessage("Hello! I'm your Disease Diagnosis Assistant. Please describe your symptoms, and I'll help identify possible conditions.");

    async function handleSendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Add user message
        addUserMessage(message);
        userInput.value = '';
        
        // Show typing indicator
        showTypingIndicator();

        try {
            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ symptoms: message })
            });

            if (!response.ok) throw new Error('Failed to get prediction');
            
            const data = await response.json();
            removeTypingIndicator();
            
            // Add prediction message
            addBotMessage(`Based on your symptoms, you might have: ${data.predicted_disease}`);

            // Get detailed explanation
            try {
                const explanationResponse = await fetch('http://127.0.0.1:8000/explain', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ disease_name: data.predicted_disease })
                });

                if (explanationResponse.ok) {
                    const explanationData = await explanationResponse.json();
                    addBotMessage(explanationData.explanation);
                }
            } catch (error) {
                console.error('Failed to get explanation:', error);
            }

        } catch (error) {
            removeTypingIndicator();
            addErrorMessage('Sorry, I had trouble processing your symptoms. Please try again with more specific symptoms.');
        }
    }

    function addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.textContent = text;
        messagesContainer.appendChild(messageDiv);
        scrollToBottom();
    }

    function addBotMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        messageDiv.textContent = text;
        messagesContainer.appendChild(messageDiv);
        scrollToBottom();
    }

    function addErrorMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message error-message';
        messageDiv.textContent = text;
        messagesContainer.appendChild(messageDiv);
        scrollToBottom();
    }

    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typingIndicator';
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'typing-dot';
            typingDiv.appendChild(dot);
        }
        messagesContainer.appendChild(typingDiv);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Event listeners
    sendButton.addEventListener('click', handleSendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            handleSendMessage();
        }
    });
});