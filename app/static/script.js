document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const modelSelect = document.getElementById('model-select');

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;

        appendMessage(message, 'user');
        userInput.value = '';

        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message, model: modelSelect.value }),
        })
        .then(response => response.json())
        .then(data => {
            appendMessage(data.response, 'bot');
        })
        .catch(error => {
            console.error('Erreur:', error);
            appendMessage('Désolé, une erreur est survenue.', 'bot');
        });
    }

    function appendMessage(message, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        messageElement.textContent = message;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});
