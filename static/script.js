document.addEventListener('DOMContentLoaded', () => {
    const state = {
        currentConversationId: null,
        conversations: {},
        settings: {
            system_prompt: "You are a helpful assistant.",
            temperature: 0.3,
            min_p: 0.15,
            repetition_penalty: 1.05,
        },
        rag_enabled: false,
    };

    // DOM Elements
    const modelSelector = document.getElementById('model-selector');
    const ejectButton = document.getElementById('eject-button');
    const settingsButton = document.getElementById('settings-button');
    const newChatButton = document.getElementById('new-chat-button');
    const historyList = document.getElementById('history-list');
    const chatArea = document.getElementById('chat-area');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const ragToggle = document.getElementById('rag-toggle');
    const loadDocsButton = document.getElementById('load-docs-button');
    const documentUpload = document.getElementById('document-upload');
    const ragStatus = document.getElementById('rag-status');
    const settingsModal = document.getElementById('settings-modal');
    const closeModalButton = document.querySelector('.close-button');
    const settingsForm = document.getElementById('settings-form');

    // --- API Functions ---
    const api = {
        get: (url) => fetch(url).then(res => res.json()),
        post: (url, data) => fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        }).then(res => res.json()),
        delete: (url) => fetch(url, { method: 'DELETE' }).then(res => res.json()),
        upload: (url, formData) => fetch(url, {
            method: 'POST',
            body: formData,
        }).then(res => res.json()),
    };

    // --- UI Update Functions ---
    function renderHistory() {
        historyList.innerHTML = '';
        Object.keys(state.conversations).forEach(convId => {
            const history = state.conversations[convId];
            const title = history.find(m => m.role === 'user')?.content.substring(0, 25) || 'Nouvelle Discussion';
            const item = document.createElement('div');
            item.className = 'history-item';
            item.textContent = title + '...';
            item.dataset.id = convId;
            if (convId === state.currentConversationId) {
                item.classList.add('active');
            }
            historyList.appendChild(item);
        });
    }

    function renderChat() {
        chatArea.innerHTML = '';
        if (!state.currentConversationId) return;

        const history = state.conversations[state.currentConversationId];
        history.forEach(msg => {
            if (msg.role === 'system') return;
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${msg.role}-message`;
            messageDiv.innerHTML = `<b>${msg.role === 'user' ? 'Vous' : 'LiquidAI'}:</b><br>${msg.content}`;
            chatArea.appendChild(messageDiv);
        });
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    // --- Event Handlers ---
    async function handleNewChat() {
        const { conversation_id, history } = await api.post('/api/conversations');
        state.currentConversationId = conversation_id;
        state.conversations[conversation_id] = history;
        renderHistory();
        renderChat();
    }

    async function handleSendMessage() {
        const message = userInput.value.trim();
        if (!message || !state.currentConversationId) return;

        userInput.value = '';
        state.conversations[state.currentConversationId].push({ role: 'user', content: message });
        renderChat();

        const response = await api.post('/api/chat', {
            conversation_id: state.currentConversationId,
            message: message,
            rag_enabled: state.rag_enabled,
            settings: state.settings,
        });

        state.conversations[state.currentConversationId].push({ role: 'assistant', content: response.response });
        renderChat();
        renderHistory(); // Update title
    }

    function handleHistoryClick(e) {
        if (e.target.classList.contains('history-item')) {
            state.currentConversationId = e.target.dataset.id;
            renderHistory();
            renderChat();
        }
    }

    async function handleDocUpload() {
        const files = documentUpload.files;
        if (files.length === 0) return;

        const formData = new FormData();
        for (const file of files) {
            formData.append('file', file);
        }

        ragStatus.textContent = 'Chargement...';
        await api.upload('/api/documents', formData);
        ragStatus.textContent = 'Documents à jour';
    }

    function openSettingsModal() {
        settingsForm.system_prompt.value = state.settings.system_prompt;
        settingsForm.temperature.value = state.settings.temperature;
        settingsForm.min_p.value = state.settings.min_p;
        settingsForm.repetition_penalty.value = state.settings.repetition_penalty;
        settingsModal.style.display = 'block';
    }

    function closeSettingsModal() {
        settingsModal.style.display = 'none';
    }

    function handleSaveSettings(e) {
        e.preventDefault();
        state.settings.system_prompt = settingsForm.system_prompt.value;
        state.settings.temperature = parseFloat(settingsForm.temperature.value);
        state.settings.min_p = parseFloat(settingsForm.min_p.value);
        state.settings.repetition_penalty = parseFloat(settingsForm.repetition_penalty.value);
        closeSettingsModal();
    }

    // --- Initialization ---
    async function initialize() {
        // Load models
        const models = await api.get('/api/models');
        modelSelector.innerHTML = models.map(m => `<option>${m}</option>`).join('');

        // Load conversations
        state.conversations = await api.get('/api/conversations');
        if (Object.keys(state.conversations).length > 0) {
            state.currentConversationId = Object.keys(state.conversations)[0];
        } else {
            await handleNewChat();
        }

        renderHistory();
        renderChat();

        // Event Listeners
        newChatButton.addEventListener('click', handleNewChat);
        sendButton.addEventListener('click', handleSendMessage);
        userInput.addEventListener('keydown', (e) => e.key === 'Enter' && handleSendMessage());
        historyList.addEventListener('click', handleHistoryClick);
        ragToggle.addEventListener('change', () => state.rag_enabled = ragToggle.checked);
        loadDocsButton.addEventListener('click', () => documentUpload.click());
        documentUpload.addEventListener('change', handleDocUpload);
        settingsButton.addEventListener('click', openSettingsModal);
        closeModalButton.addEventListener('click', closeSettingsModal);
        settingsForm.addEventListener('submit', handleSaveSettings);
        window.addEventListener('click', (e) => e.target === settingsModal && closeSettingsModal());
    }

    initialize();
});
