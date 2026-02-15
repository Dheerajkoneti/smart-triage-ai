document.addEventListener("DOMContentLoaded", function () {

    // ================================
    // 🎯 ELEMENTS
    // ================================
    const toggleBtn = document.getElementById("chat-toggle");
    const chatBox = document.getElementById("chatbox");
    const chatMessages = document.getElementById("chat-messages");
    const userInput = document.getElementById("chat-input-field");
    const micBtn = document.getElementById("mic-btn");
    const roleSelect = document.getElementById("chat-role");
    const languageSelect = document.getElementById("chat-language");

    if (!toggleBtn || !chatBox) return;

    // ================================
    // 🔥 TOGGLE CHATBOX
    // ================================
    toggleBtn.addEventListener("click", () => {
        chatBox.classList.toggle("hidden");
    });

    // ================================
    // 💬 SEND MESSAGE FUNCTION
    // ================================
    window.sendMessage = function () {

        const message = userInput.value.trim();
        if (!message) return;

        const selectedRole = roleSelect ? roleSelect.value : "triage";
        const selectedLanguage = languageSelect ? languageSelect.value : "en-US";

        // Add user message
        chatMessages.innerHTML += `
            <div class="chat-bubble user">${message}</div>
        `;

        userInput.value = "";
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Typing indicator
        const typing = document.createElement("div");
        typing.className = "chat-bubble ai";
        typing.id = "typing-indicator";
        typing.innerHTML = "AI is typing...";
        chatMessages.appendChild(typing);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Send to backend
        fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: message,
                role: selectedRole,
                language: selectedLanguage
            })
        })
        .then(res => res.json())
        .then(data => {

            const typingIndicator = document.getElementById("typing-indicator");
            if (typingIndicator) typingIndicator.remove();

            chatMessages.innerHTML += `
                <div class="chat-bubble ai">${data.reply}</div>
            `;

            chatMessages.scrollTop = chatMessages.scrollHeight;

            // 🔊 Speak AI reply in selected language
            if ('speechSynthesis' in window) {
                const speech = new SpeechSynthesisUtterance(data.reply);
                speech.lang = selectedLanguage;
                window.speechSynthesis.speak(speech);
            }

        })
        .catch(() => {

            const typingIndicator = document.getElementById("typing-indicator");
            if (typingIndicator) typingIndicator.remove();

            chatMessages.innerHTML += `
                <div class="chat-bubble ai">Error: AI service unavailable.</div>
            `;
        });
    };

    // ================================
    // ⌨️ ENTER KEY SUPPORT
    // ================================
    userInput.addEventListener("keypress", function (e) {
        if (e.key === "Enter") {
            sendMessage();
        }
    });

    // ================================
    // 🎤 VOICE INPUT FEATURE
    // ================================
    if (micBtn && ('webkitSpeechRecognition' in window)) {

        const recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;

        micBtn.addEventListener("click", () => {
            recognition.lang = languageSelect ? languageSelect.value : "en-US";
            recognition.start();
            micBtn.classList.add("listening");
        });

        recognition.onresult = function (event) {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript;
            micBtn.classList.remove("listening");
        };

        recognition.onerror = function () {
            micBtn.classList.remove("listening");
        };

        recognition.onend = function () {
            micBtn.classList.remove("listening");
        };

    } else {
        if (micBtn) micBtn.style.display = "none";
    }

});