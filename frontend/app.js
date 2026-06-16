const API_BASE = "http://localhost:8000";
let sessionId = localStorage.getItem("zenthi_session_id") || uuidv4();
localStorage.setItem("zenthi_session_id", sessionId);

// Elements
const chatFeed = document.getElementById("chat-feed");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const clearChatBtn = document.getElementById("clear-chat-btn");
const modeSelect = document.getElementById("mode-select");
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const uploadedFilesList = document.getElementById("uploaded-files-list");

// Helper: Generate UUID
function uuidv4() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Auto-expand input area
userInput.addEventListener("input", function() {
    this.style.height = "auto";
    this.style.height = (this.scrollHeight) + "px";
});

// Send message on Enter key (without shift)
userInput.addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener("click", sendMessage);

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Reset input height
    userInput.value = "";
    userInput.style.height = "auto";

    // Append user message
    appendMessage("user", text);

    // Show typing indicator
    const typingIndicator = appendTypingIndicator();
    chatFeed.scrollTop = chatFeed.scrollHeight;

    const mode = modeSelect.value;

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                query: text,
                session_id: sessionId,
                mode: mode
            })
        });

        // Remove typing indicator
        typingIndicator.remove();

        if (response.ok) {
            const data = await response.json();
            appendMessage("ai", data.response, data.citations, data.mode);
        } else {
            const errorMsg = await response.text();
            appendMessage("ai", `Error: ${errorMsg}`);
        }
    } catch (err) {
        typingIndicator.remove();
        appendMessage("ai", `Connection Error: Make sure backend API is running on ${API_BASE}`);
    }

    chatFeed.scrollTop = chatFeed.scrollHeight;
}

function appendMessage(sender, text, citations = [], mode = "") {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender);

    const avatar = document.createElement("div");
    avatar.classList.add("avatar");
    avatar.innerHTML = sender === "user" ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';

    const contentDiv = document.createElement("div");
    contentDiv.classList.add("msg-content");
    
    const textPara = document.createElement("p");
    textPara.textContent = text;
    contentDiv.appendChild(textPara);

    // Append Mode used
    if (mode) {
        const modeBadge = document.createElement("div");
        modeBadge.classList.add("mode-badge");
        modeBadge.innerHTML = `<i class="fa-solid fa-code-branch"></i> Mode: ${mode}`;
        contentDiv.appendChild(modeBadge);
    }

    // Append Citations
    if (citations && citations.length > 0) {
        const citationsDiv = document.createElement("div");
        citationsDiv.classList.add("citations-container");
        
        citations.forEach(cit => {
            const badge = document.createElement("span");
            badge.classList.add("citation-badge");
            badge.innerHTML = `<i class="fa-solid fa-bookmark"></i> ${cit}`;
            citationsDiv.appendChild(badge);
        });
        contentDiv.appendChild(citationsDiv);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatFeed.appendChild(messageDiv);
    chatFeed.scrollTop = chatFeed.scrollHeight;
}

function appendTypingIndicator() {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", "ai", "typing-msg");

    const avatar = document.createElement("div");
    avatar.classList.add("avatar");
    avatar.innerHTML = '<i class="fa-solid fa-robot"></i>';

    const contentDiv = document.createElement("div");
    contentDiv.classList.add("msg-content");
    contentDiv.innerHTML = '<p><i class="fa-solid fa-spinner fa-spin"></i> Zenthi-AI is thinking...</p>';

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatFeed.appendChild(messageDiv);
    return messageDiv;
}

// Clear chat
clearChatBtn.addEventListener("click", () => {
    sessionId = uuidv4();
    localStorage.setItem("zenthi_session_id", sessionId);
    chatFeed.innerHTML = `
        <div class="message system">
            <div class="avatar"><i class="fa-solid fa-robot"></i></div>
            <div class="msg-content">
                <p>Chat history cleared. Session reset. Ready for new conversations!</p>
            </div>
        </div>
    `;
});

// Drag & Drop
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
});

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        uploadFile(e.target.files[0]);
    }
});

async function uploadFile(file) {
    uploadStatus.textContent = "Uploading and indexing document...";
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            uploadStatus.textContent = `Uploaded successfully!`;
            
            // Add file to list
            const li = document.createElement("li");
            li.innerHTML = `<span><i class="fa-solid fa-file-lines"></i> ${file.name}</span> <span class="badge">Indexed</span>`;
            uploadedFilesList.appendChild(li);
        } else {
            const err = await response.text();
            uploadStatus.textContent = `Upload failed: ${err}`;
        }
    } catch (err) {
        uploadStatus.textContent = `Error connecting to indexing backend.`;
    }
}
