const API_BASE = window.location.origin;
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
const attachmentPreviewContainer = document.getElementById("attachment-preview-container");

// Attached images store (holds raw base64 data)
let attachedImages = [];

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
    if (!text && attachedImages.length === 0) return;

    // Reset input height
    userInput.value = "";
    userInput.style.height = "auto";

    // Copy attached images to send and reset attachment UI
    const imagesToSend = [...attachedImages];
    attachedImages = [];
    attachmentPreviewContainer.innerHTML = "";
    attachmentPreviewContainer.style.display = "none";

    // Append user message to UI
    appendUserMessage(text, imagesToSend);

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
                mode: mode,
                images: imagesToSend.map(img => img.split(",")[1] || img) # Strip base64 headers
            })
        });

        // Remove typing indicator
        typingIndicator.remove();

        if (response.ok) {
            const data = await response.json();
            appendMessage("ai", data.response, data.citations, data.mode, data.workflow_steps);
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

function appendUserMessage(text, base64Images) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", "user");

    const avatar = document.createElement("div");
    avatar.classList.add("avatar");
    avatar.innerHTML = '<i class="fa-solid fa-user"></i>';

    const contentDiv = document.createElement("div");
    contentDiv.classList.add("msg-content");

    // Add images if attached
    if (base64Images && base64Images.length > 0) {
        const imgGallery = document.createElement("div");
        imgGallery.classList.add("message-image-gallery");
        base64Images.forEach(imgData => {
            const img = document.createElement("img");
            img.src = imgData;
            img.classList.add("message-preview-img");
            imgGallery.appendChild(img);
        });
        contentDiv.appendChild(imgGallery);
    }

    if (text) {
        const textPara = document.createElement("p");
        textPara.textContent = text;
        contentDiv.appendChild(textPara);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatFeed.appendChild(messageDiv);
    chatFeed.scrollTop = chatFeed.scrollHeight;
}

function appendMessage(sender, text, citations = [], mode = "", workflowSteps = []) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender);

    const avatar = document.createElement("div");
    avatar.classList.add("avatar");
    avatar.innerHTML = sender === "user" ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';

    const contentDiv = document.createElement("div");
    contentDiv.classList.add("msg-content");
    
    // Main text
    const textPara = document.createElement("p");
    textPara.innerHTML = formatMarkdown(text);
    contentDiv.appendChild(textPara);

    // Collapsible workflow steps (WOW factor)
    if (workflowSteps && workflowSteps.length > 0) {
        const details = document.createElement("details");
        details.classList.add("workflow-steps-details");
        
        const summary = document.createElement("summary");
        summary.innerHTML = '<i class="fa-solid fa-diagram-project"></i> Agentic Workflow Steps';
        details.appendChild(summary);
        
        const stepsList = document.createElement("ul");
        workflowSteps.forEach(step => {
            const li = document.createElement("li");
            li.textContent = step;
            stepsList.appendChild(li);
        });
        details.appendChild(stepsList);
        contentDiv.appendChild(details);
    }

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

// Basic markdown code block and bold formatter
function formatMarkdown(text) {
    if (!text) return "";
    let formatted = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
        
    // Bold
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    
    // Code blocks
    formatted = formatted.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
    
    // Inline code
    formatted = formatted.replace(/`(.*?)`/g, "<code>$1</code>");
    
    return formatted;
}

function appendTypingIndicator() {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", "ai", "typing-msg");

    const avatar = document.createElement("div");
    avatar.classList.add("avatar");
    avatar.innerHTML = '<i class="fa-solid fa-robot"></i>';

    const contentDiv = document.createElement("div");
    contentDiv.classList.add("msg-content");
    contentDiv.innerHTML = '<p><i class="fa-solid fa-spinner fa-spin"></i> Zenthi-AI OS is orchestrating...</p>';

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

// Drag & Drop / File selection logic
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
        handleFileSelect(files[0]);
    }
});

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    const isImage = file.type.startsWith("image/");
    if (isImage) {
        // Read image and display preview for thread submission
        const reader = new FileReader();
        reader.onload = function(e) {
            const base64Data = e.target.result;
            attachedImages.push(base64Data);
            renderAttachmentPreviews();
        };
        reader.readAsDataURL(file);
    } else {
        // Index documents into ChromaDB
        uploadFile(file);
    }
}

function renderAttachmentPreviews() {
    attachmentPreviewContainer.innerHTML = "";
    if (attachedImages.length === 0) {
        attachmentPreviewContainer.style.display = "none";
        return;
    }
    
    attachedImages.forEach((imgSrc, index) => {
        const item = document.createElement("div");
        item.classList.add("attachment-preview-item");
        
        const img = document.createElement("img");
        img.src = imgSrc;
        
        const removeBtn = document.createElement("button");
        removeBtn.innerHTML = '<i class="fa-solid fa-xmark"></i>';
        removeBtn.onclick = () => {
            attachedImages.splice(index, 1);
            renderAttachmentPreviews();
        };
        
        item.appendChild(img);
        item.appendChild(removeBtn);
        attachmentPreviewContainer.appendChild(item);
    });
    
    attachmentPreviewContainer.style.display = "flex";
}

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
