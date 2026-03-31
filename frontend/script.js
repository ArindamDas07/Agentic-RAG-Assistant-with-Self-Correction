const API_BASE_URL = "http://127.0.0.1:8000";

let currentUserId = sessionStorage.getItem("user_id") || null;
let selectedFiles = [];

// DOM
const pdfInput = document.getElementById('pdf-input');
const fileList = document.getElementById('file-list');
const uploadBtn = document.getElementById('upload-btn');
const statusContainer = document.getElementById('status-container');
const progressFill = document.getElementById('progress-fill');
const statusText = document.getElementById('status-text');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const chatWindow = document.getElementById('chat-window');


// =========================
// ✅ Restore session
// =========================
window.addEventListener("load", async () => {
    if (!currentUserId) return;

    try {
        const res = await fetch(`${API_BASE_URL}/uploads/${currentUserId}`);
        const data = await res.json();

        if (data.files && data.files.length > 0) {
            fileList.innerText = data.files.join(', ');
            unlockChat();
            appendMessage("bot", "✅ Your documents are ready. Ask your question!");
        } else {
            sessionStorage.removeItem("user_id");
            currentUserId = null;
        }

    } catch (err) {
        console.error(err);
    }
});


// =========================
// ✅ File selection
// =========================
pdfInput.addEventListener('change', () => {
    const newFiles = Array.from(pdfInput.files);

    selectedFiles = [...selectedFiles, ...newFiles];

    const uniqueFiles = [];
    const names = new Set();

    for (const file of selectedFiles) {
        if (!names.has(file.name)) {
            uniqueFiles.push(file);
            names.add(file.name);
        }
    }

    selectedFiles = uniqueFiles;

    fileList.innerText = selectedFiles.length > 0
        ? selectedFiles.map(f => f.name).join(', ')
        : "No files selected";

    pdfInput.value = "";
});


// =========================
// ✅ Upload + Process
// =========================
uploadBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0)
        return alert("Please select PDFs first!");

    if (selectedFiles.length > 5)
        return alert("Maximum 5 files allowed!");

    statusContainer.classList.remove('hidden');
    uploadBtn.disabled = true;

    try {
        // Upload
        statusText.innerText = "Uploading...";
        progressFill.style.width = '30%';

        const formData = new FormData();
        selectedFiles.forEach(file => formData.append("files", file));

        const uploadRes = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!uploadRes.ok) throw new Error("Upload failed");

        const uploadData = await uploadRes.json();

        currentUserId = uploadData.user_id;
        sessionStorage.setItem("user_id", currentUserId);

        // Start processing
        statusText.innerText = "Processing documents...";
        progressFill.style.width = '60%';

        await fetch(`${API_BASE_URL}/process-pdf/${currentUserId}`, {
            method: 'POST'
        });

        // 👇 IMPORTANT: Start polling status
        checkProcessingStatus();

    } catch (err) {
        statusText.innerText = "❌ Error: " + err.message;
        uploadBtn.disabled = false;
    }
});


// =========================
// ✅ Poll processing status
// =========================
async function checkProcessingStatus() {
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/status/${currentUserId}`);
            const data = await res.json();

            if (data.status === "completed") {
                clearInterval(interval);

                progressFill.style.width = "100%";
                statusText.innerText = "✅ Processing complete!";

                unlockChat();

                // 👇 THIS IS WHAT YOU WANTED
                appendMessage("bot", "🎉 Documents processed successfully! You can now ask questions.");

            } else if (data.status.startsWith("error")) {
                clearInterval(interval);

                statusText.innerText = "❌ " + data.status;
                uploadBtn.disabled = false;

            } else {
                // still running
                statusText.innerText = "Processing documents...";
            }

        } catch (err) {
            console.error(err);
        }
    }, 2000); // every 2 sec
}


// =========================
// ✅ Enable Chat
// =========================
function unlockChat() {
    userInput.disabled = false;
    sendBtn.disabled = false;
    userInput.placeholder = "Ask a question about your documents...";
}


// =========================
// ✅ Send Message
// =========================
async function sendMessage() {
    const text = userInput.value.trim();

    if (!text) return;

    if (!currentUserId) {
        appendMessage("bot", "⚠️ Please upload and process PDFs first.");
        return;
    }

    appendMessage("user", text);
    userInput.value = "";

    const loadingId = "load-" + Date.now();
    appendMessage("bot", "🔍 Analysing documents...", loadingId);

    try {
        const formData = new FormData();
        formData.append("question", text);

        const response = await fetch(`${API_BASE_URL}/ask/${currentUserId}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Failed to get answer");

        const data = await response.json();

        document.getElementById(loadingId).remove();

        appendMessage("bot", data.answer.trim());

    } catch (error) {
        document.getElementById(loadingId).innerText =
            "❌ Error: " + error.message;
    }
}


// =========================
// Events
// =========================
sendBtn.addEventListener('click', sendMessage);

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});


// =========================
// UI helper
// =========================
function appendMessage(sender, text, id = null) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', `${sender}-message`);
    if (id) msgDiv.id = id;
    msgDiv.innerText = text;
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}