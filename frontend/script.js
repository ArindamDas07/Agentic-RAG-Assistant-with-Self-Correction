const API_BASE_URL = "http://127.0.0.1:8000";

// ✅ Load user_id from sessionStorage (per tab)
let currentUserId = sessionStorage.getItem("user_id") || null;

// ✅ Store selected files (NO Ctrl needed)
let selectedFiles = [];

// DOM Elements
const pdfInput = document.getElementById('pdf-input');
const fileList = document.getElementById('file-list');
const uploadBtn = document.getElementById('upload-btn');
const statusContainer = document.getElementById('status-container');
const progressFill = document.getElementById('progress-fill');
const statusText = document.getElementById('status-text');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const chatWindow = document.getElementById('chat-window');


// ✅ On Page Load → Restore session + files
window.addEventListener("load", async () => {
    if (!currentUserId) return;

    try {
        const res = await fetch(`${API_BASE_URL}/uploads/${currentUserId}`);
        const data = await res.json();

        if (data.files && data.files.length > 0) {
            fileList.innerText = data.files.join(', ');
            unlockChat();
            appendMessage("bot", "Welcome back! Your documents are still loaded.");
        } else {
            fileList.innerText = "No files found";
            sessionStorage.removeItem("user_id");
            currentUserId = null;
        }

    } catch (err) {
        console.error("Failed to fetch uploaded files:", err);
    }
});


// ✅ 1. Add files WITHOUT Ctrl (accumulate)
pdfInput.addEventListener('change', () => {
    const newFiles = Array.from(pdfInput.files);

    // Add new files
    selectedFiles = [...selectedFiles, ...newFiles];

    // ✅ Remove duplicates
    const uniqueFiles = [];
    const names = new Set();

    for (const file of selectedFiles) {
        if (!names.has(file.name)) {
            uniqueFiles.push(file);
            names.add(file.name);
        }
    }

    selectedFiles = uniqueFiles;

    // Show files
    fileList.innerText = selectedFiles.length > 0
        ? selectedFiles.map(f => f.name).join(', ')
        : "No files selected";

    // Reset input so same file can be reselected
    pdfInput.value = "";
});


// ✅ 2. Upload + Process PDFs
uploadBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0)
        return alert("Please select PDFs first!");

    if (selectedFiles.length > 5)
        return alert("Maximum 5 files allowed!");

    statusContainer.classList.remove('hidden');
    uploadBtn.disabled = true;

    try {
        // Upload
        statusText.innerText = "Uploading to server...";
        progressFill.style.width = '30%';

        const formData = new FormData();

        for (const file of selectedFiles) {
            formData.append("files", file);
        }

        const uploadRes = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!uploadRes.ok) throw new Error("File upload failed.");

        const uploadData = await uploadRes.json();

        // Save user_id
        currentUserId = uploadData.user_id;
        sessionStorage.setItem("user_id", currentUserId);

        // Process PDFs
        statusText.innerText = "Embedding documents (this may take a minute)...";
        progressFill.style.width = '60%';

        const processRes = await fetch(`${API_BASE_URL}/process-pdf/${currentUserId}`, {
            method: 'POST'
        });

        if (!processRes.ok) throw new Error("Processing failed.");

        const processData = await processRes.json();

        // Update UI
        fileList.innerText = selectedFiles.map(f => f.name).join(', ');

        progressFill.style.width = '100%';
        statusText.innerText = `✅ Success! ${processData.chunks} chunks ready.`;

        unlockChat();

        // Clear selection after upload
        selectedFiles = [];

    } catch (error) {
        console.error(error);
        statusText.innerText = "❌ Error: " + error.message;
        statusText.style.color = "#ef4444";
        uploadBtn.disabled = false;
    }
});


// ✅ 3. Enable Chat
function unlockChat() {
    userInput.disabled = false;
    sendBtn.disabled = false;
    userInput.placeholder = "Ask a question about your documents...";
}


// ✅ 4. Send Message
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
    appendMessage("bot", "Analysing documents...", loadingId);

    try {
        const formData = new FormData();
        formData.append("question", text);

        const response = await fetch(`${API_BASE_URL}/ask/${currentUserId}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Could not get answer.");

        const data = await response.json();

        document.getElementById(loadingId).remove();
        appendMessage("bot", data.answer);

    } catch (error) {
        document.getElementById(loadingId).innerText =
            "❌ Error: " + error.message;
    }
}


// Event Listeners
sendBtn.addEventListener('click', sendMessage);

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});


// Helper: Append Message
function appendMessage(sender, text, id = null) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', `${sender}-message`);
    if (id) msgDiv.id = id;
    msgDiv.innerText = text;
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}