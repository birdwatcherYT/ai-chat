const chatBox = document.getElementById("chat-box");
const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const micButton = document.getElementById("mic-button");
const statusArea = document.getElementById("status-area");
const imagePanel = document.getElementById("image-panel");
const contextImage = document.getElementById("context-image");
const attachButton = document.getElementById("attach-button");
const imageInput = document.getElementById("image-input");
const previewArea = document.getElementById("preview-area");
const webcamButton = document.getElementById("webcam-button");
const webcamPanel = document.getElementById("webcam-panel");
const webcamVideo = document.getElementById("webcam-video");
const scaleSlider = document.getElementById("scale-slider");
const scaleValueLabel = document.getElementById("scale-value");
const startOverlay = document.getElementById("start-overlay");

const audioContext = new (window.AudioContext || window.webkitAudioContext)();
const ws = new WebSocket(`ws://${window.location.host}/ws`);

let audioQueue = [],
    isPlaying = false,
    currentAiMessageElement = null,
    userName = "U";
let mediaRecorder,
    audioChunks = [],
    stream;
let isContinuousMode = false;
let aiTurnFinished = true;
let recognition;
let userInputMode = "browser_asr";
let isFullAutoMode = false;
let manualStop = false;
let tempUserMessageElement = null;
let attachedImage = null;
let isWebcamOn = false;
let webcamStream = null;
let silenceDetectionTimer = null;
let audioStreamSource = null;

ws.onopen = () => {
    /* 初期化は onmessage の config で行う */
};
ws.onclose = () => {
    update_status("main", "サーバーとの接続が切れました。");
    disable_input(true);
};
ws.onerror = (e) => {
    update_status("main", "エラーが発生しました。");
    disable_input(true);
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log("[RECEIVE]", message);

    switch (message.type) {
        case "config":
            userName = message.data.user_name;
            userInputMode = message.data.user_input_mode;

            // AIモードの場合はオーバーレイを表示する
            if (userInputMode === "ai") {
                isFullAutoMode = true;
                disable_input(true);
                micButton.style.display = "none";
                sendButton.style.display = "none";
                attachButton.style.display = "none";
                webcamButton.style.display = "none";
                messageInput.placeholder = "全自動モードです";
                startOverlay.style.display = "flex"; // オーバーレイを表示
            } else {
                micButton.style.display = "flex";
                if (userInputMode === "browser_asr") {
                    initializeSpeechRecognition();
                    micButton.title = `常時音声入力 (ブラウザ認識)`;
                } else {
                    micButton.title = `常時音声入力 (サーバー認識)`;
                }
                // 通常モードではすぐに会話可能にする
                finish_ai_turn();
            }
            console.log(`GUI Input mode set to: ${userInputMode}`);
            break;

        case "user_transcription":
            remove_status("asr-server");
            remove_temp_user_message();
            if (message.data && message.data.trim()) {
                append_message(userName, message.data);
            }
            break;
        case "retry_audio_input":
            console.log("ASR failed or empty. Retrying input.");
            remove_status("asr-server");
            remove_temp_user_message();
            finish_ai_turn();
            break;
        case "history":
            const content = message.data.content;
            const textOnly = content.replace(/\n\(画像添付\)$/, "");
            append_message(message.data.name, textOnly, null);
            break;
        case "next_speaker":
            aiTurnFinished = false;
            if (micButton.classList.contains("recording")) {
                manualStop = true;
                stopRecording();
            }
            remove_status("speaker-decision");
            update_status(
                "ai_thinking",
                `🧠 ${message.data}が発言を考えています...`,
            );
            if (message.data !== userName || isFullAutoMode) {
                currentAiMessageElement = append_message(message.data, "");
            }
            break;
        case "chunk":
            update_ai_message(message.data.content);
            break;
        case "utterance_end":
            remove_status("ai_thinking");
            if (!isFullAutoMode) {
                update_status(
                    "speaker-decision",
                    "➡️ 次の話者を決めています...",
                );
            }
            currentAiMessageElement = null;
            break;
        case "audio":
            audioQueue.push(message.data);
            if (!isPlaying) {
                play_from_queue();
            }
            break;
        case "conversation_end":
            finish_ai_turn();
            break;
        case "image":
            imagePanel.dataset.state = "loaded";
            contextImage.src = message.url;
            break;
        case "status_update":
            update_status(message.data.id, message.data.text);
            break;
        case "status_remove":
            remove_status(message.data.id);
            break;
    }
};

// オーバーレイがクリックされた時の処理
startOverlay.onclick = async () => {
    try {
        // AudioContextを有効化する (これが最も重要)
        await audioContext.resume();
        console.log("AudioContext resumed by user gesture.");

        // オーバーレイを隠す
        startOverlay.style.display = "none";
        update_status("main", "🤖 全自動モードで実行中...");

        // サーバーにAI会話の開始を通知する
        ws.send(JSON.stringify({ type: "start_ai_conversation" }));
    } catch (err) {
        console.error("Failed to resume AudioContext:", err);
        statusArea.textContent = "音声の再生を開始できませんでした。";
    }
};

const initializeSpeechRecognition = () => {
    if (!("webkitSpeechRecognition" in window)) {
        alert("このブラウザは音声認識をサポートしていません。");
        micButton.disabled = true;
        return;
    }
    recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "ja-JP";
    recognition.onresult = (event) => {
        let interimTranscript = "";
        let finalTranscript = "";
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript;
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }
        update_temp_user_message(interimTranscript);
        if (finalTranscript) {
            const trimmedTranscript = finalTranscript.trim();
            manualStop = true;
            recognition.abort();
            sendUserMessage(trimmedTranscript);
        }
    };
    recognition.onend = () => {
        micButton.classList.remove("recording");
        micButton.textContent = isContinuousMode ? "🎙️" : "🎤";
        remove_temp_user_message();
        remove_status("main-prompt");
        if (manualStop) {
            manualStop = false;
            return;
        }
        if (isContinuousMode && aiTurnFinished && !isPlaying) {
            setTimeout(() => startRecording(), 100);
        } else if (!isContinuousMode) {
            enable_input();
        }
    };
    recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        if (event.error !== "aborted") {
            update_status("asr-failed", `音声認識エラー: ${event.error}`, 3000);
        }
    };
};
const update_status = (id, text, timeout = 0) => {
    let el = document.getElementById(id);
    if (!el) {
        el = document.createElement("div");
        el.id = id;
        el.className = "status-item";
        statusArea.appendChild(el);
    }
    el.textContent = text;
    if (timeout > 0) {
        setTimeout(() => {
            if (el) el.remove();
        }, timeout);
    }
};
const remove_status = (id) => {
    const el = document.getElementById(id);
    if (el) el.remove();
};
const enable_input = (isManual = true) => {
    if (isFullAutoMode) return;
    messageInput.disabled = false;
    sendButton.disabled = false;
    micButton.disabled = false;
    attachButton.disabled = false;
    if (!isContinuousMode || isManual) {
        remove_status("main-prompt");
        update_status(
            "main-prompt",
            "メッセージを入力またはマイクを押してください。",
        );
    }
};
const disable_input = (isError = false) => {
    messageInput.disabled = true;
    sendButton.disabled = true;
    micButton.disabled = true;
    if (!isFullAutoMode) remove_status("main-prompt");
};
const append_message = (name, text, imageSrc = null) => {
    const isUser = name === userName;
    const messageClass =
        isUser && !isFullAutoMode ? "user-message" : "ai-message";
    const el = document.createElement("div");
    el.classList.add("message", messageClass);
    let imageHTML = "";
    if (imageSrc) {
        imageHTML = `<img src="${imageSrc}" class="message-image" alt="添付画像">`;
    }
    el.innerHTML = `<strong class="name">${name}</strong>${imageHTML}<p>${text || ""}</p>`;
    chatBox.appendChild(el);
    chatBox.scrollTop = chatBox.scrollHeight;
    return el;
};
const update_ai_message = (chunk) => {
    if (currentAiMessageElement) {
        currentAiMessageElement.querySelector("p").textContent += chunk;
        chatBox.scrollTop = chatBox.scrollHeight;
    }
};
const create_temp_user_message = () => {
    if (tempUserMessageElement) return;
    tempUserMessageElement = document.createElement("div");
    tempUserMessageElement.classList.add(
        "message",
        "user-message",
        "temp-message",
    );
    tempUserMessageElement.innerHTML = `<strong class="name">${userName}</strong><p></p>`;
    chatBox.appendChild(tempUserMessageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
};
const update_temp_user_message = (text) => {
    if (!tempUserMessageElement) create_temp_user_message();
    if (tempUserMessageElement) {
        tempUserMessageElement.querySelector("p").textContent = text;
        chatBox.scrollTop = chatBox.scrollHeight;
    }
};
const remove_temp_user_message = () => {
    if (tempUserMessageElement) {
        tempUserMessageElement.remove();
        tempUserMessageElement = null;
    }
};
const finish_ai_turn = () => {
    currentAiMessageElement = null;
    remove_status("ai_thinking");
    remove_status("speaker-decision");
    aiTurnFinished = true;
    micButton.classList.remove("recording");
    micButton.textContent = isContinuousMode ? "🎙️" : "🎤";
    if (!isPlaying) {
        if (isContinuousMode) {
            enable_input(false);
            setTimeout(() => {
                if (aiTurnFinished && !isPlaying) {
                    startRecording();
                }
            }, 500);
        } else {
            enable_input(true);
        }
    }
};
const play_from_queue = () => {
    if (audioQueue.length === 0) {
        isPlaying = false;
        if (aiTurnFinished) {
            if (isContinuousMode) {
                enable_input(false);
                setTimeout(() => {
                    if (aiTurnFinished && !isPlaying) {
                        startRecording();
                    }
                }, 500);
            } else {
                enable_input(true);
            }
        }
        return;
    }
    if (!isPlaying) {
        if (isContinuousMode && stream && userInputMode === "server_asr") {
            stream.getAudioTracks().forEach((track) => (track.enabled = false));
            console.log("🎤 Mic Muted by AI speech (Server ASR)");
        }
    }
    isPlaying = true;
    const { audio, samplerate, dtype } = audioQueue.shift();
    const rawAudio = atob(audio);
    const buffer = new ArrayBuffer(rawAudio.length);
    const view = new Uint8Array(buffer);
    for (let i = 0; i < rawAudio.length; i++) view[i] = rawAudio.charCodeAt(i);
    let float32Array;
    if (dtype === "int16") {
        const i16 = new Int16Array(buffer);
        float32Array = new Float32Array(i16.length);
        for (let i = 0; i < i16.length; i++) float32Array[i] = i16[i] / 32768.0;
    } else if (dtype === "float32") {
        float32Array = new Float32Array(buffer);
    } else {
        isPlaying = false;
        play_from_queue();
        return;
    }
    const audioBuffer = audioContext.createBuffer(
        1,
        float32Array.length,
        samplerate,
    );
    audioBuffer.copyToChannel(float32Array, 0);
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.onended = () => {
        isPlaying = false;
        play_from_queue();
    };
    source.start();
};
const sendUserMessage = (message) => {
    const text = message || "";
    const image = attachedImage || null;
    let webcamCapture = null;
    if (isWebcamOn && webcamStream) {
        const video = webcamVideo;
        const canvas = document.createElement("canvas");
        const scale = parseInt(scaleSlider.value) / 100;
        const targetWidth = Math.round(video.videoWidth * scale);
        const targetHeight = Math.round(video.videoHeight * scale);
        canvas.width = targetWidth;
        canvas.height = targetHeight;
        canvas
            .getContext("2d")
            .drawImage(video, 0, 0, targetWidth, targetHeight);
        webcamCapture = canvas.toDataURL("image/jpeg");
    }
    if (text.trim() || image || webcamCapture) {
        remove_temp_user_message();
        append_message(userName, text, image);
        ws.send(JSON.stringify({ text, image, webcam_capture: webcamCapture }));
        disable_input();
        update_status("speaker-decision", "➡️ 次の話者を決めています...");
        clearAttachment();
    }
};
const handleTextInput = () => {
    const message = messageInput.value;
    if ((message && !messageInput.disabled) || attachedImage) {
        sendUserMessage(message);
        messageInput.value = "";
    }
};
const handleImageAttachment = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        attachedImage = e.target.result;
        previewArea.innerHTML = ` <div class="preview-container"> <img src="${attachedImage}" class="image-preview" alt="Preview"/> <button class="remove-preview-btn">×</button> </div>`;
        previewArea.querySelector(".remove-preview-btn").onclick =
            clearAttachment;
    };
    reader.readAsDataURL(file);
    imageInput.value = "";
};
const clearAttachment = () => {
    attachedImage = null;
    previewArea.innerHTML = "";
};
const startRecording = async () => {
    if (isFullAutoMode || micButton.classList.contains("recording")) return;
    manualStop = false;
    messageInput.disabled = true;
    sendButton.disabled = true;
    micButton.classList.add("recording");
    micButton.textContent = "⏹️";
    update_status("main-prompt", "🎤 話してください...");
    if (userInputMode === "browser_asr") {
        if (!recognition) {
            console.error("Recognition not initialized.");
            return;
        }
        try {
            if (audioContext.state === "suspended") {
                await audioContext.resume();
            }
            create_temp_user_message();
            recognition.start();
        } catch (e) {
            console.warn("Recognition might be already active:", e);
        }
    } else if (userInputMode === "server_asr") {
        try {
            if (audioContext.state === "suspended") {
                await audioContext.resume();
            }
            if (!stream || !stream.active) {
                stream = await navigator.mediaDevices.getUserMedia({
                    audio: true,
                });
            }
            if (stream.getAudioTracks().some((track) => !track.enabled)) {
                stream.getAudioTracks().forEach((track) => {
                    track.enabled = true;
                });
                console.log("🎤 Mic Unmuted for recording (Server ASR)");
            }
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: "audio/webm",
            });
            audioChunks = [];
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };
            mediaRecorder.onstop = () => {
                if (silenceDetectionTimer) {
                    cancelAnimationFrame(silenceDetectionTimer);
                    silenceDetectionTimer = null;
                }
                if (audioStreamSource) {
                    audioStreamSource.disconnect();
                    audioStreamSource = null;
                }
                if (manualStop) {
                    manualStop = false;
                    return;
                }
                const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
                audioChunks = [];
                if (audioBlob.size > 1000) {
                    ws.send(audioBlob);
                    disable_input();
                    create_temp_user_message();
                    update_status("asr-server", "サーバーで音声を認識中...");
                } else {
                    remove_temp_user_message();
                    if (isContinuousMode) {
                        setTimeout(startRecording, 100);
                    } else {
                        enable_input();
                    }
                }
            };
            mediaRecorder.start();
            if (isContinuousMode) {
                audioStreamSource =
                    audioContext.createMediaStreamSource(stream);
                const analyser = audioContext.createAnalyser();
                analyser.fftSize = 512;
                audioStreamSource.connect(analyser);
                const bufferLength = analyser.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);
                let silentSince = Date.now();
                const SILENCE_THRESHOLD = 5;
                const VAD_HANGOVER_MS = 1500;
                const detectSilence = () => {
                    analyser.getByteFrequencyData(dataArray);
                    const avg =
                        dataArray.reduce((acc, val) => acc + val, 0) /
                        bufferLength;
                    if (avg < SILENCE_THRESHOLD) {
                        if (Date.now() - silentSince > VAD_HANGOVER_MS) {
                            if (
                                mediaRecorder &&
                                mediaRecorder.state === "recording"
                            ) {
                                mediaRecorder.stop();
                            }
                            return;
                        }
                    } else {
                        silentSince = Date.now();
                    }
                    silenceDetectionTimer =
                        requestAnimationFrame(detectSilence);
                };
                detectSilence();
            }
        } catch (err) {
            console.error("マイクアクセスエラー:", err);
            update_status(
                "main-prompt",
                "マイクへのアクセスが拒否されました。",
            );
            enable_input();
            micButton.classList.remove("recording");
            micButton.textContent = "🎤";
        }
    }
};
const stopRecording = () => {
    micButton.classList.remove("recording");
    micButton.textContent = isContinuousMode ? "🎙️" : "🎤";
    remove_temp_user_message();
    remove_status("main-prompt");
    if (silenceDetectionTimer) {
        cancelAnimationFrame(silenceDetectionTimer);
        silenceDetectionTimer = null;
    }
    if (audioStreamSource) {
        audioStreamSource.disconnect();
        audioStreamSource = null;
    }
    if (userInputMode === "browser_asr") {
        if (recognition) {
            recognition.abort();
        }
    } else if (userInputMode === "server_asr") {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
        }
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            stream = null;
        }
    }
};
const startWebcam = async () => {
    try {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: true,
            });
            webcamVideo.srcObject = webcamStream;
            webcamPanel.dataset.state = "on";
            webcamButton.classList.add("on");
            isWebcamOn = true;
        } else {
            update_status(
                "webcam-error",
                "カメラ機能がサポートされていません。",
                3000,
            );
        }
    } catch (err) {
        update_status(
            "webcam-error",
            "カメラへのアクセスを許可してください。",
            3000,
        );
        console.error("Webカメラアクセスエラー:", err);
    }
};
const stopWebcam = () => {
    if (webcamStream) {
        webcamStream.getTracks().forEach((track) => track.stop());
    }
    webcamStream = null;
    webcamVideo.srcObject = null;
    webcamPanel.dataset.state = "off";
    webcamButton.classList.remove("on");
    isWebcamOn = false;
};
micButton.onclick = () => {
    if (isFullAutoMode) return;
    isContinuousMode = !isContinuousMode;
    micButton.classList.toggle("continuous-mode", isContinuousMode);
    if (isContinuousMode) {
        if (!micButton.classList.contains("recording") && !isPlaying) {
            startRecording();
        }
    } else {
        if (micButton.classList.contains("recording")) {
            manualStop = true;
            stopRecording();
        }
        enable_input();
    }
};
webcamButton.onclick = () => {
    if (isWebcamOn) {
        stopWebcam();
    } else {
        startWebcam();
    }
};
scaleSlider.addEventListener("input", (e) => {
    scaleValueLabel.textContent = e.target.value;
});
sendButton.onclick = handleTextInput;
messageInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleTextInput();
});
attachButton.onclick = () => imageInput.click();
imageInput.onchange = handleImageAttachment;
