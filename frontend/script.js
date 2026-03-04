document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('textInput');
    const micBtn = document.getElementById('micBtn');
    const predictBtn = document.getElementById('predictBtn');
    const spinner = document.getElementById('spinner');
    const btnText = predictBtn.querySelector('.btn-text');
    const micStatus = document.getElementById('micStatus');
    const resultsSection = document.getElementById('resultsSection');

    const finalCard = document.getElementById('finalCard');
    const finalLang = document.getElementById('finalLang');
    const finalProb = document.getElementById('finalProb');
    const finalConfLabel = document.getElementById('finalConfLabel');
    const romanizedBadge = document.getElementById('romanizedBadge');
    const transScriptName = document.getElementById('transScriptName');
    const l1Script = document.getElementById('l1Script');
    const l1Candidates = document.getElementById('l1Candidates');
    const l2Bars = document.getElementById('l2Bars');
    const l3Bars = document.getElementById('l3Bars');

    // -----------------------------------------------------------------------
    // Speech Recognition
    // -----------------------------------------------------------------------
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;
    let isRecording = false;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.lang = 'en-IN';
        recognition.interimResults = true;
        recognition.continuous = false;
        recognition.onstart = () => { isRecording = true; micBtn.classList.add('recording'); micStatus.classList.remove('hidden'); };
        recognition.onresult = (event) => {
            let finalTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) finalTranscript += event.results[i][0].transcript;
            }
            if (finalTranscript) textInput.value = textInput.value ? textInput.value + ' ' + finalTranscript : finalTranscript;
        };
        recognition.onerror = () => stopRecording();
        recognition.onend = () => { stopRecording(); if (textInput.value.trim().length > 0) triggerPrediction(); };
    } else {
        micBtn.style.display = 'none';
    }

    micBtn.addEventListener('click', () => {
        if (!recognition) return;
        if (isRecording) recognition.stop();
        else { textInput.value = ''; recognition.start(); }
    });

    function stopRecording() { isRecording = false; micBtn.classList.remove('recording'); micStatus.classList.add('hidden'); }

    // -----------------------------------------------------------------------
    // Example chip click — insert text + auto-predict
    // -----------------------------------------------------------------------
    document.querySelectorAll('.example-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const sentence = chip.getAttribute('data-text');
            if (!sentence) return;

            // Highlight the clicked chip briefly
            document.querySelectorAll('.example-chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');

            // Fill textarea
            textInput.value = sentence;
            // Scroll to textarea smoothly
            textInput.scrollIntoView({ behavior: 'smooth', block: 'center' });

            // Auto-trigger prediction
            triggerPrediction();
        });
    });

    // -----------------------------------------------------------------------
    // Predict
    // -----------------------------------------------------------------------
    predictBtn.addEventListener('click', triggerPrediction);

    async function triggerPrediction() {
        const text = textInput.value.trim();
        if (!text) return;

        predictBtn.disabled = true;
        btnText.classList.add('hidden');
        spinner.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            if (!response.ok) throw new Error('Server error');
            displayResults(await response.json());
        } catch (error) {
            console.error('Prediction failed:', error);
            alert("Failed to connect to the backend. Make sure the server is running.");
        } finally {
            predictBtn.disabled = false;
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
        }
    }

    // -----------------------------------------------------------------------
    // Display results
    // -----------------------------------------------------------------------
    function displayResults(data) {
        // --- Final Prediction ---
        if (data.final_prediction) {
            finalLang.textContent = data.final_prediction;
            const pct = (data.final_confidence * 100).toFixed(1);
            finalProb.textContent = pct + '%';

            finalConfLabel.className = 'conf-badge';
            const conf = data.final_confidence;
            if (conf >= 0.8) {
                finalConfLabel.textContent = 'High Confidence';
                finalConfLabel.classList.add('conf-high');
            } else if (conf >= 0.4) {
                finalConfLabel.textContent = 'Medium Confidence';
                finalConfLabel.classList.add('conf-med');
            } else {
                finalConfLabel.textContent = 'Low Confidence';
                finalConfLabel.classList.add('conf-low');
            }
        } else {
            finalLang.textContent = 'Inconclusive';
            finalProb.textContent = '--';
            finalConfLabel.className = 'conf-badge conf-none';
            finalConfLabel.textContent = 'Unable to determine';
        }

        // --- Romanized-mode badge ---
        if (data.romanized_mode && data.used_transliteration) {
            transScriptName.textContent = data.used_transliteration;
            romanizedBadge.classList.remove('hidden');
        } else {
            romanizedBadge.classList.add('hidden');
        }

        // --- Layer 1 ---
        l1Script.textContent = data.layer1_script || '--';
        l1Candidates.textContent = (data.layer1_candidates || []).join(', ') || '--';

        // --- Layer 2 & 3 Bars ---
        renderBars(l2Bars, data.layer2_top3, 'blue');
        renderBars(l3Bars, data.layer3_top3, 'violet');

        resultsSection.classList.remove('hidden');
        document.querySelectorAll('.pop-in').forEach(card => {
            card.style.animation = 'none';
            card.offsetHeight;
            card.style.animation = null;
        });
    }

    function renderBars(container, items, colorClass) {
        container.innerHTML = '';
        if (!items || items.length === 0) {
            container.innerHTML = '<div class="bar-labels"><span style="color:var(--text-muted)">No data</span></div>';
            return;
        }
        items.forEach(([name, prob], index) => {
            const pct = (prob * 100).toFixed(1);
            const html = '<div class="bar-row">' +
                '<div class="bar-labels"><span>' + name + '</span><span>' + pct + '%</span></div>' +
                '<div class="bar-bg"><div class="bar-fill ' + colorClass + '" style="width:0%"></div></div>' +
                '</div>';
            container.insertAdjacentHTML('beforeend', html);
            setTimeout(() => {
                const fills = container.querySelectorAll('.bar-fill');
                if (fills[index]) fills[index].style.width = pct + '%';
            }, 80);
        });
    }

});
