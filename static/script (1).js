document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewArea = document.getElementById('previewArea');
    const imagePreview = document.getElementById('imagePreview');
    const removeBtn = document.getElementById('removeBtn');
    const classifyBtn = document.getElementById('classifyBtn');
    const btnText = classifyBtn.querySelector('.btn-text');
    const loader = classifyBtn.querySelector('.loader');
    const resultsSection = document.getElementById('resultsSection');
    const predictionsList = document.getElementById('predictionsList');

    let currentFile = null;

    // Drag & Drop
    dropZone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', handleFiles);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles({ target: { files: files } });
    }

    function handleFiles(e) {
        if (!e.target.files.length) return;

        const file = e.target.files[0];
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        currentFile = file;
        showPreview(file);
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = () => {
            imagePreview.src = reader.result;
            dropZone.classList.add('hidden');
            previewArea.classList.remove('hidden');
            classifyBtn.disabled = false;
            resultsSection.classList.add('hidden');
        }
    }

    removeBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        dropZone.classList.remove('hidden');
        previewArea.classList.add('hidden');
        classifyBtn.disabled = true;
        resultsSection.classList.add('hidden');
    });

    // Classification
    classifyBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        setLoading(true);
        previewArea.classList.add('scanning'); // Start scanning effect

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Prediction failed');

            const data = await response.json();
            displayResults(data.predictions);
        } catch (error) {
            console.error('Error:', error);
            alert('Something went wrong. Please try again.');
        } finally {
            setLoading(false);
            previewArea.classList.remove('scanning'); // Stop scanning effect
        }
    });

    function setLoading(isLoading) {
        classifyBtn.disabled = isLoading;
        if (isLoading) {
            btnText.classList.add('hidden');
            loader.classList.remove('hidden');
        } else {
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    }

    function displayResults(predictions) {
        predictionsList.innerHTML = '';

        predictions.forEach(pred => {
            const percent = (pred.confidence * 100).toFixed(1);

            const item = document.createElement('div');
            item.className = 'prediction-item';

            item.innerHTML = `
                <div class="info">
                    <div class="species-name">${pred.species}</div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar" style="width: 0%"></div>
                    </div>
                </div>
                <div class="confidence-text">${percent}%</div>
            `;

            predictionsList.appendChild(item);

            // Animate bar
            setTimeout(() => {
                item.querySelector('.confidence-bar').style.width = `${percent}%`;
            }, 100);
        });

        resultsSection.classList.remove('hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
});
