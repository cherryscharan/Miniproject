document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewArea = document.getElementById('previewArea');
    const imagePreview = document.getElementById('imagePreview');
    const removeBtn = document.getElementById('removeBtn');
    const classifyBtn = document.getElementById('classifyBtn');
    const resultsSection = document.getElementById('resultsSection');
    const predictionsList = document.getElementById('predictionsList');
    const loader = document.querySelector('.loader');
    const btnText = document.querySelector('.btn-text');

    let currentFile = null;

    // --- Event Listeners ---

    // Click on drop zone to trigger file input
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle file selection via input
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and Drop events
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Remove image
    removeBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = ''; // Reset input
        previewArea.classList.add('hidden');
        dropZone.classList.remove('hidden');
        classifyBtn.disabled = true;
        resultsSection.classList.add('hidden');
    });

    // Classify button
    classifyBtn.addEventListener('click', async () => {
        if (!currentFile) return;
        
        // UI Loading State
        classifyBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        try {
            const formData = new FormData();
            formData.append('file', currentFile);

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server Error: ${response.statusText}`);
            }

            const data = await response.json();
            displayResults(data.predictions);

        } catch (error) {
            console.error(error);
            alert('An error occurred during classification. Please try again.');
        } finally {
            // Reset UI State
            classifyBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    });


    // --- Helper Functions ---

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }

        currentFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewArea.classList.remove('hidden');
            classifyBtn.disabled = false;
            resultsSection.classList.add('hidden'); // Hide previous results if any
        };
        reader.readAsDataURL(file);
    }

    function displayResults(predictions) {
        predictionsList.innerHTML = '';
        
        if (predictions.length === 0) {
            predictionsList.innerHTML = '<div class="result-item">No species detected.</div>';
            return;
        }

        predictions.forEach(pred => {
            const confidencePercent = (pred.confidence * 100).toFixed(1);
            
            const item = document.createElement('div');
            item.className = 'result-item';
            
            // Result content
            item.innerHTML = `
                <div class="species-name">${pred.species}</div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar" style="width: ${confidencePercent}%"></div>
                </div>
                <div class="confidence-text">${confidencePercent}% Interest</div>
            `;
            
            predictionsList.appendChild(item);
        });

        resultsSection.classList.remove('hidden');
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
});
