let mediaRecorder = null;
let recordedChunks = [];
let stream = null;
let currentMetrics = null;

const exerciseSelect = document.getElementById('exercise-select');
const webcamBtn = document.getElementById('webcam-btn');
const stopBtn = document.getElementById('stop-btn');
const videoUpload = document.getElementById('video-upload');
const canvas = document.getElementById('video-canvas');
const ctx = canvas.getContext('2d');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('results-section');
const getAiBtn = document.getElementById('get-ai-btn');
const repsCount = document.getElementById('reps-count');
const stageValue = document.getElementById('stage-value');
const feedbackValue = document.getElementById('feedback-value');

webcamBtn.addEventListener('click', startWebcam);
stopBtn.addEventListener('click', stopWebcam);

async function startWebcam() {
    try {
        resultsSection.style.display = 'none';
        recordedChunks = [];
        
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 }, 
            audio: false 
        });
        
        const video = document.createElement('video');
        video.srcObject = stream;
        video.play();
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            const drawFrame = () => {
                if (stream && stream.active) {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    requestAnimationFrame(drawFrame);
                }
            };
            drawFrame();
        };
        
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) recordedChunks.push(event.data);
        };
        mediaRecorder.start();
        
        webcamBtn.disabled = true;
        stopBtn.disabled = false;
        exerciseSelect.disabled = true;
        repsCount.textContent = '0';
        stageValue.textContent = 'Recording...';
        feedbackValue.textContent = 'Press Stop when done';
        
    } catch (error) {
        alert('Failed to access webcam. Please allow camera permissions.');
    }
}

async function stopWebcam() {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
    
    mediaRecorder.stop();
    
    mediaRecorder.onstop = async () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        
        webcamBtn.disabled = false;
        stopBtn.disabled = true;
        exerciseSelect.disabled = false;
        
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        uploadVideo(blob, 'webcam-recording.webm');
    };
}

videoUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    uploadVideo(file, file.name);
    videoUpload.value = '';
});

async function uploadVideo(file, filename) {
    const exercise = exerciseSelect.value;
    const formData = new FormData();
    formData.append('file', file, filename);
    
    loading.style.display = 'block';
    loading.querySelector('p').textContent = 'Uploading...';
    resultsSection.style.display = 'none';
    webcamBtn.disabled = true;
    exerciseSelect.disabled = true;
    
    try {
        const response = await fetch(`https://ml-model-md0x.onrender.com/api/upload-video?exercise_type=${exercise}`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success && data.job_id) {
            console.log('Upload successful, job ID:', data.job_id); // Debug log
            pollStatus(data.job_id);
        } else {
            throw new Error(data.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Failed to upload video: ' + error.message);
        loading.style.display = 'none';
        webcamBtn.disabled = false;
        exerciseSelect.disabled = false;
    }
}

function pollStatus(jobId) {
    const checkStatus = setInterval(async () => {
        try {
            const response = await fetch(`https://ml-model-md0x.onrender.com/api/video-status/${jobId}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            console.log('Status:', data); // Debug log
            
            if (data.status === 'processing') {
                loading.querySelector('p').textContent = `Processing... ${data.progress || 0}%`;
            } else if (data.status === 'completed') {
                clearInterval(checkStatus);
                loading.style.display = 'none';
                repsCount.textContent = data.reps || 0;
                stageValue.textContent = '-';
                feedbackValue.textContent = 'Completed';
                if (data.metrics) {
                    currentMetrics = data.metrics;
                    displayMetrics(data.metrics);
                }
            } else if (data.status === 'failed') {
                clearInterval(checkStatus);
                loading.style.display = 'none';
                alert('Processing failed: ' + (data.error || 'Unknown error'));
                webcamBtn.disabled = false;
                exerciseSelect.disabled = false;
            }
        } catch (err) {
            console.error('Status check error:', err);
            clearInterval(checkStatus);
            loading.style.display = 'none';
            alert('Failed to check processing status');
            webcamBtn.disabled = false;
            exerciseSelect.disabled = false;
        }
    }, 2000);
}

function displayMetrics(metrics) {
    if (!metrics || Object.keys(metrics).length === 0) {
        alert('No metrics available');
        return;
    }
    
    resultsSection.style.display = 'block';
    document.getElementById('overall-score').textContent = metrics.overall_score || '-';
    
    const score = metrics.overall_score || 0;
    let rating = 'KEEP PRACTICING';
    if (score >= 90) rating = '⭐⭐⭐⭐⭐ EXCELLENT';
    else if (score >= 75) rating = '⭐⭐⭐⭐ VERY GOOD';
    else if (score >= 60) rating = '⭐⭐⭐ GOOD';
    else if (score >= 45) rating = '⭐⭐ FAIR';
    else if (score >= 30) rating = '⭐ NEEDS IMPROVEMENT';
    document.getElementById('rating').textContent = rating;
    
    const goodReps = metrics.details?.good_reps || metrics.good_reps || 0;
    const badReps = metrics.details?.bad_reps || metrics.bad_reps || 0;
    document.getElementById('total-reps').textContent = goodReps + badReps;
    document.getElementById('good-reps').textContent = goodReps;
    document.getElementById('bad-reps').textContent = badReps;
    
    const scoresList = document.getElementById('scores-list');
    scoresList.innerHTML = '';
    
    const allScores = {...metrics.details, ...metrics};
    for (const [key, value] of Object.entries(allScores)) {
        if (typeof value === 'number' && !['good_reps', 'bad_reps', 'overall_score'].includes(key)) {
            const div = document.createElement('div');
            div.innerHTML = `<strong>${formatKey(key)}:</strong> ${value}/100`;
            scoresList.appendChild(div);
        }
    }
}

function formatKey(key) {
    return key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

getAiBtn.addEventListener('click', async () => {
    if (!currentMetrics) {
        alert('No metrics available. Please complete a workout first.');
        return;
    }
    
    getAiBtn.disabled = true;
    getAiBtn.textContent = 'Loading...';
    
    try {
        const response = await fetch('https://ml-model-md0x.onrender.com/api/ai-feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentMetrics)
        });
        
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('ai-feedback').textContent = data.feedback;
        } else {
            document.getElementById('ai-feedback').textContent = 'Failed to get AI feedback: ' + data.error;
        }
    } catch (error) {
        console.error('AI feedback error:', error);
        document.getElementById('ai-feedback').textContent = 'Failed to get AI feedback. Please try again.';
    } finally {
        getAiBtn.disabled = false;
        getAiBtn.textContent = 'Refresh AI Feedback';
    }
});
