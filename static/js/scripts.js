
        const startCameraBtn = document.getElementById('startCameraBtn');
        const video = document.getElementById('video');
        const captureBtn = document.getElementById('captureBtn');
        const canvas = document.getElementById('canvas');
        const cameraForm = document.getElementById('cameraForm');
        const capturedPhotoInput = document.getElementById('capturedPhoto');
        const uploadForm = document.getElementById('uploadForm');
        const fileInput = document.getElementById('fileInput');
        const photoPreview = document.getElementById('photoPreview');
        const loadingSection = document.getElementById('loadingSection');
        const ajaxResults = document.getElementById('ajaxResults');
        let stream;

        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    photoPreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
                    photoPreview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });

        uploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData(uploadForm);
            showLoading();
            try {
                const response = await fetch("/upload/", {
                    method: "POST",
                    body: formData,
                    headers: {'X-Requested-With': 'XMLHttpRequest'}
                });
                const data = await response.json();
                hideLoading();
                if (data.error) {
                    alert("Error: " + data.error);
                    return;
                }
                showResults(data);
            } catch (err) {
                hideLoading();
                alert("Something went wrong: " + err.message);
            }
        });

        function showLoading() {
            loadingSection.style.display = 'block';
            document.querySelector('.upload-section').style.display = 'none';
            document.querySelector('.camera-section').style.display = 'none';
            document.querySelector('.divider').style.display = 'none';
        }

        function hideLoading() {
            loadingSection.style.display = 'none';
        }

        function createProbBars(containerId, labels, probs, colors) {
            const container = document.getElementById(containerId);
            container.innerHTML = '';
            probs.forEach((p, i) => {
                const barWrapper = document.createElement('div');
                barWrapper.classList.add('prob-bar-wrapper');
                const barInfo = document.createElement('div');
                barInfo.classList.add('prob-bar-info');
                const label = document.createElement('span');
                label.classList.add('prob-bar-label');
                label.textContent = labels[i];
                const value = document.createElement('span');
                value.classList.add('prob-bar-value');
                value.textContent = (p * 100).toFixed(1) + '%';
                barInfo.appendChild(label);
                barInfo.appendChild(value);
                const barContainer = document.createElement('div');
                barContainer.classList.add('prob-bar');
                const fill = document.createElement('div');
                fill.classList.add('prob-bar-fill');
                fill.style.width = (p * 100) + '%';
                fill.style.backgroundColor = colors[i] || colors[0];
                barContainer.appendChild(fill);
                barWrapper.appendChild(barInfo);
                barWrapper.appendChild(barContainer);
                container.appendChild(barWrapper);
            });
        }

        function showResults(data) {
    document.getElementById("skinType").textContent = data.skin_type || data.predicted_type;
    document.getElementById("skinDefect").textContent = data.skin_defect || data.predicted_defect;

    if (data.cropped_face) {
        const croppedFaceSection = document.getElementById('croppedFaceSection');
        const croppedFaceImage = document.getElementById('croppedFaceImage');
        croppedFaceImage.src = data.cropped_face;
        croppedFaceSection.style.display = 'block';

        // Show YOLO detection if available
        if (data.yolo_boxes && data.yolo_boxes.length > 0) {
            const yoloDetectionSection = document.getElementById('yoloDetectionSection');
            const yoloCanvas = document.getElementById('yoloCanvas');
            const ctx = yoloCanvas.getContext('2d');
            const image = new Image();

            image.onload = function () {
                // Set canvas size to match image
                yoloCanvas.width = image.width;
                yoloCanvas.height = image.height;
                
                // Draw the image
                ctx.drawImage(image, 0, 0);

                // Draw bounding boxes
                data.yolo_boxes.forEach(box => {
                    const [x1, y1, x2, y2] = box.bbox;
                    const label = box.label;
                    const confidence = box.confidence;

                    // Draw bounding box
                    ctx.strokeStyle = '#00ff00';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                    // Draw semi-transparent background for text
                    ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
                    ctx.fillRect(x1, y1 - 25, (label.length + 10) * 8, 25);

                    // Draw label text
                    ctx.fillStyle = '#000000';
                    ctx.font = 'bold 14px Arial';
                    ctx.fillText(`${label} ${(confidence * 100).toFixed(1)}%`, x1 + 5, y1 - 8);
                });
            };
            image.crossOrigin = "anonymous";
            image.src = data.cropped_face;
            yoloDetectionSection.style.display = 'block';
        } else {
            document.getElementById('yoloDetectionSection').style.display = 'none';
        }
    }

    const skinDefectLabels = ['Acne', 'Redness', 'Bags', 'None'];
    const skinTypeLabels = ['Dry', 'Normal', 'Oily'];
    const defectColors = ['#ff6b6b', '#ff8a65', '#ffb74d', '#81c784'];
    const typeColors = ['#64b5f6', '#4fc3f7', '#4dd0e1'];

    if (data.type_probs && data.type_probs.length === skinTypeLabels.length) {
        document.getElementById('typeProbsSection').style.display = 'block';
        createProbBars('typeProbsBars', skinTypeLabels, data.type_probs, typeColors);
    } else {
        document.getElementById('typeProbsSection').style.display = 'none';
    }

    if (data.defect_probs && data.defect_probs.length === skinDefectLabels.length) {
        document.getElementById('defectProbsSection').style.display = 'block';
        createProbBars('defectProbsBars', skinDefectLabels, data.defect_probs, defectColors);
    } else {
        document.getElementById('defectProbsSection').style.display = 'none';
    }

    // Show eye colors as simple text
    document.getElementById('leftEyeColor').textContent = data.left_eye_color ? `Left Eye: ${data.left_eye_color}` : '';
    document.getElementById('rightEyeColor').textContent = data.right_eye_color ? `Right Eye: ${data.right_eye_color}` : '';

    ajaxResults.style.display = 'block';
    ajaxResults.classList.add('active');
}


        function resetForm() {
            ajaxResults.style.display = 'none';
            ajaxResults.classList.remove('active');
            document.querySelector('.upload-section').style.display = 'block';
            document.querySelector('.camera-section').style.display = 'block';
            document.querySelector('.divider').style.display = 'block';
            uploadForm.reset();
            photoPreview.style.display = 'none';
            photoPreview.innerHTML = '';
            document.getElementById('defectProbsSection').style.display = 'none';
            document.getElementById('typeProbsSection').style.display = 'none';
            document.getElementById('croppedFaceSection').style.display = 'none';
            document.getElementById('yoloDetectionSection').style.display = 'none';
        }

        // Camera functionality
        startCameraBtn.addEventListener('click', async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    } 
                });
                video.srcObject = stream;
                video.parentElement.style.display = 'block';
                captureBtn.disabled = false;
                startCameraBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Camera';
                startCameraBtn.onclick = stopCamera;
            } catch (err) {
                alert('Could not access camera: ' + err.message);
            }
        });

        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                video.parentElement.style.display = 'none';
                captureBtn.disabled = true;
                startCameraBtn.innerHTML = '<i class="fas fa-video"></i> Start Camera';
                startCameraBtn.onclick = () => startCameraBtn.click();
            }
        }

        captureBtn.addEventListener('click', async () => {
            const context = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
            capturedPhotoInput.value = dataUrl;  // includes 'data:image/jpeg;base64,...'

            
            showLoading();
            captureBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            captureBtn.disabled = true;
            
            const formData = new FormData(cameraForm);
            
            try {
                const response = await fetch("/upload/", {
                    method: "POST",
                    body: formData,
                    headers: {'X-Requested-With': 'XMLHttpRequest'}
                });

                const data = await response.json();
                hideLoading();

                if (data.error) {
                    alert("Error: " + data.error);
                    return;
                }

                showResults(data);
                
            } catch (err) {
                hideLoading();
                alert("Something went wrong: " + err.message);
            }
            
            stopCamera();
            captureBtn.innerHTML = '<i class="fas fa-camera-retro"></i> Capture Photo';
            captureBtn.disabled = false;
        });
    