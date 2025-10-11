const startCameraBtn = document.getElementById("startCameraBtn")
const video = document.getElementById("video")
const captureBtn = document.getElementById("captureBtn")
const canvas = document.getElementById("canvas")
const cameraForm = document.getElementById("cameraForm")
const capturedPhotoInput = document.getElementById("capturedPhoto")
const uploadForm = document.getElementById("uploadForm")
const fileInput = document.getElementById("fileInput")
const photoPreview = document.getElementById("photoPreview")
const loadingSection = document.getElementById("loadingSection")
const ajaxResults = document.getElementById("ajaxResults")

// New feedback elements
const feedbackSection = document.getElementById("feedbackSection")
const likeBtn = document.getElementById("likeBtn")
const dislikeBtn = document.getElementById("dislikeBtn")
const dislikeFeedbackArea = document.getElementById("dislikeFeedbackArea")
const dislikeReason = document.getElementById("dislikeReason")
const submitDislikeFeedbackBtn = document.getElementById("submitDislikeFeedbackBtn")
const feedbackMessage = document.getElementById("feedbackMessage")

let stream
let beautyTips // Declare the beautyTips variable

// File input preview
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0]
  if (file) {
    const reader = new FileReader()
    reader.onload = (e) => {
      photoPreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`
      photoPreview.style.display = "block"
    }
    reader.readAsDataURL(file)
  }
})

// AJAX form submission
uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault()
  const formData = new FormData(uploadForm)
  showLoading()
  try {
    const response = await fetch("/upload/", {
      method: "POST",
      body: formData,
      headers: { "X-Requested-With": "XMLHttpRequest" },
    })
    const data = await response.json()
    hideLoading()
    if (data.error) {
      alert("Error: " + data.error)
      resetForm() // Re-show buttons on error
      return
    }
    showResults(data)
  } catch (err) {
    hideLoading()
    alert("Something went wrong: " + err.message)
    resetForm() // Re-show buttons on error
  }
})

function showLoading() {
  loadingSection.style.display = "block"
  document.querySelector(".upload-card").style.display = "none"
  document.querySelector(".camera-card").style.display = "none"
  document.querySelector(".divider").style.display = "none"
}

function hideLoading() {
  loadingSection.style.display = "none"
}

function createProbBars(containerId, labels, probs, colors) {
  const container = document.getElementById(containerId)
  container.innerHTML = ""
  probs.forEach((p, i) => {
    const barWrapper = document.createElement("div")
    barWrapper.classList.add("prob-bar-wrapper")

    const barInfo = document.createElement("div")
    barInfo.classList.add("prob-bar-info")

    const label = document.createElement("span")
    label.classList.add("prob-bar-label")
    label.textContent = labels[i]

    const value = document.createElement("span")
    value.classList.add("prob-bar-value")
    value.textContent = (p * 100).toFixed(1) + "%"

    barInfo.appendChild(label)
    barInfo.appendChild(value)

    const barContainer = document.createElement("div")
    barContainer.classList.add("prob-bar")

    const fill = document.createElement("div")
    fill.classList.add("prob-bar-fill")
    fill.style.width = p * 100 + "%"
    fill.style.backgroundColor = colors[i] || colors[0]

    barContainer.appendChild(fill)
    barWrapper.appendChild(barInfo)
    barWrapper.appendChild(barContainer)
    container.appendChild(barWrapper)
  })
}

function showResults(data) {
  document.getElementById("skinType").textContent = data.skin_type || data.predicted_type || "Unknown"

  document.getElementById("leftEyeColor").textContent = data.left_eye_color ? `Left Eye: ${data.left_eye_color}` : ""
  document.getElementById("rightEyeColor").textContent = data.right_eye_color
    ? `Right Eye: ${data.right_eye_color}`
    : ""

  const acneLevelElement = document.getElementById("acneLevel")
  const acneConfidenceElement = document.getElementById("acneConfidence")

  acneLevelElement.textContent = data.acne_pred || "Unknown"

  if (data.acne_confidence) {
    const confidencePercent = data.acne_confidence * 100
    acneConfidenceElement.textContent = `Confidence: ${confidencePercent.toFixed(1)}%`
  } else {
    acneConfidenceElement.textContent = ""
  }

  if (data.cropped_face) {
    const croppedFaceSection = document.getElementById("croppedFaceSection")
    const croppedFaceImage = document.getElementById("croppedFaceImage")
    croppedFaceImage.src = data.cropped_face
    croppedFaceSection.style.display = "block"

    if (data.segmentation_overlay) {
      const segmentationSection = document.getElementById("segmentationSection")
      document.getElementById("segmentedImage").src = data.segmentation_overlay
      segmentationSection.style.display = "block"
    } else {
      document.getElementById("segmentationSection").style.display = "none"
    }

    if (data.yolo_boxes && data.yolo_boxes.length > 0) {
      const yoloDetectionSection = document.getElementById("yoloDetectionSection")
      const yoloCanvas = document.getElementById("yoloCanvas")
      const ctx = yoloCanvas.getContext("2d")
      const image = new Image()

      image.onload = () => {
        yoloCanvas.width = image.width
        yoloCanvas.height = image.height

        ctx.drawImage(image, 0, 0)

        data.yolo_boxes.forEach((box) => {
          const [x1, y1, x2, y2] = box.bbox
          const label = box.label
          const confidence = box.confidence

          ctx.strokeStyle = "#16a34a"
          ctx.lineWidth = 3
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

          ctx.fillStyle = "rgba(22, 163, 74, 0.8)"
          ctx.fillRect(x1, y1 - 25, (label.length + 10) * 8, 25)

          ctx.fillStyle = "#ffffff"
          ctx.font = "bold 14px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto"
          ctx.fillText(`${label} ${(confidence * 100).toFixed(1)}%`, x1 + 5, y1 - 8)
        })
      }
      image.crossOrigin = "anonymous"
      image.src = data.cropped_face
      yoloDetectionSection.style.display = "block"
    } else {
      document.getElementById("yoloDetectionSection").style.display = "none"
    }
  }

  const skinTypeLabels = ["Dry", "Normal", "Oily"]
  const typeColors = ["#2563eb", "#7c3aed", "#f59e0b"]

  if (data.type_probs && data.type_probs.length === skinTypeLabels.length) {
    document.getElementById("typeProbsSection").style.display = "block"
    createProbBars("typeProbsBars", skinTypeLabels, data.type_probs, typeColors)
  } else {
    document.getElementById("typeProbsSection").style.display = "none"
  }

  if (data.acne_pred && data.acne_confidence) {
    const acneAnalysisSection = document.getElementById("acneAnalysisSection")
    const acneConfidenceFill = document.getElementById("acneConfidenceFill")
    const acneConfidenceText = document.getElementById("acneConfidenceText")

    const confidencePercent = data.acne_confidence * 100

    acneConfidenceFill.style.width = confidencePercent + "%"
    acneConfidenceText.textContent = `${data.acne_pred} - ${confidencePercent.toFixed(1)}% Confidence`

    acneAnalysisSection.style.display = "block"
  } else {
    document.getElementById("acneAnalysisSection").style.display = "none"
  }

  generateRecommendation(data)

  ajaxResults.style.display = "block"
  ajaxResults.classList.add("active")

  feedbackSection.style.display = "block"
  feedbackMessage.style.display = "none"
  dislikeFeedbackArea.style.display = "none"
  dislikeReason.value = ""
  likeBtn.disabled = false
  dislikeBtn.disabled = false
}

function generateRecommendation(data) {
  // Check if beautyTips is available from tips.js
  if (typeof beautyTips === "undefined") {
    console.error("beautyTips not loaded from tips.js")
    document.getElementById("recommendationText").textContent =
      "Unable to load recommendations. Please refresh the page."
    return
  }

  const language = "en" // Default to English, can be made dynamic
  const tips = []

  // Get skin type tips
  if (data.skin_type) {
    const skinType = data.skin_type.toLowerCase()
    if (beautyTips[language].skinType[skinType]) {
      tips.push(...beautyTips[language].skinType[skinType])
    }
  }

  // Get eye color tips
  if (data.left_eye_color) {
    const eyeColor = data.left_eye_color.toLowerCase().replace(" ", "")
    if (beautyTips[language].eyeColor[eyeColor]) {
      tips.push(...beautyTips[language].eyeColor[eyeColor])
    }
  }

  // Get acne level tips
  if (data.acne_pred) {
    const acneLevel = data.acne_pred.toLowerCase()
    if (beautyTips[language].acneLevel[acneLevel]) {
      tips.push(...beautyTips[language].acneLevel[acneLevel])
    }
  }

  // If we have tips, display them
  if (tips.length > 0) {
    const recommendation = tips.join(" ")
    document.getElementById("recommendationText").textContent = recommendation
  } else {
    // Fallback recommendation
    document.getElementById("recommendationText").textContent =
      "Based on your analysis, we recommend consulting with a beauty professional for personalized advice."
  }
}

function resetForm() {
  stopCamera() // Ensure camera is stopped and capture button is disabled
  ajaxResults.style.display = "none"
  ajaxResults.classList.remove("active")
  document.querySelector(".upload-card").style.display = "block"
  document.querySelector(".camera-card").style.display = "block"
  document.querySelector(".divider").style.display = "block"
  uploadForm.reset()
  photoPreview.style.display = "none"
  photoPreview.innerHTML = ""
  document.getElementById("typeProbsSection").style.display = "none"
  document.getElementById("croppedFaceSection").style.display = "none"
  document.getElementById("yoloDetectionSection").style.display = "none"
  document.getElementById("acneAnalysisSection").style.display = "none"
  document.getElementById("segmentationSection").style.display = "none"

  // Reset feedback section
  feedbackSection.style.display = "none"
  feedbackMessage.style.display = "none"
  dislikeFeedbackArea.style.display = "none"
  dislikeReason.value = ""
  likeBtn.disabled = false
  dislikeBtn.disabled = false
}

// Camera functionality
startCameraBtn.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user",
      },
    })
    video.srcObject = stream
    video.parentElement.style.display = "block"
    captureBtn.disabled = false
    startCameraBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Camera'
    startCameraBtn.onclick = stopCamera
  } catch (err) {
    alert("Could not access camera: " + err.message)
    resetForm() // Re-show buttons on camera access error
  }
})

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop())
    video.parentElement.style.display = "none"
    captureBtn.disabled = true
    startCameraBtn.innerHTML = '<i class="fas fa-video"></i> Start Camera'
    startCameraBtn.onclick = () => startCameraBtn.click() // Re-attach original click handler
  }
}

captureBtn.addEventListener("click", async () => {
  const context = canvas.getContext("2d")
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  context.drawImage(video, 0, 0, canvas.width, canvas.height)

  const dataUrl = canvas.toDataURL("image/jpeg", 0.8)
  capturedPhotoInput.value = dataUrl

  showLoading()
  captureBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...'
  captureBtn.disabled = true

  const formData = new FormData(cameraForm)

  try {
    const response = await fetch("/upload/", {
      method: "POST",
      body: formData,
      headers: { "X-Requested-With": "XMLHttpRequest" },
    })

    const data = await response.json()
    hideLoading()

    if (data.error) {
      alert("Error: " + data.error)
      resetForm() // Re-show buttons on error
      return
    }

    showResults(data)
  } catch (err) {
    hideLoading()
    alert("Something went wrong: " + err.message)
    resetForm() // Re-show buttons on error
  }

  stopCamera()
  captureBtn.innerHTML = '<i class="fas fa-camera-retro"></i> Capture Photo'
  captureBtn.disabled = false
})

// Function to send feedback to the backend
async function sendFeedback(feedbackType, reason = "") {
  feedbackMessage.style.display = "none" // Hide previous messages
  likeBtn.disabled = true
  dislikeBtn.disabled = true
  submitDislikeFeedbackBtn.disabled = true // Disable submit button for dislike

  try {
    const response = await fetch("/submit-feedback/", {
      // Ensure this URL matches your Django urls.py
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // Since your Django view is @csrf_exempt, we don't strictly need CSRF token here.
        // If you remove @csrf_exempt later, you'll need to add:
        // "X-CSRFToken": getCookie('csrftoken'),
      },
      body: JSON.stringify({
        feedback_type: feedbackType,
        dislike_reason: reason,
      }),
    })

    const data = await response.json()

    if (response.ok) {
      feedbackMessage.textContent = data.message || "Thank you for your feedback!"
      feedbackMessage.style.color = "#15803d"
      feedbackMessage.style.backgroundColor = "#f0fdf4"
      feedbackMessage.style.border = "1px solid #bbf7d0"
    } else {
      feedbackMessage.textContent = data.error || "Failed to submit feedback."
      feedbackMessage.style.color = "#b91c1c"
      feedbackMessage.style.backgroundColor = "#fef2f2"
      feedbackMessage.style.border = "1px solid #fecaca"
      // Re-enable buttons if submission failed
      likeBtn.disabled = false
      dislikeBtn.disabled = false
      if (feedbackType === "dislike") {
        submitDislikeFeedbackBtn.disabled = false
      }
    }
  } catch (error) {
    feedbackMessage.textContent = "Network error: Could not submit feedback."
    feedbackMessage.style.color = "#b91c1c"
    feedbackMessage.style.backgroundColor = "#fef2f2"
    feedbackMessage.style.border = "1px solid #fecaca"
    // Re-enable buttons on network error
    likeBtn.disabled = false
    dislikeBtn.disabled = false
    if (feedbackType === "dislike") {
      submitDislikeFeedbackBtn.disabled = false
    }
  } finally {
    feedbackMessage.style.display = "block"
    feedbackMessage.style.padding = "12px"
    feedbackMessage.style.borderRadius = "6px"
    feedbackMessage.style.fontWeight = "500"
    dislikeFeedbackArea.style.display = "none" // Hide dislike area after submission attempt
    dislikeReason.value = "" // Clear reason
  }
}

// Feedback functionality event listeners
likeBtn.addEventListener("click", () => {
  sendFeedback("like")
})

dislikeBtn.addEventListener("click", () => {
  dislikeFeedbackArea.style.display = "block"
  feedbackMessage.style.display = "none" // Hide any previous message
  likeBtn.disabled = true // Disable like button once dislike is chosen
  dislikeBtn.disabled = true // Disable dislike button to prevent multiple clicks
  submitDislikeFeedbackBtn.disabled = false // Enable submit button for dislike
})

submitDislikeFeedbackBtn.addEventListener("click", () => {
  const reason = dislikeReason.value.trim()
  if (reason) {
    sendFeedback("dislike", reason)
  } else {
    alert("Please provide a reason for disliking the analysis.")
  }
})
