const video = document.querySelector("#camera-video");
const canvas = document.querySelector("#landmark-canvas");
const canvasContext = canvas.getContext("2d");
const cameraButton = document.querySelector("#camera-toggle");
const cameraPlaceholder = document.querySelector("#camera-placeholder");
const cameraStatus = document.querySelector("#camera-status");
const cameraIndicator = document.querySelector("#camera-indicator");
const cameraMessage = document.querySelector("#camera-message");
const handResult = document.querySelector("#hand-result");
const speechText = document.querySelector("#speech-text");
const playButton = document.querySelector("#play-audio");
const stopButton = document.querySelector("#stop-audio");
const speechToggle = document.querySelector("#speech-toggle");
const speechMessage = document.querySelector("#speech-message");
const transcription = document.querySelector("#transcription");
const trainingLabelInput = document.querySelector("#training-label");
const captureSampleButton = document.querySelector("#capture-sample");
const trainClassifierButton = document.querySelector("#train-classifier");
const clearTrainingButton = document.querySelector("#clear-training");
const trainingCount = document.querySelector("#training-count");
const trainingPrediction = document.querySelector("#training-prediction");
const trainingMessage = document.querySelector("#training-message");

let stream;
let handLandmarker;
let animationFrame;
let isCameraRunning = false;
let lastVideoTime = -1;
let recognition;
let latestFeatures;
let trainingSamples = [];
let classifierIsTrained = false;

const minimumSamplesPerLabel = 5;
const neighborCount = 3;

const handConnections = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]
];

async function createHandLandmarker() {
  if (handLandmarker) return handLandmarker;
  const { FilesetResolver, HandLandmarker } = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm"
  );
  const files = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(files, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "CPU"
    },
    runningMode: "VIDEO",
    numHands: 2
  });
  return handLandmarker;
}

function resizeCanvas() {
  const bounds = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(bounds.width * ratio);
  canvas.height = Math.round(bounds.height * ratio);
  canvasContext.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawLandmarks(landmarks, handedness) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvasContext.clearRect(0, 0, width, height);
  latestFeatures = landmarks.length === 1 ? normalizeLandmarks(landmarks[0]) : undefined;
  captureSampleButton.disabled = !isCameraRunning || !latestFeatures;
  if (landmarks.length === 0) {
    handResult.textContent = "لم تُرصد يد بعد — قرّب يدك من الكاميرا";
    if (classifierIsTrained) trainingPrediction.textContent = "لم تُرصد يد؛ لا توجد نتيجة.";
    return;
  }

  handResult.textContent = `تم رصد ${landmarks.length === 1 ? "يد واحدة" : `${landmarks.length} يدين`} — التتبّع فقط، بلا ترجمة`;
  if (landmarks.length !== 1) {
    if (classifierIsTrained) trainingPrediction.textContent = "ضع يدًا واحدة في مجال الكاميرا للتصنيف التجريبي.";
  } else if (classifierIsTrained) {
    trainingPrediction.textContent = predictSign(latestFeatures);
  }
  for (const [handIndex, points] of landmarks.entries()) {
    const color = handIndex === 0 ? "#ffc28d" : "#8fe0bc";
    canvasContext.strokeStyle = color;
    canvasContext.lineWidth = 3;
    canvasContext.lineCap = "round";
    canvasContext.beginPath();
    for (const [start, end] of handConnections) {
      canvasContext.moveTo(points[start].x * width, points[start].y * height);
      canvasContext.lineTo(points[end].x * width, points[end].y * height);
    }
    canvasContext.stroke();
    for (const point of points) {
      canvasContext.beginPath();
      canvasContext.fillStyle = color;
      canvasContext.arc(point.x * width, point.y * height, 3.4, 0, Math.PI * 2);
      canvasContext.fill();
    }
    const label = handedness[handIndex]?.[0]?.displayName;
    if (label && points[0]) {
      canvasContext.font = '12px "IBM Plex Sans Arabic", sans-serif';
      canvasContext.fillStyle = "#fff";
      canvasContext.fillText(label === "Left" ? "يسرى" : "يمنى", points[0].x * width + 10, points[0].y * height - 8);
    }
  }
}

function normalizeLandmarks(points) {
  const wrist = points[0];
  const relativePoints = points.flatMap((point) => [
    point.x - wrist.x,
    point.y - wrist.y,
    point.z - wrist.z
  ]);
  let scale = 0;
  for (let index = 0; index < relativePoints.length; index += 3) {
    const distance = Math.hypot(
      relativePoints[index],
      relativePoints[index + 1],
      relativePoints[index + 2]
    );
    scale = Math.max(scale, distance);
  }
  if (scale === 0) return undefined;
  return relativePoints.map((coordinate) => coordinate / scale);
}

function getTrainingCounts() {
  return trainingSamples.reduce((counts, sample) => {
    counts[sample.label] = (counts[sample.label] || 0) + 1;
    return counts;
  }, {});
}

function updateTrainingStatus() {
  const counts = getTrainingCounts();
  const labels = Object.keys(counts);
  const summary = labels.length
    ? labels.map((label) => `${label}: ${counts[label]}`).join("، ")
    : "لا توجد أمثلة بعد.";
  trainingCount.textContent = `الأمثلة (${trainingSamples.length}): ${summary}`;
  trainClassifierButton.disabled = labels.length < 2
    || labels.some((label) => counts[label] < minimumSamplesPerLabel);
}

function predictSign(features) {
  const nearest = trainingSamples
    .map((sample) => ({
      label: sample.label,
      distance: Math.sqrt(sample.features.reduce((sum, value, index) => {
        const difference = value - features[index];
        return sum + difference * difference;
      }, 0))
    }))
    .sort((first, second) => first.distance - second.distance)
    .slice(0, neighborCount);
  const votes = nearest.reduce((counts, sample) => {
    counts[sample.label] = (counts[sample.label] || 0) + 1;
    return counts;
  }, {});
  const [label, voteCount] = Object.entries(votes)
    .sort((first, second) => second[1] - first[1])[0] || [];

  if (!label || voteCount / nearest.length < 2 / 3) {
    return "التصنيف غير واثق — أعد الإشارة أو اجمع أمثلة أفضل.";
  }
  return `تخمين تجريبي: ${label} (${voteCount} من أقرب ${nearest.length} أمثلة)`;
}

function detectHands() {
  if (!isCameraRunning || !handLandmarker) return;
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = handLandmarker.detectForVideo(video, performance.now());
    drawLandmarks(result.landmarks, result.handedness);
  }
  animationFrame = requestAnimationFrame(detectHands);
}

function showCameraError(message) {
  cameraMessage.textContent = message;
}

async function startCamera() {
  cameraMessage.textContent = "";
  if (!navigator.mediaDevices?.getUserMedia) {
    showCameraError("لا يمكن الوصول للكاميرا هنا. افتح الموقع عبر HTTPS أو localhost واستخدم متصفحًا حديثًا.");
    return;
  }

  cameraButton.disabled = true;
  cameraButton.textContent = "جارٍ تشغيل الكاميرا…";
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }
    });
    video.srcObject = stream;
    await video.play();
    video.classList.add("is-visible");
    cameraPlaceholder.hidden = true;
    resizeCanvas();
    isCameraRunning = true;
    cameraIndicator.classList.add("is-live");
    cameraStatus.textContent = "الكاميرا تعمل";
    cameraButton.innerHTML = '<span class="button-icon">■</span> أوقف الكاميرا';
    cameraButton.disabled = false;

    try {
      await createHandLandmarker();
      detectHands();
    } catch (error) {
      showCameraError(`عرض الكاميرا يعمل، لكن تعذّر تحميل نموذج تتبّع اليد. تحقّق من الاتصال بالإنترنت ثم أعد المحاولة. (${error.message})`);
      handResult.textContent = "تعذّر تحميل نموذج التتبّع";
    }
  } catch (error) {
    cameraButton.disabled = false;
    cameraButton.textContent = "◉ ابدأ الكاميرا";
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      showCameraError("لم يُسمح بالوصول إلى الكاميرا. امنح الموقع الإذن من إعدادات المتصفح ثم أعد المحاولة.");
    } else if (error.name === "NotFoundError") {
      showCameraError("لم نعثر على كاميرا متصلة بهذا الجهاز.");
    } else {
      showCameraError(`تعذّر تشغيل الكاميرا: ${error.message}`);
    }
  }
}

function stopCamera() {
  isCameraRunning = false;
  latestFeatures = undefined;
  captureSampleButton.disabled = true;
  cancelAnimationFrame(animationFrame);
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = undefined;
  }
  video.srcObject = null;
  video.classList.remove("is-visible");
  cameraPlaceholder.hidden = false;
  cameraIndicator.classList.remove("is-live");
  cameraStatus.textContent = "الكاميرا متوقفة";
  cameraButton.innerHTML = '<span class="button-icon">◉</span> ابدأ الكاميرا';
  canvasContext.clearRect(0, 0, canvas.width, canvas.height);
  handResult.textContent = "بانتظار تشغيل الكاميرا";
  if (classifierIsTrained) trainingPrediction.textContent = "الكاميرا متوقفة.";
}

cameraButton.addEventListener("click", () => {
  if (isCameraRunning) stopCamera();
  else startCamera();
});

captureSampleButton.addEventListener("click", () => {
  const label = trainingLabelInput.value.trim();
  if (!label) {
    trainingMessage.textContent = "اكتب اسم الإشارة أولًا.";
    trainingLabelInput.focus();
    return;
  }
  if (!latestFeatures) {
    trainingMessage.textContent = "لم تُرصد يد واحدة بوضوح. أظهر يدًا واحدة ثم حاول مجددًا.";
    return;
  }
  trainingSamples.push({ label, features: [...latestFeatures] });
  classifierIsTrained = false;
  trainingPrediction.textContent = "أُضيف المثال. أعد التدريب بعد جمع أمثلة كافية.";
  trainingMessage.textContent = "تم حفظ نقاط اليد في ذاكرة هذه الصفحة فقط.";
  updateTrainingStatus();
});

trainClassifierButton.addEventListener("click", () => {
  const counts = getTrainingCounts();
  const labels = Object.keys(counts);
  if (labels.length < 2 || labels.some((label) => counts[label] < minimumSamplesPerLabel)) {
    trainingMessage.textContent = `اجمع ${minimumSamplesPerLabel} أمثلة على الأقل لكل واحدة من إشارتين مختلفتين.`;
    return;
  }
  classifierIsTrained = true;
  trainingPrediction.textContent = "تم تفعيل المصنّف التجريبي. أظهر إشارة واحدة أمام الكاميرا.";
  trainingMessage.textContent = "المصنّف يقارن نقاط اليد بأقرب الأمثلة التي جمعتها؛ هذه ليست دقة معتمدة.";
});

clearTrainingButton.addEventListener("click", () => {
  trainingSamples = [];
  classifierIsTrained = false;
  trainingPrediction.textContent = "لم يتم تدريب المصنّف بعد.";
  trainingMessage.textContent = "تم مسح أمثلة التدريب من ذاكرة الصفحة.";
  updateTrainingStatus();
});

window.addEventListener("resize", () => {
  if (isCameraRunning) resizeCanvas();
});

playButton.addEventListener("click", () => {
  const text = speechText.value.trim();
  if (!text) {
    speechMessage.textContent = "اكتب نصًا أولًا لتشغيله.";
    speechText.focus();
    return;
  }
  if (!("speechSynthesis" in window)) {
    speechMessage.textContent = "تحويل النص إلى صوت غير مدعوم في هذا المتصفح.";
    return;
  }
  speechMessage.textContent = "";
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "ar-EG";
  utterance.onstart = () => { stopButton.disabled = false; };
  utterance.onend = () => { stopButton.disabled = true; };
  utterance.onerror = (event) => {
    stopButton.disabled = true;
    if (event.error !== "interrupted" && event.error !== "canceled") {
      speechMessage.textContent = `تعذّر تشغيل الصوت: ${event.error}`;
    }
  };
  window.speechSynthesis.speak(utterance);
});

stopButton.addEventListener("click", () => {
  window.speechSynthesis.cancel();
  stopButton.disabled = true;
});

document.querySelector("#clear-result").addEventListener("click", () => {
  speechText.value = "";
  speechMessage.textContent = "";
  transcription.textContent = "سيظهر النص هنا…";
  if (recognition) recognition.abort();
});

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.lang = "ar-EG";
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.onstart = () => {
    speechToggle.classList.add("is-listening");
    speechToggle.setAttribute("aria-label", "إيقاف الاستماع");
    speechMessage.textContent = "";
  };
  recognition.onresult = (event) => {
    const text = Array.from(event.results).map((result) => result[0].transcript).join("");
    transcription.textContent = text || "لم يتم التقاط كلام بعد…";
  };
  recognition.onerror = (event) => {
    speechMessage.textContent = event.error === "not-allowed"
      ? "لم يُسمح بالوصول إلى الميكروفون. فعّل الإذن من المتصفح."
      : `تعذّر التعرّف على الكلام: ${event.error}`;
  };
  recognition.onend = () => {
    speechToggle.classList.remove("is-listening");
    speechToggle.setAttribute("aria-label", "ابدأ الاستماع");
  };
  speechToggle.addEventListener("click", () => {
    try {
      recognition.start();
    } catch (error) {
      if (error.name === "InvalidStateError") recognition.stop();
      else speechMessage.textContent = `تعذّر بدء الاستماع: ${error.message}`;
    }
  });
} else {
  speechToggle.disabled = true;
  speechToggle.title = "التعرّف على الكلام غير مدعوم في هذا المتصفح";
  speechMessage.textContent = "ميزة التعرّف على الكلام غير مدعومة في هذا المتصفح.";
}

updateTrainingStatus();
