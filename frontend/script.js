const BACKEND_URL = "http://localhost:8000";
let currentStyle = null;
let trainedStyles = new Set(["real"]); // Start with real as default
let selectedStyles = ["real"]; // Keeps style selection order
const allStyles = ["real", "cartoon", "sketch", "edge", "blur"];
let correctPredictions = 0;
let totalPredictions = 0;
let correctCount = 0;
let wrongCount = 0;
let currentMode = "predict"; // "predict" or "retrain"
let selectedPredictStyles = new Set(["real"]); // Styles selected for prediction display
let useTechnicalTerms = false; // Toggle between technical and simple terms

// ========== NAVIGATION ==========

// Highlight active nav link on scroll
const sections = document.querySelectorAll('.section');
const navLinks = document.querySelectorAll('.nav-link');

function updateActiveNav() {
  let current = '';
  sections.forEach(section => {
    const sectionTop = section.offsetTop - 100;
    const sectionHeight = section.offsetHeight;
    if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
      current = section.getAttribute('id');
    }
  });
  
  navLinks.forEach(link => {
    link.classList.remove('active');
    if (link.getAttribute('href') === `#${current}`) {
      link.classList.add('active');
    }
  });
}

window.addEventListener('scroll', updateActiveNav);
window.addEventListener('load', updateActiveNav);

// Smooth scroll for nav links
navLinks.forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const targetId = link.getAttribute('href');
    const targetSection = document.querySelector(targetId);
    if (targetSection) {
      targetSection.scrollIntoView({ behavior: 'smooth' });
    }
  });
});

// ========== THEME TOGGLE ==========

const themeToggle = document.getElementById('theme-toggle');

// Check for saved theme preference or default to dark
const savedTheme = localStorage.getItem('theme') || 'dark';
document.documentElement.setAttribute('data-theme', savedTheme);

themeToggle.addEventListener('click', () => {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
});

// ========== UTILITIES ==========

function toggleStyleButtons(state) {
  document.querySelectorAll(".style-button").forEach(btn => {
    if (!trainedStyles.has(btn.id)) {
      btn.disabled = !state;
    }
  });
}

function getTrueLabelFromPath(path) {
  if (path.includes("/cat/") || path.includes("_cat_")) return "cat";
  if (path.includes("/dog/") || path.includes("_dog_")) return "dog";
  return null;
}

function updateAccuracyDisplay() {
  const accuracyEl = document.getElementById("accuracy-percent");
  
  if (totalPredictions === 0) {
    accuracyEl.textContent = "0%";
    return;
  }

  const acc = Math.round((correctPredictions / totalPredictions) * 100);
  accuracyEl.textContent = `${acc}%`;
}

async function fileFromUrl(url) {
  const response = await fetch(url);
  const blob = await response.blob();
  return new File([blob], "image.jpg", { type: "image/jpeg" });
}

function shuffleArray(array) {
  return array
    .map(val => ({ val, sort: Math.random() }))
    .sort((a, b) => a.sort - b.sort)
    .map(({ val }) => val);
}

// ========== BUTTON EVENT HANDLERS ==========

document.getElementById("tech-info-toggle").addEventListener("click", () => {
  useTechnicalTerms = !useTechnicalTerms;
  updateTerminology();
});

document.getElementById("train").addEventListener("click", () => {
  currentMode = "retrain";
  updateModeUI();
  document.getElementById("baby-speech").textContent = useTechnicalTerms 
    ? "Ready to train on new data..." 
    : "Ready to learn from new data...";
});

document.querySelectorAll(".style-button").forEach(button => {
  button.addEventListener("click", async () => {
    const style = button.id;

    if (currentMode === "predict") {
      // Toggle style selection for prediction display
      if (selectedPredictStyles.has(style)) {
        selectedPredictStyles.delete(style);
        button.classList.remove("style-selected");
      } else {
        selectedPredictStyles.add(style);
        button.classList.add("style-selected");
      }
      loadGuessingImages();
    } else if (currentMode === "retrain") {
      // Retrain mode: load model for selected style
      if (trainedStyles.has(style)) return;

      trainedStyles.add(style);
      selectedStyles.push(style);
      
      // Show loading animation
      showLoadingAnimation();

      const styleList = Array.from(trainedStyles).sort();
      const query = styleList.map(s => `styles=${s}`).join("&");

      try {
        const response = await fetch(`${BACKEND_URL}/train/?${query}`, { method: "POST" });
        const data = await response.json();
        console.log(data);

        hideLoadingAnimation();

        if (response.ok) {
          document.getElementById("baby-speech").textContent = useTechnicalTerms 
            ? "Model trained on new styles!" 
            : "I have learned new styles!";
          document.getElementById("mother-speech").textContent = useTechnicalTerms 
            ? "Model retrained successfully. Ready for new predictions." 
            : "Great job! Let's see how well you do now!";
          button.classList.add("style-trained");
          showTrainingImages(style);
          
          // Switch back to predict mode
          currentMode = "predict";
          selectedPredictStyles = new Set(trainedStyles);
          updateModeUI();
          
          // Reset prediction counters
          correctPredictions = 0;
          totalPredictions = 0;
          correctCount = 0;
          wrongCount = 0;
          updateAccuracyDisplay();
          document.getElementById("correct-count").textContent = "0";
          document.getElementById("wrong-count").textContent = "0";
          
          // Load guessing images after short delay
          setTimeout(() => loadGuessingImages(), 1500);
        } else {
          alert("Model loading failed.");
        }
      } catch (e) {
        hideLoadingAnimation();
        alert("Model loading failed.");
      }
    }
  });
});

document.getElementById("guess").addEventListener("click", async () => {
  currentMode = "predict";
  updateModeUI();

  // Fetch currently loaded model from backend
  const response = await fetch(`${BACKEND_URL}/trained_styles`);
  if (response.ok) {
    const data = await response.json();
    const styles = data.loaded_styles.join(", ");
    console.log(`Predicting with model trained on: ${styles}`);
  }

  loadGuessingImages();
  document.getElementById("baby-speech").textContent = useTechnicalTerms 
    ? "Running predictions on test data..." 
    : "Click on images to guess...";
  document.getElementById("mother-speech").textContent = useTechnicalTerms 
    ? "Monitoring model performance metrics..." 
    : "Monitoring the toddler guessing cats and dogs...";
});

document.getElementById("restart").addEventListener("click", async () => {
  showLoadingAnimation();
  await fetch(`${BACKEND_URL}/reset`, { method: "POST" });
  hideLoadingAnimation();

  currentStyle = null;
  trainedStyles = new Set(["real"]);
  selectedStyles = ["real"];
  selectedPredictStyles = new Set(["real"]);
  currentMode = "predict";
  updateModeUI();

  document.getElementById("image-panel").innerHTML = "";
  document.getElementById("baby-speech").textContent = useTechnicalTerms 
    ? "Model reset. Ready to predict with real style." 
    : "Model reset. Ready to guess with real style!";
  document.getElementById("mother-speech").textContent = useTechnicalTerms 
    ? "Monitoring model performance metrics..." 
    : "Monitoring the toddler guessing cats and dogs...";

  correctPredictions = 0;
  totalPredictions = 0;
  updateAccuracyDisplay();
  correctCount = 0;
  wrongCount = 0;
  document.getElementById("correct-count").textContent = "0";
  document.getElementById("wrong-count").textContent = "0";
  
  loadGuessingImages();
});

// ========== IMAGE DISPLAY ==========

function showTrainingImages(style) {
  const panel = document.getElementById("image-panel");
  panel.innerHTML = "";

  const indices = shuffleArray([...Array(500).keys()]).slice(0, 25);

  const catImgs = indices.map(i =>
    `${BACKEND_URL}/dataset/train/${style}/cat/cat_${style}_${i}.jpg`
  );
  const dogImgs = indices.map(i =>
    `${BACKEND_URL}/dataset/train/${style}/dog/dog_${style}_${i}.jpg`
  );

  const combined = [];
  for (let i = 0; i < 25; i++) {
    combined.push(catImgs[i]);
    combined.push(dogImgs[i]);
  }

  combined.forEach(src => {
    const img = document.createElement("img");
    img.src = src;
    panel.appendChild(img);
  });
}

function loadGuessingImages() {
  const panel = document.getElementById("image-panel");
  panel.innerHTML = "";

  if (selectedPredictStyles.size === 0) {
    document.getElementById("baby-speech").textContent = useTechnicalTerms 
      ? "Please select at least one filter for prediction." 
      : "Please select at least one style to guess.";
    return;
  }

  const pool = [];

  // Only load images for selected predict styles
  selectedPredictStyles.forEach(style => {
    const indices = shuffleArray([...Array(100).keys()].map(i => i + 500)).slice(0, 10);

    const catImgs = indices.map(i =>
      `${BACKEND_URL}/dataset/test/${style}/cat/cat_${style}_${i}.jpg`
    );
    const dogImgs = indices.map(i =>
      `${BACKEND_URL}/dataset/test/${style}/dog/dog_${style}_${i}.jpg`
    );

    pool.push(...catImgs, ...dogImgs);
  });

  // Shuffle the combined pool
  const shuffledPool = shuffleArray(pool);

  shuffledPool.forEach(src => {
    const img = document.createElement("img");
    img.src = src;
    img.onclick = () => predict(img);
    panel.appendChild(img);
  });
  
  document.getElementById("baby-speech").textContent = useTechnicalTerms 
    ? "Click on images to run predictions..." 
    : "Click on images to guess...";
}

// ========== PREDICTION ==========

async function predict(imgEl) {
  console.log(imgEl)
  const file = await fileFromUrl(imgEl.src);
  const formData = new FormData();
  formData.append("file", file);

  const response = fetch(`${BACKEND_URL}/predict/`, {
    method: "POST",
    body: formData
  }).then(response => response.json())
    .then(result => {

      imgEl.classList.add("disabled-image");
      imgEl.onclick = null;

    document.getElementById("baby-speech").textContent = useTechnicalTerms 
      ? `Prediction: ${result.label}` 
      : `I guess it is a ${result.label}`;

      const trueLabel = getTrueLabelFromPath(imgEl.src);
      totalPredictions++;
      if (result.label === trueLabel) {
        correctPredictions++;
        correctCount++;
      } else {
        wrongCount++;
      }

    updateAccuracyDisplay();
    document.getElementById("correct-count").textContent = correctCount;
    document.getElementById("wrong-count").textContent = wrongCount;

      const accuracy = correctPredictions / totalPredictions;

    if (accuracy < 0.75) {
      document.getElementById("mother-speech").textContent = useTechnicalTerms 
        ? "Concept drift detected! Performance degradation observed." 
        : "Oh no, this image looks a little bit different... Consider retraining!";

        // Disable all remaining images in the guessing phase
        document.querySelectorAll("#image-panel img").forEach(img => {
          img.onclick = null;
          img.classList.add("disabled-image");
        });
        
        // Show retrain prompt
        showRetrainPrompt();
      }

    }).catch(e => { alert("Prediction failed.") });

}

// ========== MODE UI & LOADING ==========

function updateModeUI() {
  const trainBtn = document.getElementById("train");
  const guessBtn = document.getElementById("guess");
  
  if (currentMode === "predict") {
    trainBtn.classList.remove("btn-active");
    guessBtn.classList.add("btn-active");
    
    // Update hint text for predict mode
    updateHintText();
    
    // Enable all style buttons for selection
    document.querySelectorAll(".style-button").forEach(btn => {
      btn.disabled = false;
      // Mark selected styles
      if (selectedPredictStyles.has(btn.id)) {
        btn.classList.add("style-selected");
      } else {
        btn.classList.remove("style-selected");
      }
      // Mark trained styles
      if (trainedStyles.has(btn.id)) {
        btn.classList.add("style-trained");
      }
    });
  } else if (currentMode === "retrain") {
    trainBtn.classList.add("btn-active");
    guessBtn.classList.remove("btn-active");
    
    // Update hint text for train mode
    updateHintText();
    
    // Only enable untrained style buttons
    document.querySelectorAll(".style-button").forEach(btn => {
      btn.classList.remove("style-selected");
      if (trainedStyles.has(btn.id)) {
        btn.disabled = true;
        btn.classList.add("style-trained");
      } else {
        btn.disabled = false;
      }
    });
  }
}

function showLoadingAnimation() {
  // Create loading overlay if it doesn't exist
  let overlay = document.getElementById("loading-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "loading-overlay";
    const loadingText = useTechnicalTerms ? "CNN is training..." : "Toddler is learning...";
    overlay.innerHTML = `
      <div class="loading-content">
        <div class="loading-spinner"></div>
        <p class="loading-text">${loadingText}</p>
        <p class="loading-subtext">Model is loading</p>
      </div>
    `;
    document.body.appendChild(overlay);
  } else {
    // Update text based on current terminology
    const loadingText = useTechnicalTerms ? "CNN is training..." : "Toddler is learning...";
    overlay.querySelector(".loading-text").textContent = loadingText;
  }
  overlay.style.display = "flex";
  
  // Ensure minimum 5 second display
  window.loadingStartTime = Date.now();
}

function hideLoadingAnimation() {
  const overlay = document.getElementById("loading-overlay");
  if (overlay) {
    const elapsed = Date.now() - (window.loadingStartTime || 0);
    const minDuration = 5000; // 5 seconds minimum
    const remaining = Math.max(0, minDuration - elapsed);
    
    setTimeout(() => {
      overlay.style.display = "none";
    }, remaining);
  }
}

function updateTerminology() {
  const toggleBtn = document.getElementById("tech-info-toggle");
  
  // Update toggle button text
  toggleBtn.textContent = useTechnicalTerms 
    ? "Switch to non-technical terms" 
    : "Switch to technical terms";
  
  // Update all elements with data-technical and data-simple attributes
  document.querySelectorAll("[data-technical][data-simple]").forEach(el => {
    el.textContent = useTechnicalTerms ? el.dataset.technical : el.dataset.simple;
  });
  
  // Update character images
  const babyImg = document.getElementById("baby-img");
  const motherImg = document.getElementById("mother-img");
  
  if (babyImg) {
    babyImg.src = useTechnicalTerms 
      ? "images/neural-network-sketch.svg" 
      : "images/baby-sketch.svg";
    babyImg.alt = useTechnicalTerms 
      ? "Convolutional Neural Network" 
      : "ML Model (Toddler)";
  }
  
  if (motherImg) {
    motherImg.src = useTechnicalTerms 
      ? "images/drift-detector-sketch.svg" 
      : "images/mother-sketch.svg";
    motherImg.alt = useTechnicalTerms 
      ? "Concept Drift Detector" 
      : "Drift Detector (Parent)";
  }
  
  // Update speech bubbles
  updateSpeechBubbles();
  
  // Update hint text based on mode and terminology
  updateHintText();
}

function updateSpeechBubbles() {
  const babySpeech = document.getElementById("baby-speech");
  const motherSpeech = document.getElementById("mother-speech");
  
  // Get current context and update appropriately
  if (babySpeech) {
    const currentText = babySpeech.textContent;
    
    // Map common phrases between technical and simple
    if (currentText.includes("Ready to guess") || currentText.includes("Ready to predict")) {
      babySpeech.textContent = useTechnicalTerms 
        ? "Ready to predict with real style!" 
        : "Ready to guess with real style!";
    } else if (currentText.includes("learning") || currentText.includes("training")) {
      babySpeech.textContent = useTechnicalTerms 
        ? "Ready to train on new data..." 
        : "Ready to learn from new data...";
    } else if (currentText.includes("learned") || currentText.includes("trained")) {
      babySpeech.textContent = useTechnicalTerms 
        ? "I have been trained on new styles!" 
        : "I have learned new styles!";
    } else if (currentText.includes("guess it is") || currentText.includes("predict it is")) {
      const label = currentText.match(/a (\w+)$/)?.[1] || "cat";
      babySpeech.textContent = useTechnicalTerms 
        ? `Prediction: ${label}` 
        : `I guess it is a ${label}`;
    } else if (currentText.includes("Click on images")) {
      babySpeech.textContent = useTechnicalTerms 
        ? "Click on images to run predictions..." 
        : "Click on images to guess...";
    } else if (currentText.includes("Select") && currentText.includes("style")) {
      babySpeech.textContent = useTechnicalTerms 
        ? "Select a new style to train on..." 
        : "Select a new style to learn...";
    }
  }
  
  if (motherSpeech) {
    const currentText = motherSpeech.textContent;
    
    if (currentText.includes("monitoring") || currentText.includes("Monitoring")) {
      motherSpeech.textContent = useTechnicalTerms 
        ? "Monitoring model performance metrics..." 
        : "Monitoring the toddler guessing cats and dogs...";
    } else if (currentText.includes("Great job") || currentText.includes("retrained successfully")) {
      motherSpeech.textContent = useTechnicalTerms 
        ? "Model retrained successfully. Ready for new predictions." 
        : "Great job! Let's see how well you do now!";
    } else if (currentText.includes("different") || currentText.includes("drift")) {
      motherSpeech.textContent = useTechnicalTerms 
        ? "Concept drift detected! Performance degradation observed." 
        : "Oh no, this image looks a little bit different... Consider retraining!";
    }
  }
}

function updateHintText() {
  const hintText = document.getElementById("style-hint-text");
  if (!hintText) return;
  
  if (currentMode === "predict") {
    hintText.textContent = useTechnicalTerms 
      ? "Select filters to show images to predict" 
      : "Select filters to show images to guess";
  } else {
    hintText.textContent = useTechnicalTerms 
      ? "Select filters to train more" 
      : "Select filters to learn more";
  }
}

function showRetrainPrompt() {
  // Show retrain button/prompt in the UI
  let retrainPrompt = document.getElementById("retrain-prompt");
  if (!retrainPrompt) {
    retrainPrompt = document.createElement("div");
    retrainPrompt.id = "retrain-prompt";
    const promptText = useTechnicalTerms 
      ? "Performance dropped! Select a new style to retrain:" 
      : "The toddler is struggling! Select a new style to learn:";
    const btnText = useTechnicalTerms ? "Retrain Model" : "Learn More";
    retrainPrompt.innerHTML = `
      <p>${promptText}</p>
      <button id="retrain-btn" class="btn btn-primary">${btnText}</button>
    `;
    document.getElementById("controls").appendChild(retrainPrompt);
    
    document.getElementById("retrain-btn").addEventListener("click", () => {
      currentMode = "retrain";
      updateModeUI();
      retrainPrompt.style.display = "none";
      document.getElementById("baby-speech").textContent = useTechnicalTerms 
        ? "Select a new style to train on..." 
        : "Select a new style to learn...";
    });
  }
  retrainPrompt.style.display = "block";
}

// ========== INITIALIZATION ==========

function initializeApp() {
  // Set initial mode to predict
  currentMode = "predict";
  updateModeUI();
  
  // Mark real as trained and selected
  const realBtn = document.getElementById("real");
  if (realBtn) {
    realBtn.classList.add("style-trained");
    realBtn.classList.add("style-selected");
  }
  
  // Load initial guessing images
  loadGuessingImages();
  
  document.getElementById("baby-speech").textContent = useTechnicalTerms 
    ? "Ready to predict with real style!" 
    : "Ready to guess with real style!";
  document.getElementById("mother-speech").textContent = useTechnicalTerms 
    ? "Monitoring model performance metrics..." 
    : "Monitoring the toddler guessing cats and dogs...";
}

// Initialize when DOM is ready
window.addEventListener("load", () => {
  initializeApp();
});

