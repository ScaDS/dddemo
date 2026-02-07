const BACKEND_URL = "http://localhost:8000";
let currentStyle = null;
let trainedStyles = new Set(["real"]); // Start with real as default
let selectedStyles = ["real"]; // Keeps style selection order
const allStyles = ["real", "cartoon", "sketch", "edge", "blur"];
let correctPredictions = 0;
let totalPredictions = 0;
let correctCount = 0;
let wrongCount = 0;
let predictionWindow = []; // Sliding window of last 10 predictions (true/false for correct/incorrect)
const WINDOW_SIZE = 10;
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

// ========== ROUNDABOUT SCROLL EFFECT ==========

let roundaboutTransitioned = false;

function initRoundabout() {
  const panelIntro = document.querySelector('.panel-intro');
  const panelGame = document.querySelector('.panel-game');
  const fundamentalsSection = document.getElementById('fundamentals');
  
  if (!panelIntro || !panelGame || !fundamentalsSection) return;
  
  function checkScroll() {
    const sectionRect = fundamentalsSection.getBoundingClientRect();
    // Trigger only when the section top is well above the viewport (scrolled significantly into view)
    const triggerPoint = -200; // Must scroll 200px past the section top
    
    // Transition when the fundamentals section is scrolled past the trigger point
    if (sectionRect.top < triggerPoint && !roundaboutTransitioned) {
      roundaboutTransitioned = true;
      panelIntro.classList.remove('active');
      panelIntro.classList.add('rotating-out');
      panelGame.classList.add('active');
      
      // Hide intro panel after animation completes
      setTimeout(() => {
        panelIntro.classList.add('hidden');
      }, 800);
    }
  }
  
  window.addEventListener('scroll', checkScroll);
  
  // Also allow clicking on the intro to transition
  panelIntro.addEventListener('click', () => {
    if (!roundaboutTransitioned) {
      transitionToGame();
    }
  });
  
  // Back button to show intro again
  const backBtn = document.getElementById('back-to-intro');
  if (backBtn) {
    backBtn.addEventListener('click', () => {
      transitionToIntro();
    });
  }
}

function transitionToGame() {
  const panelIntro = document.querySelector('.panel-intro');
  const panelGame = document.querySelector('.panel-game');
  
  roundaboutTransitioned = true;
  panelIntro.classList.remove('active');
  panelIntro.classList.add('rotating-out');
  panelGame.classList.add('active');
  
  setTimeout(() => {
    panelIntro.classList.add('hidden');
  }, 800);
}

function transitionToIntro() {
  const panelIntro = document.querySelector('.panel-intro');
  const panelGame = document.querySelector('.panel-game');
  
  roundaboutTransitioned = false;
  panelIntro.classList.remove('hidden');
  panelIntro.classList.remove('rotating-out');
  panelIntro.classList.add('active');
  panelGame.classList.remove('active');
}

window.addEventListener('load', initRoundabout);

// ========== THEME ==========

// Set default theme to light
document.documentElement.setAttribute('data-theme', 'light');

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
  const hintEl = document.getElementById("accuracy-hint");
  const correctEl = document.getElementById("correct-count");
  const wrongEl = document.getElementById("wrong-count");
  
  if (totalPredictions === 0) {
    accuracyEl.textContent = "0%";
    if (hintEl) hintEl.textContent = `Based on last ${WINDOW_SIZE} samples`;
    if (correctEl) correctEl.textContent = "0";
    if (wrongEl) wrongEl.textContent = "0";
    return;
  }

  // Use sliding window for all stats
  const windowCorrect = predictionWindow.filter(p => p).length;
  const windowWrong = predictionWindow.length - windowCorrect;
  
  if (correctEl) correctEl.textContent = windowCorrect;
  if (wrongEl) wrongEl.textContent = windowWrong;
  
  let acc;
  if (predictionWindow.length >= WINDOW_SIZE) {
    acc = Math.round((windowCorrect / WINDOW_SIZE) * 100);
    if (hintEl) hintEl.textContent = `Based on last ${WINDOW_SIZE} samples`;
  } else {
    acc = Math.round((windowCorrect / predictionWindow.length) * 100);
    if (hintEl) hintEl.textContent = `${predictionWindow.length}/${WINDOW_SIZE} samples`;
  }
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
  
  // Hide retrain prompt if visible
  const retrainPrompt = document.getElementById("retrain-prompt");
  if (retrainPrompt) {
    retrainPrompt.style.display = "none";
  }
  
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
          predictionWindow = [];
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
  await fetch(`${BACKEND_URL}/reset`, { method: "POST" });

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

// Prediction phrase variants for the toddler
const predictionPhrases = [
  "I think it is a",
  "I believe it is a",
  "I guess it is a",
  "It looks like a",
  "That must be a",
  "I'm pretty sure it's a"
];

function getRandomPredictionPhrase() {
  return predictionPhrases[Math.floor(Math.random() * predictionPhrases.length)];
}

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

    const babySpeech = document.getElementById("baby-speech");
    const phrase = getRandomPredictionPhrase();
    babySpeech.innerHTML = useTechnicalTerms 
      ? `Prediction: <strong>${result.label}</strong>` 
      : `<span style="white-space:nowrap">${phrase} <strong>${result.label}</strong></span>`;
    
    // Trigger shine animation
    babySpeech.classList.remove("shine");
    void babySpeech.offsetWidth; // Force reflow to restart animation
    babySpeech.classList.add("shine");

      const trueLabel = getTrueLabelFromPath(imgEl.src);
      totalPredictions++;
      const isCorrect = result.label === trueLabel;
      if (isCorrect) {
        correctPredictions++;
        correctCount++;
        imgEl.classList.add("prediction-correct");
      } else {
        wrongCount++;
        imgEl.classList.add("prediction-wrong");
      }
      imgEl.classList.add("disabled-image");
      
      // Add to sliding window
      predictionWindow.push(isCorrect);
      if (predictionWindow.length > WINDOW_SIZE) {
        predictionWindow.shift();
      }

    updateAccuracyDisplay();

      // Compute accuracy on sliding window (only if we have enough predictions)
      const windowCorrect = predictionWindow.filter(p => p).length;
      const windowAccuracy = predictionWindow.length >= WINDOW_SIZE 
        ? windowCorrect / WINDOW_SIZE 
        : correctPredictions / totalPredictions;

    if (windowAccuracy < 0.75 && predictionWindow.length >= WINDOW_SIZE) {
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
      const label = currentText.match(/a (\w+)$/)?.[1] || currentText.match(/<strong>(\w+)<\/strong>/)?.[1] || "cat";
      babySpeech.innerHTML = useTechnicalTerms 
        ? `Prediction: <strong>${label}</strong>` 
        : `I guess it is <span style="white-space:nowrap">a <strong>${label}</strong></span>`;
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
window.addEventListener("load", async () => {
  initializeApp();
  await loadDynamicSlider('.application-slider');
  await loadDynamicSlider('.research-slider');
});

// ========== DYNAMIC CONTENT LOADING ==========

// Content file registry - add new files here to include them in the sliders
const contentRegistry = {
  'content/applications': [
    'content/applications/transport-mode/index.html',
    'content/applications/marine-event/index.html',
    'content/applications/your-usecase/index.html'
  ],
  'content/research': [
    'content/research/overview/index.html',
    'content/research/computational-requirements/index.html',
    'content/research/evaluation-approaches/index.html'
  ]
};

async function loadDynamicSlider(containerSelector) {
  const container = document.querySelector(containerSelector);
  if (!container) return;
  
  const contentDir = container.dataset.contentDir;
  if (!contentDir || !contentRegistry[contentDir]) {
    console.warn(`No content registry found for: ${contentDir}`);
    return;
  }
  
  const sliderNav = container.querySelector('.slider-nav');
  const sliderTrack = container.querySelector('.slider-track');
  
  if (!sliderNav || !sliderTrack) return;
  
  // Clear existing content
  sliderNav.innerHTML = '';
  sliderTrack.innerHTML = '';
  
  const contentFiles = contentRegistry[contentDir];
  const slides = [];
  
  // Load all content files
  for (const filePath of contentFiles) {
    try {
      const response = await fetch(filePath);
      if (!response.ok) continue;
      
      const html = await response.text();
      const metadata = parseContentMetadata(html);
      const content = extractContentBody(html);
      
      slides.push({
        tabName: metadata.tabName || 'Untitled',
        order: metadata.order || 999,
        content: content
      });
    } catch (e) {
      console.warn(`Failed to load content file: ${filePath}`, e);
    }
  }
  
  // Sort by order
  slides.sort((a, b) => a.order - b.order);
  
  // Generate tabs and slides
  slides.forEach((slide, index) => {
    // Create nav button
    const navBtn = document.createElement('button');
    navBtn.className = 'slider-nav-btn' + (index === 0 ? ' active' : '');
    navBtn.dataset.slide = index;
    navBtn.textContent = slide.tabName;
    sliderNav.appendChild(navBtn);
    
    // Create slide
    const slideDiv = document.createElement('div');
    slideDiv.className = 'slider-slide';
    slideDiv.dataset.slide = index;
    slideDiv.innerHTML = slide.content;
    sliderTrack.appendChild(slideDiv);
    
    // Execute any script tags in the content
    executeScripts(slideDiv);
  });
  
  // Initialize slider functionality
  initSlider(containerSelector);
}

function executeScripts(container) {
  const scripts = container.querySelectorAll('script');
  scripts.forEach(oldScript => {
    const newScript = document.createElement('script');
    
    // Copy attributes
    Array.from(oldScript.attributes).forEach(attr => {
      newScript.setAttribute(attr.name, attr.value);
    });
    
    // Copy content or src
    if (oldScript.src) {
      newScript.src = oldScript.src;
    } else {
      newScript.textContent = oldScript.textContent;
    }
    
    // Replace old script with new one to trigger execution
    oldScript.parentNode.replaceChild(newScript, oldScript);
  });
}

function parseContentMetadata(html) {
  const metadata = {
    tabName: 'Untitled',
    order: 999
  };
  
  // Parse HTML comment metadata
  const commentMatch = html.match(/<!--([\s\S]*?)-->/);
  if (commentMatch) {
    const commentContent = commentMatch[1];
    
    const tabNameMatch = commentContent.match(/tab-name:\s*(.+)/i);
    if (tabNameMatch) {
      metadata.tabName = tabNameMatch[1].trim();
    }
    
    const orderMatch = commentContent.match(/order:\s*(\d+)/i);
    if (orderMatch) {
      metadata.order = parseInt(orderMatch[1]);
    }
  }
  
  return metadata;
}

function extractContentBody(html) {
  // Remove the metadata comment and return the rest
  return html.replace(/<!--[\s\S]*?-->/, '').trim();
}

// ========== SLIDER ==========

function initSlider(containerSelector) {
  const container = document.querySelector(containerSelector);
  if (!container) return;
  
  const sliderTrack = container.querySelector('.slider-track');
  const navBtns = container.querySelectorAll('.slider-nav-btn');
  const prevBtn = container.querySelector('.slider-prev');
  const nextBtn = container.querySelector('.slider-next');
  
  if (!sliderTrack || navBtns.length === 0) return;
  
  let currentSlide = 0;
  const totalSlides = navBtns.length;
  
  function goToSlide(index) {
    if (index < 0 || index >= totalSlides) return;
    
    currentSlide = index;
    sliderTrack.style.transform = `translateX(-${currentSlide * 100}%)`;
    
    // Update nav buttons
    navBtns.forEach(btn => btn.classList.remove('active'));
    navBtns[currentSlide].classList.add('active');
    
    // Update arrow states
    updateArrows();
  }
  
  function updateArrows() {
    if (prevBtn) prevBtn.disabled = currentSlide === 0;
    if (nextBtn) nextBtn.disabled = currentSlide === totalSlides - 1;
  }
  
  // Nav button clicks
  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const slideIndex = parseInt(btn.dataset.slide);
      goToSlide(slideIndex);
    });
  });
  
  // Arrow clicks
  if (prevBtn) {
    prevBtn.addEventListener('click', () => goToSlide(currentSlide - 1));
  }
  if (nextBtn) {
    nextBtn.addEventListener('click', () => goToSlide(currentSlide + 1));
  }
  
  // Initialize
  updateArrows();
}

