const BACKEND_URL = "http://localhost:8000";
let currentStyle = null;
let trainedStyles = new Set();
let selectedStyles = []; // Keeps style selection order
const allStyles = ["real", "cartoon", "sketch", "edge", "blur"];
let correctPredictions = 0;
let totalPredictions = 0;
let correctCount = 0;
let wrongCount = 0;

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
  const box = document.getElementById("tech-info-box");
  box.style.display = box.style.display === "none" ? "block" : "none";
});

document.getElementById("train").addEventListener("click", () => {
  toggleStyleButtons(true);                     // Enable only unselected styles
  document.getElementById("train").disabled = true;  // Disable Train
  document.getElementById("guess").disabled = true;  // Disable Guess
  document.getElementById("baby-speech").textContent = "Ready to learn from new data...";
});

document.querySelectorAll(".style-button").forEach(button => {
  button.addEventListener("click", async () => {
    const style = button.id;

    if (trainedStyles.has(style)) return;

    trainedStyles.add(style);
    selectedStyles.push(style);
    toggleStyleButtons(false); // Disable during loading

    const styleList = Array.from(trainedStyles).sort();
    const query = styleList.map(s => `styles=${s}`).join("&");

    const response = await fetch(`${BACKEND_URL}/train/?${query}`, { method: "POST" });
    const data = await response.json();
    console.log(data);

    if (response.ok) {
      document.getElementById("baby-speech").textContent = "I have seen cat and dog images.";
      button.disabled = true;
      showTrainingImages(style);
      toggleStyleButtons(false);               // Disable all style buttons
      document.getElementById("train").disabled = false;  // Re-enable Train
      document.getElementById("guess").disabled = false;  // Enable Guess
    } else {
      alert("Model loading failed.");
    }

  });
});

document.getElementById("guess").addEventListener("click", async () => {
  if (trainedStyles.size === 0) {
    alert("Please train on at least one style first.");
    return;
  }

  // Fetch currently loaded model from backend
  const response = await fetch(`${BACKEND_URL}/trained_styles`);
  if (response.ok) {
    const data = await response.json();
    const styles = data.loaded_styles.join(", ");
    console.log(`Predicting with model trained on: ${styles}`);
  }

  loadGuessingImages();
  document.getElementById("baby-speech").textContent = "Running predictions on test data...";
  document.getElementById("mother-speech").textContent = "Monitoring model performance...";
});

document.getElementById("restart").addEventListener("click", async () => {
  await fetch(`${BACKEND_URL}/reset`, { method: "POST" });

  currentStyle = null;
  trainedStyles.clear();
  selectedStyles = [];
  toggleStyleButtons(false); // disable all styles
  document.getElementById("train").disabled = false;
  document.getElementById("guess").disabled = false;

  document.getElementById("image-panel").innerHTML = "";
  document.getElementById("baby-speech").textContent = "Model reset. Ready for new training.";
  document.getElementById("mother-speech").textContent = "System reinitialized. Select a training style.";

  correctPredictions = 0;
  totalPredictions = 0;
  updateAccuracyDisplay();
  correctCount = 0;
  wrongCount = 0;
  document.getElementById("correct-count").textContent = "0";
  document.getElementById("wrong-count").textContent = "0";
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

  const pool = [];

  allStyles.forEach(style => {
    const indices = shuffleArray([...Array(100).keys()].map(i => i + 500)).slice(0, 10);

    const catImgs = indices.map(i =>
      `${BACKEND_URL}/dataset/test/${style}/cat/cat_${style}_${i}.jpg`
    );
    const dogImgs = indices.map(i =>
      `${BACKEND_URL}/dataset/test/${style}/dog/dog_${style}_${i}.jpg`
    );

    pool.push(...catImgs, ...dogImgs);
  });

  pool.forEach(src => {
    const img = document.createElement("img");
    img.src = src;
    img.onclick = () => predict(img);
    panel.appendChild(img);
  });
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

    document.getElementById("baby-speech").textContent = `I guess it is a ${result.label}`;

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
      document.getElementById("mother-speech").textContent =
        "Oh no, this image looks a little bit different...";

        // Disable all remaining images in the guessing phase
        document.querySelectorAll("#image-panel img").forEach(img => {
          img.onclick = null;
          img.classList.add("disabled-image");
        });
      }

    }).catch(e => { alert("Prediction failed.") });

}

