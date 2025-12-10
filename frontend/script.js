const BACKEND_URL = "http://localhost:8000";
let currentStyle = null;
let trainedStyles = new Set();
let selectedStyles = []; // Keeps style selection order
const allStyles = ["real", "cartoon", "sketch", "edge", "blur"];
let correctPredictions = 0;
let totalPredictions = 0;
let correctCount = 0;
let wrongCount = 0;

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

function updateAccuracyBar() {
  const bar = document.getElementById("accuracy-fill");
  const label = document.getElementById("accuracy-label");

  if (totalPredictions === 0) {
    bar.style.width = "0%";
    bar.textContent = "0%";
    label.textContent = "Model Accuracy";
    bar.style.backgroundColor = "transparent";
    return;
  }

  const acc = Math.round((correctPredictions / totalPredictions) * 100);
  bar.style.width = `${acc}%`;
  bar.textContent = `${acc}%`;
  label.textContent = "Model Accuracy";

  bar.style.backgroundColor =
    acc < 30 ? "red" :
      acc < 50 ? "orange" :
        acc < 70 ? "yellow" :
          acc < 90 ? "lightgreen" : "green";
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
  document.getElementById("baby-speech").textContent = "👶 I'm ready to learn!";
  // document.getElementById("mother-speech").textContent = "👩 Pick a style to train on!";
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
      document.getElementById("baby-speech").textContent = "👶 I saw new pictures!";
      // document.getElementById("mother-speech").textContent = "👩 You're learning fast!";
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

  // 🔍 Fetch currently loaded model from backend
  const response = await fetch(`${BACKEND_URL}/trained_styles`);
  if (response.ok) {
    const data = await response.json();
    const styles = data.loaded_styles.join(", ");
    console.log(`✅ Guessing with model trained on: ${styles}`);
  }

  loadGuessingImages();
  document.getElementById("baby-speech").textContent = "👶 Let me try guessing...";
  document.getElementById("mother-speech").textContent = "👩 I'm watching you guess!";
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
  document.getElementById("baby-speech").textContent = "👶 I forgot everything!";
  document.getElementById("mother-speech").textContent = "👩 Let's start from the beginning!";

  correctPredictions = 0;
  totalPredictions = 0;
  updateAccuracyBar();
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

      document.getElementById("baby-speech").textContent = `👶 I think it's a ${result.label}!`;

      const trueLabel = getTrueLabelFromPath(imgEl.src);
      totalPredictions++;
      if (result.label === trueLabel) {
        correctPredictions++;
        correctCount++;
      } else {
        wrongCount++;
      }

      updateAccuracyBar();
      document.getElementById("correct-count").textContent = correctCount;
      document.getElementById("wrong-count").textContent = wrongCount;

      const accuracy = correctPredictions / totalPredictions;

      if (accuracy < 0.75) {
        document.getElementById("mother-speech").textContent =
          "👩 I think you're confused! You need to learn more!";

        // Disable all remaining images in the guessing phase
        document.querySelectorAll("#image-panel img").forEach(img => {
          img.onclick = null;
          img.classList.add("disabled-image");
        });
      }

    }).catch(e => { alert("Prediction failed.") });

}

