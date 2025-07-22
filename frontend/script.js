const BACKEND_URL = "http://localhost:8000";
let currentStyle = null;
let trainingStarted = false;
const trainedStyles = new Set();
const allStyles = ["real", "cartoon", "sketch", "edge", "blur"];
let correctPredictions = 0;
let totalPredictions = 0;
let correctCount = 0;
let wrongCount = 0;

// ========== UTILITY FUNCTIONS ==========

function toggleStyleButtons(state) {
  document.querySelectorAll(".style-button").forEach(btn => {
    btn.disabled = !state;
  });
}

// function updateConfidenceBar(confidence) {
//   const fill = document.getElementById("accuracy-fill");
//   fill.style.width = `${confidence}%`;
//   fill.textContent = `${Math.round(confidence)}%`;

//   fill.style.backgroundColor =
//     confidence < 30 ? "red" :
//       confidence < 50 ? "orange" :
//         confidence < 70 ? "yellow" :
//           confidence < 90 ? "lightgreen" : "green";
// }

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

// ========== BUTTON EVENT HANDLERS ==========

document.getElementById("tech-info-toggle").addEventListener("click", () => {
  const box = document.getElementById("tech-info-box");
  box.style.display = box.style.display === "none" ? "block" : "none";
});

document.getElementById("train").addEventListener("click", () => {
  trainingStarted = true;
  toggleStyleButtons(true);
  document.getElementById("baby-speech").textContent = "👶 Ready to learn!";
  // document.getElementById("mother-speech").textContent = "👩 Select a style to start learning.";
});

document.querySelectorAll(".style-button").forEach(button => {
  button.addEventListener("click", async () => {
    if (!trainingStarted) return;
    currentStyle = button.id;
    toggleStyleButtons(false);

    const response = await fetch(`${BACKEND_URL}/train/?style=${currentStyle}`, {
      method: "POST"
    });

    if (response.ok) {
      trainedStyles.add(currentStyle);
      document.getElementById("baby-speech").textContent = "👶 I learned 100 pictures!";
      // document.getElementById("mother-speech").textContent = "👩 Great job, baby!";
      showTrainingImages(currentStyle);
    } else {
      alert("Training failed");
    }

    trainingStarted = false;
  });
});

document.getElementById("guess").addEventListener("click", () => {
  if (trainedStyles.size === 0) {
    alert("Please train the model first.");
    return;
  }
  loadGuessingImages();
  document.getElementById("baby-speech").textContent = "👶 Let me try guessing...";
  // document.getElementById("mother-speech").textContent = "👩 Go ahead, baby!";
});

document.getElementById("restart").addEventListener("click", async () => {
  await fetch(`${BACKEND_URL}/reset`, { method: "POST" });
  currentStyle = null;
  trainingStarted = false;
  trainedStyles.clear();
  toggleStyleButtons(false);
  document.getElementById("image-panel").innerHTML = "";
  document.getElementById("baby-speech").textContent = "👶 I forgot everything!";
  document.getElementById("mother-speech").textContent = "👩 I am still watching you!";
  // updateConfidenceBar(0);
  correctPredictions = 0;
  totalPredictions = 0;
  updateAccuracyBar();
  correctCount = 0;
  wrongCount = 0;
  document.getElementById("correct-count").textContent = "0";
  document.getElementById("wrong-count").textContent = "0";
});

// ========== IMAGE HANDLING ==========

function showTrainingImages(style) {
  const panel = document.getElementById("image-panel");
  panel.innerHTML = "";
  for (let i = 0; i < 50; i++) {
    const img1 = document.createElement("img");
    img1.src = `${BACKEND_URL}/dataset/train/${style}/cat/cat_${style}_${i}.jpg`;
    panel.appendChild(img1);

    const img2 = document.createElement("img");
    img2.src = `${BACKEND_URL}/dataset/train/${style}/dog/dog_${style}_${i}.jpg`;
    panel.appendChild(img2);
  }
}

function loadGuessingImages() {
  const panel = document.getElementById("image-panel");
  panel.innerHTML = "";

  allStyles.forEach(style => {
    for (let i = 80; i < 85; i++) {
      const catImg = document.createElement("img");
      catImg.src = `${BACKEND_URL}/dataset/test/${style}/cat/cat_${style}_${i}.jpg`;
      // catImg.addEventListener("click", () => predict(catImg));
      catImg.onclick = () => predict(catImg);
      panel.appendChild(catImg);

      const dogImg = document.createElement("img");
      dogImg.src = `${BACKEND_URL}/dataset/test/${style}/dog/dog_${style}_${i}.jpg`;
      // dogImg.addEventListener("click", () => predict(dogImg));
      dogImg.onclick = () => predict(dogImg);
      panel.appendChild(dogImg);
    }
  });
}

async function predict(imgEl) {
  // console.log(imgEl)
  const file = await fileFromUrl(imgEl.src);
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${BACKEND_URL}/predict/`, {
    method: "POST",
    body: formData
  });

  if (response.ok) {
    const result = await response.json();
    imgEl.classList.add("disabled-image");
    imgEl.onclick = null;
    // document.getElementById("baby-speech").textContent = `👶 I think it's a ${result.label} (${result.confidence}%)`;
    // document.getElementById("mother-speech").textContent =
    //   result.confidence < 50
    //     ? "👩 Hmm, I think you are confused!"
    //     : "👩 Well done!";
    document.getElementById("baby-speech").textContent = `👶 I think it's a ${result.label}!`;
    const accuracy = correctPredictions / totalPredictions;
    // document.getElementById("mother-speech").textContent =
    // accuracy < 0.5
    //   ? "👩 I think you're confused! You need to learn more!"
    //   : "👩 Well done!";
    if (accuracy < 0.4) {
      document.getElementById("mother-speech").textContent = "👩 I think you're confused! You need to learn more!";
    }
    // updateConfidenceBar(result.confidence);
    const trueLabel = getTrueLabelFromPath(imgEl.src);
    totalPredictions++;
    // if (result.label === trueLabel) correctPredictions++;
    if (result.label === trueLabel) {
      correctPredictions++;
      correctCount++;
    } else {
      wrongCount++;
    }
    updateAccuracyBar();
    document.getElementById("correct-count").textContent = correctCount;
    document.getElementById("wrong-count").textContent = wrongCount;
  } else {
    alert("Prediction failed");
  }
}
