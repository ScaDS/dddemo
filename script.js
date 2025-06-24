const allImages = [
  { src: 'images/real/real_cat_1.jpg', label: 'cat', style: 'real' },
  { src: 'images/real/real_cat_2.jpg', label: 'cat', style: 'real' },
  { src: 'images/real/real_cat_3.jpg', label: 'cat', style: 'real' },
  { src: 'images/real/real_dog_1.jpg', label: 'dog', style: 'real' },
  { src: 'images/real/real_dog_2.jpg', label: 'dog', style: 'real' },
  { src: 'images/real/real_dog_3.jpg', label: 'dog', style: 'real' },
  { src: 'images/cartoon/cartoon_cat_1.jpg', label: 'cat', style: 'cartoon' },
  { src: 'images/cartoon/cartoon_cat_2.jpg', label: 'cat', style: 'cartoon' },
  { src: 'images/cartoon/cartoon_cat_3.png', label: 'cat', style: 'cartoon' },
  { src: 'images/cartoon/cartoon_dog_1.jpg', label: 'dog', style: 'cartoon' },
  { src: 'images/cartoon/cartoon_dog_2.jpg', label: 'dog', style: 'cartoon' },
  { src: 'images/cartoon/cartoon_dog_3.jpg', label: 'dog', style: 'cartoon' },
  { src: 'images/sketch/sketch_cat_1.jpg', label: 'cat', style: 'sketch' },
  { src: 'images/sketch/sketch_cat_2.jpg', label: 'cat', style: 'sketch' },
  { src: 'images/sketch/sketch_cat_3.jpg', label: 'cat', style: 'sketch' },
  { src: 'images/sketch/sketch_dog_1.jpg', label: 'dog', style: 'sketch' },
  { src: 'images/sketch/sketch_dog_2.jpg', label: 'dog', style: 'sketch' },
  { src: 'images/sketch/sketch_dog_3.jpg', label: 'dog', style: 'sketch' }
];

let learning = false;
let guessing = false;
let confidence = 0;
const learned = new Set();
const rewardedLearning = new Set();
let mistakeCount = 0;

const imageGroupsDiv = document.getElementById('image-groups');
const babySpeech = document.getElementById('baby-speech');
const motherSpeech = document.getElementById('mother-speech');
const confidenceFill = document.getElementById('confidence-fill');

function updateConfidenceBar() {
  const clamped = Math.max(0, Math.min(100, confidence));
  confidenceFill.style.width = `${clamped}%`;

  let color = 'transparent';
  if (clamped === 0) color = 'transparent';
  else if (clamped < 30) color = 'red';
  else if (clamped < 50) color = 'orange';
  else if (clamped < 70) color = 'yellow';
  else if (clamped < 90) color = 'lightgreen';
  else color = 'green';

  confidenceFill.style.backgroundColor = color;
  confidenceFill.textContent = `${Math.round(clamped)}%`;
}

function renderImages() {
  imageGroupsDiv.innerHTML = '';
  const grouped = {};

  allImages.forEach(img => {
    if (!grouped[img.style]) grouped[img.style] = {};
    if (!grouped[img.style][img.label]) grouped[img.style][img.label] = [];
    grouped[img.style][img.label].push(img);
  });

  for (const style in grouped) {
    const groupDiv = document.createElement('div');
    groupDiv.className = 'image-group';

    const header = document.createElement('h3');
    header.textContent = style.toUpperCase();
    groupDiv.appendChild(header);

    for (const label in grouped[style]) {
      const row = document.createElement('div');
      row.className = 'image-row';

      grouped[style][label].forEach(img => {
        const imageEl = document.createElement('img');
        imageEl.src = img.src;
        imageEl.alt = img.label;
        imageEl.addEventListener('click', () => handleImageClick(img));
        row.appendChild(imageEl);
      });

      groupDiv.appendChild(row);
    }

    imageGroupsDiv.appendChild(groupDiv);
  }
}

function handleImageClick(img) {
  const key = `${img.style}-${img.label}`;
  if (learning) {
    learned.add(key);
    if (!rewardedLearning.has(key)) {
      confidence += 20;
      rewardedLearning.add(key);
    }
    babySpeech.textContent = '👶 Wow!';
    motherSpeech.textContent = `👩 This is a ${img.label}.`;
  } else if (guessing) {
    if (learned.has(key)) {
      if (confidence < 100) confidence += 5;
      babySpeech.textContent = `👶 I think it's a ${img.label}! ✅`;
      motherSpeech.textContent = `👩 Well done, baby!`;
    } else {
      confidence -= 10;
      mistakeCount++;
      babySpeech.textContent = `👶 I'm confused... ❌`;
      motherSpeech.textContent = mistakeCount === 1
        ? "👩 Oh, no dear, let's try one more time!"
        : "👩 I think it's time to learn more!";
    }

    if (rewardedLearning.size === 6 && mistakeCount < 2) {
      motherSpeech.textContent = "👩 Now, I think you know cats and dogs perfectly";
    }
  }

  confidence = Math.max(0, Math.min(100, confidence));
  updateConfidenceBar();
}

document.getElementById('start-learning').addEventListener('click', () => {
  learning = true;
  document.getElementById('start-learning').disabled = true;
  document.getElementById('stop-learning').disabled = false;
  babySpeech.textContent = '👶 Ready to learn!';
});

document.getElementById('stop-learning').addEventListener('click', () => {
  learning = false;
  document.getElementById('start-learning').disabled = false;
  document.getElementById('stop-learning').disabled = true;
  motherSpeech.textContent = "👩 Now, let's see how much you learned!";
});

document.getElementById('start-guessing').addEventListener('click', () => {
  guessing = true;
  document.getElementById('start-guessing').disabled = true;
  document.getElementById('stop-guessing').disabled = false;
  motherSpeech.textContent = "👩 Can you tell me what this is, baby?";
});

document.getElementById('stop-guessing').addEventListener('click', () => {
  guessing = false;
  document.getElementById('start-guessing').disabled = false;
  document.getElementById('stop-guessing').disabled = true;
});

document.getElementById('restart').addEventListener('click', () => {
  learning = false;
  guessing = false;
  confidence = 0;
  mistakeCount = 0;
  learned.clear();
  rewardedLearning.clear();

  document.getElementById('start-learning').disabled = false;
  document.getElementById('stop-learning').disabled = true;
  document.getElementById('start-guessing').disabled = false;
  document.getElementById('stop-guessing').disabled = true;

  updateConfidenceBar();
  babySpeech.textContent = "👶 I forgot everything!";
  motherSpeech.textContent = "👩 Hey baby, let's learn about dogs and cats!";
});

document.getElementById('tech-info-toggle').addEventListener('click', () => {
  const box = document.getElementById('tech-info-box');
  box.style.display = box.style.display === 'none' ? 'block' : 'none';
});

renderImages();
updateConfidenceBar();
