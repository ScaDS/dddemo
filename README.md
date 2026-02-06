# Drift Detector Demo (DDDemo)

An interactive demo application for understanding concept drift in machine learning. Watch how a model learns to classify cats and dogs, and see what happens when the data distribution changes.

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd baby_drift
```

### 2. Set Up Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Application

Run the FastAPI server using the development mode:

```bash
fastapi dev backend/main.py
```

The application will be available at:
- **Web Application**: http://localhost:8000 (Frontend UI)
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)

That's it! The frontend and backend are served together as a unified application.

## 📁 Project Structure

```
baby_drift/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── model.py             # CNN model definition and wrapper
│   ├── utils.py             # Utility functions
│   ├── prepare_images.py    # Image preprocessing script
│   ├── train_all_models.py  # Model training script
│   └── models/              # Trained model files (.pt)
├── frontend/
│   ├── index.html           # Main HTML page
│   ├── script.js            # Frontend JavaScript
│   ├── style.css            # Styling
│   ├── images/              # Frontend assets
│   └── content/             # Dynamic content for sliders
│       ├── applications/    # Application use case pages
│       └── research/        # Research topic pages
├── dataset/                 # Training dataset (cats and dogs)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 API Endpoints

### Load Model
```
POST /train/?styles=real&styles=cartoon
```
Load a pretrained model for the selected styles.

### Predict
```
POST /predict/
```
Upload an image and get a prediction (cat or dog).

### Reset Model
```
POST /reset/
```
Reset the model to untrained state.

### Get Loaded Styles
```
GET /loaded-styles/
```
Get the currently loaded style list.

## 🎯 Usage

1. **Load a Model**: Select one or more image styles (e.g., real, cartoon) and load the corresponding pretrained model.
2. **Upload Image**: Upload a cat or dog image to get a prediction.
3. **View Results**: The application will classify the image and display the result.

## 🛠️ Development

### Training Models

To train models on your dataset:

```bash
python backend/train_all_models.py
```

### Preparing Images

To preprocess images in the dataset:

```bash
python backend/prepare_images.py
```

## 🐛 Troubleshooting

### Module Import Errors

If you encounter "No module named 'backend'" errors:

**Solution**: Always run the FastAPI server from the project root directory:
```bash
cd /home/eliasw/baby_drift
fastapi dev backend/main.py
```

### Port Already in Use

If port 8000 is already in use, you can specify a different port:
```bash
fastapi dev backend/main.py --port 8080
```

### CUDA/GPU Issues

If you have GPU issues, the application will automatically fall back to CPU. To force CPU usage:
```python
# In backend/model.py, modify:
self.device = torch.device("cpu")
```

## 📦 Dependencies

- **FastAPI**: Modern web framework for building APIs
- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision library
- **scikit-learn**: Machine learning utilities
- **River**: Online machine learning library
- **Pillow**: Image processing
- **OpenCV**: Computer vision operations

## 🐳 Docker Setup

The easiest way to run the application is using Docker.

### Prerequisites

- Docker
- Docker Compose

### Run with Docker Compose

```bash
# Build and start the container
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### Run with Docker directly

```bash
# Build the image
docker build -t dddemo .

# Run the container
docker run -p 8000:8000 dddemo
```

The application will be available at http://localhost:8000

### Stop the container

```bash
# If using docker-compose
docker-compose down

# If using docker directly
docker stop <container-id>
```

## 📝 Notes

- The application uses a simple CNN architecture for classification
- Models are saved in `backend/models/` directory
- The frontend and backend are served together by FastAPI as a unified application
- Frontend static files (HTML, CSS, JS) are served from the `frontend/` directory
- CORS is enabled for all origins (modify in production if needed)
- Dataset images are served via `/dataset` endpoint for the frontend to display
- Toggle between "technical" and "non-technical" terminology in the UI

## ➕ Adding New Applications or Research Content

The Application and Research sections use a dynamic content loading system. You can easily add new tabs by creating HTML files in the appropriate directories.

### Directory Structure

```
frontend/content/
├── applications/          # Application use case pages
│   ├── 01-transport-mode.html
│   ├── 02-marine-event.html
│   └── 03-your-usecase.html
└── research/              # Research topic pages
    ├── 01-overview.html
    ├── 02-computational-requirements.html
    └── 03-synthetic-evaluation.html
```

### Step 1: Create the Content File

Create a new HTML file in the appropriate directory:
- For applications: `frontend/content/applications/`
- For research: `frontend/content/research/`

**File naming convention**: Use a numbered prefix for organization (e.g., `04-new-topic.html`).

### Step 2: Add Metadata and Content

Each content file must include a metadata comment at the top:

```html
<!--
  tab-name: Your Tab Name Here
  order: 4
-->
<div class="slide-content">
  <h3>Your Page Title</h3>
  <p class="slide-intro">Your introduction text here...</p>
  
  <!-- Add any HTML content you want -->
  <p>More content...</p>
  <img src="images/your-image.png" alt="Description">
</div>
```

**Metadata fields:**
- `tab-name`: The text displayed on the navigation tab button
- `order`: Determines tab order (lower numbers appear first)

### Step 3: Register the File

Add the new file path to the `contentRegistry` in `frontend/script.js`:

```javascript
const contentRegistry = {
  'content/applications': [
    'content/applications/01-transport-mode.html',
    'content/applications/02-marine-event.html',
    'content/applications/03-your-usecase.html',
    'content/applications/04-your-new-file.html'  // Add here
  ],
  'content/research': [
    'content/research/01-overview.html',
    'content/research/02-computational-requirements.html',
    'content/research/03-synthetic-evaluation.html',
    'content/research/04-your-new-file.html'  // Or here
  ]
};
```

### Example: Adding a New Research Topic

1. Create `frontend/content/research/04-detection-methods.html`:

```html
<!--
  tab-name: Detection Methods
  order: 4
-->
<div class="slide-content">
  <h3>Drift Detection Methods</h3>
  <p class="slide-intro">An overview of various drift detection algorithms...</p>
  
  <h4>Statistical Methods</h4>
  <ul>
    <li>Page-Hinkley Test</li>
    <li>ADWIN</li>
    <li>DDM (Drift Detection Method)</li>
  </ul>
  
  <h4>Window-Based Methods</h4>
  <p>These methods compare distributions between sliding windows...</p>
</div>
```

2. Add to `contentRegistry` in `frontend/script.js`:

```javascript
'content/research': [
  'content/research/01-overview.html',
  'content/research/02-computational-requirements.html',
  'content/research/03-synthetic-evaluation.html',
  'content/research/04-detection-methods.html'
]
```

3. Refresh the page - the new tab will appear automatically!

### Available CSS Classes

You can use these pre-defined CSS classes in your content:

- `.slide-content` - Main content wrapper (required)
- `.slide-intro` - Styled introduction paragraph
- `.placeholder-text` - Italic placeholder text
- `.research-gifs` - Container for images/GIFs
- `.research-gif` - Styled image with shadow and rounded corners

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

[Add your license information here]
