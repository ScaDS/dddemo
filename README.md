# Drift Detection Demonstrator (DDDemo)

An interactive demo application for understanding concept drift in machine learning. Watch how a model learns to classify cats and dogs, and see what happens when the data distribution changes.

## Prerequisite

To download the pre-trained models, you need to install Git LFS (see: https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage). 

Alternatively, you can download the models after cloning the repository.

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd baby_drift
```

### 2. Download the pre-trained Models

If you have installed Git LFS on your machine, the models should be initialized when cloning in step 1.
In case they have not been downloaded, you may need to run the command `git lfs pull`.

Alternatively, the models are available via the demonstrator website:
https://launchpad.scads.ai/dddemo/download_models

After downloading them, move the models to the `backend/models` directory.


### 3. Set Up Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the Application

Run the FastAPI server using the development mode:

```bash
fastapi dev backend/main.py
```

The application will be available at:
- **Web Application**: http://localhost:8000


## Docker Setup

Alternatively, you can run the application using Docker.

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

## Adding New Applications or Research Content

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
    'content/research/03-evaluation-approaches.html',
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
  'content/research/03-evaluation-approaches.html',
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

## Contributing

Feel free to submit issues and enhancement requests!

## Contact

elias.werner@tu-dresden.de
