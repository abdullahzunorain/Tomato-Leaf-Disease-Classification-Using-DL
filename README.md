# Tomato Leaf Disease Classification Using Deep Learning

**Simple, step-by-step guide — from setup to training to running the app.**

---

## Table of contents

1. [Project overview](#project-overview)
2. [Repository structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Quick setup — one-shot (recommended)](#quick-setup)
5. [Manual setup (detailed) — step by step](#manual-setup)

   * Create virtual environment
   * Install backend dependencies
   * Install frontend dependencies
   * Configure environment variables
6. [Run the application locally](#run-locally)

   * Start backend API
   * Start frontend
   * Test with curl (example)
7. [Training the model (notebooks)](#training)
8. [Using the pre-trained model](#pretrained)
9. [Keeping repository clean (git/.gitignore) and handling large files](#git-lfs-and-large-files)
10. [Deployment (short guide)](#deployment)
11. [Troubleshooting & common issues](#troubleshooting)
12. [Useful git commands](#git-commands)
13. [License & contact](#license)

---

## Project overview

This project implements a **tomato leaf disease classification** pipeline using deep learning. It contains:

* a Python backend (API) to serve model predictions,
* a React frontend to upload leaf images and visualize results,
* Jupyter notebooks for data preprocessing and model training,
* saved model artifacts kept out of the repository via `.gitignore`.

The repository is organized so you can run the whole demo locally (frontend + backend) or re-train the model from the notebooks.

---

## Repository structure

```
tomato disease classification project/       # project root
├─ api/                      # backend API (Python) - run this to get /predict endpoint
├─ frontend/                 # React app (UI) - runs on localhost:3000 by default
├─ saved_models/             # local-only: trained model(s) (ignored by git)
├─ training (jupyter notebook)/  # Jupyter notebooks for preprocessing & training
├─ tomato_dataset_v1/        # local dataset folder (ignored by git)
├─ tomato_aug_dataset.zip    # packed dataset (ignored by git)
├─ best_model.h5             # large trained model (ignored by git)
├─ .gitignore                # important - prevents large files from being pushed
└─ README.md                 # you are reading this
```

---

## Prerequisites

* OS: Windows, macOS, or Linux (instructions below are cross-platform; Windows examples shown where necessary)
* Python 3.8+ (3.10 recommended)
* Node.js 14+ and npm (for the frontend)
* Git (already installed)
* Optional: GPU + CUDA if you want faster training with TensorFlow (make sure compatible TF build)

---

## Quick setup
 ` — one-shot (recommended) `

If you want to get running quickly (backend + frontend) and you have `python`, `pip`, `node` and `npm` installed, run the following from the project root:

```bash
# 1. create & activate a virtual environment (Windows example)
python -m venv venv
# PowerShell
venv\Scripts\Activate.ps1
# or cmd
# venv\Scripts\activate.bat
# 2. install backend requirements
pip install -r api/requirements.txt
# 3. install frontend dependencies
cd frontend
npm install
cd ..
# 4. copy environment files and edit values if needed
cp frontend/.env.example frontend/.env        # Linux / mac
copy frontend\.env.example frontend\.env    # Windows cmd
# 5. run the backend (from project root)
python api/main.py
# 6. in a new terminal start the frontend
cd frontend
npm start
```

Open the frontend in your browser at `http://localhost:3000` (or the address printed by npm). The UI should connect to the backend API and let you upload leaf images for prediction.

---

## Manual setup (detailed) — step by step

Each step below includes why it matters.

### 1) Clone the repo (if not already local)

```bash
git clone https://github.com/abdullahzunorain/Tomato-Leaf-Disease-Classification-Using-DL.git
cd Tomato-Leaf-Disease-Classification-Using-DL
```

**Why:** get project files locally.

### 2) Create & activate a Python virtual environment

```bash
# Create
python -m venv venv

# Activate (Windows PowerShell)
venv\Scripts\Activate.ps1

# Activate (Linux / macOS)
source venv/bin/activate
```

**Why:** isolates Python packages for this project so system packages remain clean and versions remain reproducible.

### 3) Install backend dependencies

```bash
pip install -r api/requirements.txt
```

**Why:** installs Flask/FastAPI, TensorFlow/Keras, Pillow, numpy, or whatever the backend needs. Without these packages the API will fail to run.

### 4) Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

**Why:** installs React and all packages defined in `package.json`. The UI requires these to build/serve locally.

### 5) Configure environment variables

* Frontend: `frontend/.env.example` shows which variables the frontend expects (for example API base URL). Copy it to `frontend/.env` and update values.

```bash
# Linux / mac
cp frontend/.env.example frontend/.env
# Windows cmd
copy frontend\.env.example frontend\.env
```

* Backend: if your API expects secrets or paths, create a `.env` in `api/` or set system environment variables. The backend will read these if implemented.

**Why:** `.env` files keep runtime configuration separate for local vs production environments. They are ignored by git so secrets are not pushed.

---

## Run the application locally

Below are detailed commands and what they do.

### Start backend API

Open a terminal (virtual environment activated) and run:

```bash
# from project root
python api/main.py
# or, if the project uses Flask entry points:
# set FLASK_APP=api/main.py  (Windows cmd)
# export FLASK_APP=api/main.py (Linux/mac)
# flask run --port 5000
```

**What to expect:** backend should print something like `Running on http://127.0.0.1:5000`. Check `api/main.py` for the exact route names. Many projects expose `/predict` or `/api/predict`.

### Start frontend

In another terminal (inside `frontend` folder):

```bash
cd frontend
npm start
```

**What to expect:** the React dev server starts (default `http://localhost:3000`). The UI should load and call your backend endpoints when you upload images.

### Test API using curl (example)

If your backend exposes an endpoint `/predict` that accepts a file field named `file`, you can test with curl:

```bash
curl -X POST -F "file=@/path/to/image.jpg" http://127.0.0.1:5000/predict
```

**Note:** Check `api/main.py` to confirm the exact endpoint and form field name. The API commonly returns a JSON with predicted class and confidence.

---

## Training the model (Jupyter notebooks)

Training steps are provided as notebooks in `training (jupyter notebook)/`. Follow these steps to run them:

1. Activate your Python virtual environment.
2. Install Jupyter if not present:

```bash
pip install jupyter
```

3. Start Jupyter:

```bash
jupyter notebook
```

4. In the web UI, open the notebook files:

   * `tomato dataset Preprocessing.ipynb`
   * `training-cnn-model-tomato-dataset-v1.ipynb`
   * `training_v1.ipynb`

5. Run cells from top to bottom. Typical notebook flow:

   * Load dataset from `tomato_dataset_v1/` (check dataset path in notebook)
   * Perform preprocessing & augmentation
   * Build model (Keras Sequential or Functional API)
   * Compile and train the model (`model.fit(...)`)
   * Save the trained model (`model.save('best_model.h5')`)

**GPU training:** if you have a GPU, make sure TensorFlow GPU is installed and CUDA drivers are correctly configured. Training on CPU is slower but works.

**Tip:** Notebooks are intentionally included to document every preprocessing and model-building step. If something fails, open the notebook cell where it fails and read the error; often it’s a missing package or wrong file path.

---

## Using the pre-trained model

If you have `best_model.h5` locally (this repo ignores it), the backend will likely load it when started. Example pseudocode the backend may use:

```python
from tensorflow import keras
model = keras.models.load_model('best_model.h5')
```

**How to test**:

* Ensure `best_model.h5` is present in the expected path (project root or `saved_models/`).
* Start the backend and use the frontend or curl to request predictions.

If you want to share the model with others without pushing it to GitHub, consider:

* Uploading to a file storage (Google Drive) and adding a script `scripts/download_model.sh` to fetch it, or
* Using Hugging Face Hub or Git LFS (instructions below).

---

## Keeping repository clean & handling large files

This repo uses a `.gitignore` so large datasets and model files are not tracked.

### If you want to keep models in GitHub history (not recommended):

Use **Git LFS** for large files:

```bash
# install git lfs (one-time)
git lfs install
# track h5 files
git lfs track "*.h5"
# commit the new .gitattributes
git add .gitattributes
git commit -m "Track .h5 with Git LFS"
```

**Note:** Git LFS stores large files differently and has bandwidth/storage limits on GitHub. For collaborative or public sharing, consider Hugging Face or cloud storage.

---

## Deployment (short guide)

A couple of common deployment targets:

### Deploy to Google Cloud Run (simple flow)

1. Create a `Dockerfile` that builds your backend (and optionally the frontend static build).
2. Build & submit to Google Cloud Build:

```bash
gcloud builds submit --tag gcr.io/<PROJECT_ID>/tomato-api
```

3. Deploy to Cloud Run:

```bash
gcloud run deploy tomato-api --image gcr.io/<PROJECT_ID>/tomato-api --platform managed --allow-unauthenticated --region <REGION>
```

**Why containerize:** it makes the environment consistent and easier to deploy.

---

## Troubleshooting & common issues

**1. `best_model.h5` or dataset accidentally committed**

* Remove them from tracking but keep local copy:

```bash
git rm --cached best_model.h5
git rm -r --cached tomato_dataset_v1/
git commit -m "Remove large files from tracking"
```

**2. `git push` rejected because remote has README**

* Run:

```bash
git pull origin main --allow-unrelated-histories
git push
```

**3. CRLF / LF warnings on Windows**

* Windows Git prints: `LF will be replaced by CRLF`. This is normal. To avoid it, set `git config --global core.autocrlf true` (Windows) or false on Linux.

**4. Port already in use**

* Change the port or kill the process using it (Windows Task Manager or `lsof -i :5000` on Linux/mac).

**5. Missing dependencies**

* If `ModuleNotFoundError` appears, install the missing package: `pip install <package>` and re-run `pip install -r api/requirements.txt`.

---

## Useful git commands (copy/paste)

```bash
# initialize
git init
# add safe files
git add .gitignore api frontend "training (jupyter notebook)" tomato_pic.jpeg
# commit
git commit -m "Initial commit: code + notebooks; ignore models/datasets"
# connect and push
git remote add origin https://github.com/abdullahzunorain/Tomato-Leaf-Disease-Classification-Using-DL.git
git branch -M main
git push -u origin main
```

Remove large tracked file while keeping local copy:

```bash
git rm --cached best_model.h5
git commit -m "remove best_model from tracking"
```

Track large files with Git LFS:

```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
git commit -m "use git-lfs for model files"
```

---

## License & contact

This project uses the license included in the repository. If you want to contact the maintainer:

* GitHub: `https://github.com/abdullahzunorain`
* Email: `abdullahzunorain2@gmail.com`
* Linkedin: `https://www.linkedin.com/in/abdullahzunorain/`

---

## Final notes

* The notebooks are the canonical source for model design and preprocessing — open them to understand exact model layers, hyperparameters, and preprocessing steps.
* Keep big datasets and models out of Git. Use `saved_models/` locally or a cloud storage and reference download instructions in the repo.

---
