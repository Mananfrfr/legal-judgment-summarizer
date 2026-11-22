# Deploying Legal Judgment Summarizer

This repository contains a full **Dockerfile** and deployment configurations for **Railway** and **Render**.

> **Important Resource Requirement**: InLegalBERT and the BiLSTM selector require **at least 1.5 GB to 2 GB of RAM** for PyTorch CPU inference. Do not use 512 MB free tiers as the container will encounter an out-of-memory (OOM) error.

---

## Option 1: Railway (Recommended)

Railway gives up to 8 GB of RAM and automatically builds the Dockerfile.

1. Go to [railway.app](https://railway.app/) and sign in with GitHub.
2. Click **+ New Project** &rarr; **Deploy from GitHub repo**.
3. Select your repository: `Mananfrfr/legal-judgment-summarizer`.
4. Railway will automatically detect `Dockerfile` and `railway.json`.
5. Under **Variables** in your project settings:
   - `APP_ALLOW_UNAUTHENTICATED`: `true` (or set `APP_PASSWORD` with 12+ characters if you want password protection)
   - Railway automatically sets the `PORT` variable.
6. Click **Generate Domain** under **Networking** to get your public HTTPS URL (e.g. `https://legal-judgment-summarizer.up.railway.app`).

---

## Option 2: Render

1. Go to [render.com](https://render.com/) and sign in.
2. Click **New +** &rarr; **Web Service**.
3. Connect your GitHub repository `Mananfrfr/legal-judgment-summarizer`.
4. Select **Docker** as the runtime.
5. Choose **Instance Type**:
   - Select **Starter** (2 GB RAM) or higher. (The free tier is 512 MB and will run out of memory during InLegalBERT inference).
6. Under **Advanced / Environment Variables**:
   - `APP_ALLOW_UNAUTHENTICATED`: `true`
7. Click **Create Web Service**. Render will build the Docker container and deploy the app at `https://<service-name>.onrender.com`.

---

## Running Locally with Docker

To test the container locally before deploying:

```bash
docker build -t legal-judgment-summarizer .
docker run -p 7861:7861 legal-judgment-summarizer
```

Open `http://localhost:7861` in your browser.
