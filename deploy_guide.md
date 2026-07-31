# Guide: How to Deploy and Make the Web Dashboard Public 🌐

This guide outlines 3 simple methods to host and deploy your **PGCB Electricity Demand Forecasting Web Dashboard** so that evaluators, research supervisors, or public users can interact with it online.

---

## Option 1: Streamlit Community Cloud (Recommended & Free) 🚀

This is the standard, free method recommended for Google Form submissions and academic portfolios.

### Steps:
1. **Push Code to GitHub**:
   - Make sure your project repository (containing `RA Task/dashboard/app.py`, `RA Task/requirements.txt`, `RA Task/outputs/`, `RA Task/data/`) is pushed to a **GitHub repository** (Public or Private).
2. **Log into Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/).
   - Click **"Continue with GitHub"** to authorize your account.
3. **Deploy New App**:
   - Click the **"New app"** button.
   - Select your GitHub Repository and Branch (`main` or `master`).
   - Main file path: `RA Task/dashboard/app.py` (or `dashboard/app.py`).
4. **Click Deploy!**
   - Streamlit will automatically install dependencies from `requirements.txt` and launch your dashboard in 1–2 minutes.
   - You will get a live public URL (e.g. `https://pgcb-power-forecasting.streamlit.app`).

---

## Option 2: Instant Public Tunnel via Ngrok / LocalTunnel (Fastest for Video Demo) ⚡

If you want an **instant live link** without committing code to GitHub immediately (for example, to record a 2-minute video demo or show a live preview):

### Steps:
1. **Launch Streamlit Locally**:
   Open terminal and run:
   ```bash
   streamlit run "RA Task/dashboard/app.py"
   ```
   (This runs locally at `http://localhost:8501`).

2. **Expose with LocalTunnel (No Installation Needed)**:
   In a new terminal window, run:
   ```bash
   npx localtunnel --port 8501
   ```
   - LocalTunnel will generate an instant public HTTPS link, e.g.:
     `https://bangladesh-grid-forecast.loca.lt`
   - Share this link with anyone to let them test your app live while Streamlit is running on your machine.

---

## Option 3: Hugging Face Spaces (Free Alternative) 🤗

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Click **"New Space"**.
3. Set **Space Name**: `pgcb-electricity-forecasting`.
4. Select **SDK**: `Streamlit`.
5. Upload your `app.py`, `requirements.txt`, `outputs/`, and `data/` files.
6. Hugging Face will automatically build and host the space at `https://huggingface.co/spaces/<your-username>/pgcb-electricity-forecasting`.
