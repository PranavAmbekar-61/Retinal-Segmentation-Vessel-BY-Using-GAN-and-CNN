# 👁️ Retinal Vessel Segmentation System

A professional, end-to-end medical application for analyzing retinal fundus images. It uses a **U-Net** deep learning model alongside a **GAN** (Generative Adversarial Network) architecture to accurately segment blood vessels, aiding in the early detection of diseases like diabetic retinopathy.

The application includes a fully-featured dashboard, patient management, admin controls, and professional PDF/CSV export capabilities.

---

## 🚀 Step-by-Step Guide to Run the Application

Follow these steps exactly to set up the environment and run the system locally on your machine.

### Step 1: Navigate to the Source Directory
All application code is stored within the `p/` directory to avoid Windows file path length limits during dependency installation.

Open your terminal or command prompt and navigate to the project folder:
```bash
cd p
```

### Step 2: Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies securely.
```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

### Step 3: Install Dependencies
With your virtual environment active, install the required packages.
```bash
pip install -r requirements.txt
```

### Step 4: Generate the GAN Model
Before running the application, you need to generate the machine learning models. The U-Net model (`model.h5`) is already included, but you must train the local GAN model (`gan_model.h5`).
```bash
python train_gan.py
```
*Wait for the script to finish executing. It will output `Model saved to gan_model.h5!`*

### Step 5: Initialize the Database and Demo Accounts
Set up the SQLite database and seed it with demo accounts so you can test both the Doctor and Admin views.
```bash
python db_guide.py seed
```

### Step 6: Start the Application Server
Run the Flask backend server.
```bash
python app.py
```
If successful, you will see output indicating the models are loaded and the server is running on `http://127.0.0.1:5000`.

---

## 💻 How to Use the Application

Once the server is running, open your web browser and go to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### 🔐 Default Login Credentials

You can use the accounts generated in Step 5 to log in:

**Doctor Account (Clinical Features):**
- **Email**: `doctor@retinaseg.com`
- **Password**: `demo1234`
- *Capabilities: Upload images, manage patients, download PDFs, view history.*

**Administrator Account (System Features):**
- **Email**: `admin@retinaseg.com`
- **Password**: `admin1234`
- *Capabilities: Access the Admin Dashboard to view all registered users and global platform statistics.*

---

## ✨ Key Features Implemented
1. **Patient Management (`/patients`)**: Securely add and track patients before uploading their scans.
2. **Dynamic Machine Learning Preprocessing**: Toggle CLAHE (Contrast Limited Adaptive Histogram Equalization) directly from the upload UI.
3. **Multi-Model Inference**: The system uses a U-Net model and a custom GAN architecture to output four distinct comparative analysis panels.
4. **Professional Medical Reports**: Instantly generate and download PDF clinical reports from any scan in your history.
5. **Data Export**: Export your entire scan history to CSV for research and analysis.
6. **Admin Dashboard**: Secure, role-based access for system administrators to monitor platform health.