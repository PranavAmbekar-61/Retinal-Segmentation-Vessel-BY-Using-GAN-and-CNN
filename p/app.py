"""
app.py — RetinaSeg Flask Application
Produces 4-panel output: Raw prediction | Filtered | U-Net clean | GAN style
"""

import os, base64, datetime, time, csv, io
import numpy as np
import cv2

from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_login import (LoginManager, login_user, logout_user,
                         login_required, current_user)
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from fpdf import FPDF

from database import (db, User, Patient, ScanHistory,
                      get_user_stats, get_recent_scans,
                      get_paginated_scans, save_scan,
                      ensure_sqlite_columns)

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY']                     = 'retinaseg-secret-key-2025'
app.config['SQLALCHEMY_DATABASE_URI']        = 'sqlite:///retinaseg.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MIN_QUALITY_SCORE']              = 45

CORS(app)
db.init_app(app)
bcrypt        = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login_page'

@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))

# ── Custom loss functions ──────────────────────────────────────────────────────
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f     = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f     = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred, smooth=1e-6):
    return 1 - dice_coef(y_true, y_pred, smooth)

def combined_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

CUSTOM_OBJECTS = {
    'combined_loss': combined_loss,
    'bce_dice_loss': bce_dice_loss,
    'dice_loss':     dice_loss,
    'dice_coef':     dice_coef,
}

# ── Load model ─────────────────────────────────────────────────────────────────
MODEL_PATH = 'model.h5'
GAN_MODEL_PATH = 'gan_model.h5'
model      = None
gan_model  = None

def load_unet():
    global model, gan_model
    if not os.path.exists(MODEL_PATH):
        print(f'WARNING: {MODEL_PATH} not found.')
    else:
        try:
            model = load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
            print(f'U-Net loaded  input_shape={model.input_shape}')
        except Exception as e:
            print(f'ERROR loading U-Net: {e}')
            
    if not os.path.exists(GAN_MODEL_PATH):
        print(f'WARNING: {GAN_MODEL_PATH} not found.')
    else:
        try:
            gan_model = load_model(GAN_MODEL_PATH, compile=False)
            print(f'GAN loaded input_shape={gan_model.input_shape}')
        except Exception as e:
            print(f'ERROR loading GAN: {e}')

load_unet()

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(img_bgr):
    """Green channel + CLAHE + resize 256x256 + normalize."""
    green    = img_bgr[:, :, 1]
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    resized  = cv2.resize(enhanced, (256, 256))
    norm     = resized.astype(np.float32) / 255.0
    return norm.reshape(1, 256, 256, 1)


def analyze_quality(img_bgr):
    """Return simple quality metrics and flags for a retinal image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    flags = []
    score = 100.0

    if blur_var < 80:
        flags.append('blurry')
        score -= 30
    if contrast < 35:
        flags.append('low_contrast')
        score -= 20
    if brightness < 35:
        flags.append('underexposed')
        score -= 20
    if brightness > 220:
        flags.append('overexposed')
        score -= 20

    score = float(max(0.0, min(100.0, score)))
    return {
        'blur_var': blur_var,
        'brightness': brightness,
        'contrast': contrast,
        'score': score,
        'flags': flags,
    }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def suggestion_model(quality_score, vessel_density, confidence_score):
    """Lightweight local model for suggestion text (non-diagnostic)."""
    q = quality_score / 100.0
    d = float(vessel_density)
    c = float(confidence_score)

    # Simple linear model with sigmoid normalization
    risk_score = sigmoid(-1.4 + (2.2 * d) + (1.6 * (1.0 - q)) + (0.8 * (1.0 - c)))

    if risk_score < 0.33:
        band = 'low'
    elif risk_score < 0.66:
        band = 'moderate'
    else:
        band = 'elevated'

    clinical = (
        f"Non-diagnostic summary: vessel density is {d:.2f}, confidence {c:.2f}, "
        f"image quality score {quality_score:.0f}/100. "
        f"Algorithmic signal level: {band}. Consider correlating with clinical context."
    )
    plain = (
        f"Plain-language note: this image looks {'clear' if quality_score >= 70 else 'average' if quality_score >= 50 else 'low quality'}, "
        f"and the vessel pattern looks {band}. This is not a diagnosis; use it as a supportive hint only."
    )
    return {
        'risk_score': float(risk_score),
        'band': band,
        'clinical': clinical,
        'plain': plain,
    }


def model_version_from_path(path):
    if not os.path.exists(path):
        return 'missing'
    ts = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y%m%d-%H%M')
    size = os.path.getsize(path)
    return f"{os.path.basename(path)}@{ts}-{size}"


def build_overlay(img_bgr, pred_raw):
    """Overlay probability heatmap on original image."""
    base = cv2.resize(img_bgr, (256, 256))
    heat = (pred_raw * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base, 0.6, heat, 0.4, 0)
    return img_to_b64(overlay)

# ── Image processing helpers ──────────────────────────────────────────────────
KERNEL3 = np.ones((3, 3), np.uint8)
KERNEL5 = np.ones((5, 5), np.uint8)


def _panel_raw(pred_raw, size):
    """Panel 1: CLAHE-enhanced raw sigmoid — boosts local contrast."""
    gray = (pred_raw * 255).astype(np.uint8)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LANCZOS4)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _panel_filter(pred_raw, size):
    """Panel 2: Binary threshold + morph open — removes noise specks."""
    binary = (pred_raw > 0.35).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, KERNEL3)
    binary = cv2.resize(binary, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _panel_unet(pred_raw, size):
    """Panel 3: Median blur + morph close — fills vessel gaps."""
    binary = (pred_raw > 0.35).astype(np.uint8) * 255
    cleaned = cv2.medianBlur(binary, 5)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, KERNEL3)
    cleaned = cv2.resize(cleaned, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


def _panel_gan(pred_raw, gan_pred, size):
    """Panel 4: GAN min-max normalized then binary-thresholded for crisp vessels."""
    if gan_pred is not None:
        gan_norm = (gan_pred - gan_pred.min()) / (gan_pred.max() - gan_pred.min() + 1e-8)
        # Binary threshold at median to get clean black-and-white vessel map
        threshold = float(np.median(gan_norm))
        # The GAN output appears inverted (vessels darker than the retina), so we use < threshold
        gan_bin = (gan_norm < threshold).astype(np.uint8) * 255
        gan_bin = cv2.medianBlur(gan_bin, 5)
        gan_bin = cv2.morphologyEx(gan_bin, cv2.MORPH_CLOSE, KERNEL3)
        
        # Apply a circular mask to ensure the background outside the retina remains black
        H, W = gan_bin.shape
        Y, X = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((X - W/2)**2 + (Y - H/2)**2)
        mask = (dist_from_center <= (W/2 - 2)).astype(np.uint8)
        gan_bin = gan_bin * mask
        
        gan_bin = cv2.resize(gan_bin, (size, size), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(gan_bin, cv2.COLOR_GRAY2BGR)
    else:
        # Fallback: use U-Net cleaned output
        binary = (pred_raw > 0.35).astype(np.uint8) * 255
        cleaned = cv2.medianBlur(binary, 5)
        cleaned = cv2.resize(cleaned, (size, size), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


# ── 4-panel image builder ──────────────────────────────────────────────────────
def build_4panel(img_bgr, pred_raw, gan_pred=None):
    """
    Produces the 4-panel output at high resolution:
      Panel 1 (top-left)  : Image generated by the model  — CLAHE-enhanced sigmoid
      Panel 2 (top-right) : Filtering the image           — morphologically cleaned binary
      Panel 3 (bot-left)  : U-Net Model                   — median blur + morph close
      Panel 4 (bot-right) : Traditional GAN Model         — normalized + binary threshold
    """
    SIZE    = 512        # high-res panels
    PAD     = 56         # border padding
    LABEL_H = 40         # height for label below each panel
    GAP     = 24         # gap between panels

    PW = SIZE
    PH = SIZE

    W = PAD + PW + GAP + PW + PAD
    H = PAD + PH + LABEL_H + GAP + PH + LABEL_H + PAD

    canvas = np.zeros((H, W, 3), dtype=np.uint8)   # pure black background

    # ── Build each panel ────────────────────────────────────────────────────────
    raw_bgr    = _panel_raw(pred_raw, SIZE)
    binary_bgr = _panel_filter(pred_raw, SIZE)
    unet_bgr   = _panel_unet(pred_raw, SIZE)
    gan_bgr    = _panel_gan(pred_raw, gan_pred, SIZE)

    # ── Place panels on canvas ──────────────────────────────────────────────────
    x1, y1 = PAD, PAD
    x2, y2 = PAD + PW + GAP, PAD
    x3, y3 = PAD, PAD + PH + LABEL_H + GAP
    x4, y4 = PAD + PW + GAP, PAD + PH + LABEL_H + GAP

    canvas[y1:y1+PH, x1:x1+PW] = raw_bgr
    canvas[y2:y2+PH, x2:x2+PW] = binary_bgr
    canvas[y3:y3+PH, x3:x3+PW] = unet_bgr
    canvas[y4:y4+PH, x4:x4+PW] = gan_bgr

    # ── Labels ─────────────────────────────────────────────────────────────────
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72
    thickness  = 1
    color      = (210, 210, 210)

    labels = [
        ('Image generated by the model', x1, y1 + PH + 26),
        ('Filtering the image',          x2, y2 + PH + 26),
        ('U-Net Model',                  x3, y3 + PH + 26),
        ('Traditional GAN Model',        x4, y4 + PH + 26),
    ]

    for text, lx, ly in labels:
        tw = cv2.getTextSize(text, font, font_scale, thickness)[0][0]
        cx_txt = lx + PW // 2 - tw // 2
        cv2.putText(canvas, text, (cx_txt, ly),
                    font, font_scale, color, thickness, cv2.LINE_AA)

    # ── Encode to base64 PNG ────────────────────────────────────────────────────
    _, buf = cv2.imencode('.png', canvas)
    return base64.b64encode(buf).decode('utf-8')


def img_to_b64(img_bgr):
    """Encode a BGR image to base64 PNG."""
    _, buf = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buf).decode('utf-8')


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/')
def root():
    return redirect(url_for('dashboard') if current_user.is_authenticated
                    else url_for('landing'))

@app.route('/home')
def landing():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        data  = request.get_json()
        email = data.get('email', '').strip().lower()
        pwd   = data.get('password', '')
        user  = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, pwd):
            login_user(user, remember=True)
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        return jsonify({'success': False, 'error': 'Incorrect email or password.'})
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup_page():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        data  = request.get_json()
        name  = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        pwd   = data.get('password', '')
        role  = data.get('role', 'doctor')
        if not name or not email or not pwd:
            return jsonify({'success': False, 'error': 'All fields are required.'})
        if len(pwd) < 6:
            return jsonify({'success': False, 'error': 'Password must be at least 6 characters.'})
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'error': 'Email already registered.'})
        hashed = bcrypt.generate_password_hash(pwd).decode('utf-8')
        user   = User(name=name, email=email, password=hashed, role=role)
        db.session.add(user)
        db.session.commit()
        login_user(user, remember=True)
        return jsonify({'success': True, 'redirect': url_for('dashboard')})
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))

# ══════════════════════════════════════════════════════════════════════════════
# PROTECTED ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/dashboard')
@login_required
def dashboard():
    stats  = get_user_stats(current_user.id)
    recent = get_recent_scans(current_user.id, limit=5)
    return render_template('dashboard.html', user=current_user, active='dashboard',
                           total=stats['total'], done=stats['completed'], recent=recent)

@app.route('/upload')
@login_required
def upload_page():
    patients = Patient.query.filter_by(user_id=current_user.id).order_by(Patient.name).all()
    return render_template('upload.html', user=current_user, active='upload', patients=patients)

@app.route('/history')
@login_required
def history_page():
    page    = request.args.get('page', 1, type=int)
    records = get_paginated_scans(current_user.id, page=page)
    return render_template('history.html', user=current_user, active='history',
                           records=records)

@app.route('/profile')
@login_required
def profile_page():
    stats = get_user_stats(current_user.id)
    return render_template('profile.html', user=current_user, active='profile',
                           total=stats['total'])

@app.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    data = request.get_json()
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Name cannot be empty.'})
    current_user.name = name
    db.session.commit()
    return jsonify({'success': True})

@app.route('/profile/password', methods=['POST'])
@login_required
def change_password():
    data    = request.get_json()
    old_pwd = data.get('old_password', '')
    new_pwd = data.get('new_password', '')
    if not bcrypt.check_password_hash(current_user.password, old_pwd):
        return jsonify({'success': False, 'error': 'Current password is incorrect.'})
    if len(new_pwd) < 6:
        return jsonify({'success': False, 'error': 'New password must be at least 6 characters.'})
    current_user.password = bcrypt.generate_password_hash(new_pwd).decode('utf-8')
    db.session.commit()
    return jsonify({'success': True})

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT API  — returns 4 individual panels + 1 combined 4-panel image
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Ensure model.h5 is in the project root.'}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400
    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'Empty filename.'}), 400

    patient_id_str = request.form.get('patient_id', '')
    patient_id = int(patient_id_str) if patient_id_str and patient_id_str.isdigit() else None

    try:
        raw      = file.read()
        arr      = np.frombuffer(raw, np.uint8)
        img_bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'error': 'Could not decode image.'}), 400

        quality = analyze_quality(img_bgr)

        # Optional Pre-processing from form
        do_clahe = request.form.get('clahe', 'true') == 'true'
        if not do_clahe:
            green = img_bgr[:, :, 1]
            resized = cv2.resize(green, (256, 256))
            norm = resized.astype(np.float32) / 255.0
            processed = norm.reshape(1, 256, 256, 1)
        else:
            processed  = preprocess(img_bgr)

        t0         = time.time()
        pred       = model.predict(processed, verbose=0)[0, :, :, 0]
        gan_pred   = None
        if gan_model is not None:
            gan_pred = gan_model.predict(processed, verbose=0)[0, :, :, 0]
            
        elapsed_ms = int((time.time() - t0) * 1000)

        # ── 4 individual panels (same clarity pipeline as build_4panel) ──────────
        PANEL_SIZE = 512

        # Panel 1 — CLAHE-enhanced raw sigmoid
        raw_gray_256 = (pred * 255).astype(np.uint8)
        clahe_p1 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        raw_gray = clahe_p1.apply(raw_gray_256)
        raw_gray = cv2.resize(raw_gray, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_LANCZOS4)

        # Panel 2 — binary threshold + morph open
        binary_256 = (pred > 0.35).astype(np.uint8) * 255
        binary_256 = cv2.morphologyEx(binary_256, cv2.MORPH_OPEN, KERNEL3)
        binary = cv2.resize(binary_256, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST)

        # Panel 3 — median blur + morph close
        cleaned_256 = cv2.medianBlur((pred > 0.35).astype(np.uint8) * 255, 5)
        cleaned_256 = cv2.morphologyEx(cleaned_256, cv2.MORPH_CLOSE, KERNEL3)
        cleaned = cv2.resize(cleaned_256, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST)

        # Panel 4 — GAN normalized + binary threshold (no red dot)
        gan_bgr = _panel_gan(pred, gan_pred, PANEL_SIZE)

        # Original resized
        orig_small = cv2.resize(img_bgr, (256, 256))

        # ── Combined 4-panel image ─────────────────────────────────────────────
        panel_b64 = build_4panel(orig_small, pred, gan_pred)

        # ── Overlay heatmap ───────────────────────────────────────────────────
        overlay_b64 = build_overlay(orig_small, pred)

        # ── AI suggestions + metrics ─────────────────────────────────────────-
        vessel_density = float((pred > 0.35).mean())
        confidence_score = float(np.clip(pred.mean() * 1.2, 0.0, 1.0))
        suggestion = suggestion_model(quality['score'], vessel_density, confidence_score)
        model_version = model_version_from_path(MODEL_PATH)
        gan_version = model_version_from_path(GAN_MODEL_PATH) if gan_model is not None else None

        # ── Individual base64 panels ───────────────────────────────────────────
        def g2b64(gray_img):
            bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            return img_to_b64(bgr)

        # Save to history
        record = save_scan(user_id=current_user.id, filename=file.filename,
                   file_size=f'{len(raw)/1024:.1f} KB',
                   inference_ms=elapsed_ms, status='completed', patient_id=patient_id,
                   model_version=model_version,
                   gan_version=gan_version,
                   quality_score=quality['score'],
                   quality_flags=','.join(quality['flags']) if quality['flags'] else None,
                   vessel_density=vessel_density,
                   confidence_score=confidence_score,
                   suggestion_clinical=suggestion['clinical'],
                   suggestion_plain=suggestion['plain'])

        return jsonify({
            'success':    True,
            'elapsed_ms': elapsed_ms,
            # Combined 4-panel (main display)
            'panel_4':    panel_b64,
            # Individual panels for download
            'p1_raw':     g2b64(raw_gray),
            'p2_filter':  g2b64(binary),
            'p3_unet':    g2b64(cleaned),
            'p4_gan':     img_to_b64(gan_bgr),
            'original':   img_to_b64(orig_small),
            'overlay':    overlay_b64,
            'scan_id':    record.id,
            'quality': {
                'score': quality['score'],
                'flags': quality['flags'],
                'blur_var': quality['blur_var'],
                'brightness': quality['brightness'],
                'contrast': quality['contrast'],
            },
            'metrics': {
                'vessel_density': vessel_density,
                'confidence_score': confidence_score,
                'model_version': model_version,
                'gan_version': gan_version,
            },
            'suggestions': {
                'clinical': suggestion['clinical'],
                'plain': suggestion['plain'],
                'band': suggestion['band'],
                'risk_score': suggestion['risk_score'],
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

@app.route('/debug/model')
def debug_model():
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    import numpy as np
    dummy = np.random.rand(1, 256, 256, 1).astype(np.float32)
    pred  = model.predict(dummy, verbose=0)
    return jsonify({
        'input_shape':            str(model.input_shape),
        'output_shape':           str(model.output_shape),
        'pred_min':               float(pred.min()),
        'pred_max':               float(pred.max()),
        'pixels_above_threshold': int((pred > 0.35).sum()),
    })

# ══════════════════════════════════════════════════════════════════════════════
# EXTRA FEATURES: Admin, Patients, Export
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/patients', methods=['GET', 'POST'])
@login_required
def patients_page():
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('name', '').strip()
        age = data.get('age')
        gender = data.get('gender', '')
        if name:
            p = Patient(user_id=current_user.id, name=name, age=age, gender=gender)
            db.session.add(p)
            db.session.commit()
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Name is required'})
    
    my_patients = Patient.query.filter_by(user_id=current_user.id).order_by(Patient.created_at.desc()).all()
    return render_template('patients.html', user=current_user, active='patients', patients=my_patients)

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        return "Access Denied", 403
    users = User.query.all()
    scans = ScanHistory.query.count()
    return render_template('admin.html', user=current_user, active='admin', users=users, total_scans=scans)

@app.route('/export/csv')
@login_required
def export_csv():
    scans = ScanHistory.query.filter_by(user_id=current_user.id).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'ID', 'Filename', 'Patient', 'Date', 'Status', 'Inference MS',
        'Quality Score', 'Vessel Density', 'Confidence', 'Tags', 'Follow-up'
    ])
    for s in scans:
        p_name = s.patient.name if s.patient else "Unassigned"
        writer.writerow([
            s.id, s.filename, p_name, s.scan_date, s.status, s.inference_ms,
            s.quality_score, s.vessel_density, s.confidence_score,
            s.diagnosis_tags, s.follow_up_date
        ])
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='scans.csv')

@app.route('/export/pdf/<int:scan_id>')
@login_required
def export_pdf(scan_id):
    scan = ScanHistory.query.get_or_404(scan_id)
    if scan.user_id != current_user.id and current_user.role != 'admin':
        return "Access Denied", 403
        
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="RetinaSeg Medical Report", ln=1, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Doctor: {scan.user.name}", ln=1)
    p_name = scan.patient.name if scan.patient else "Unassigned"
    pdf.cell(200, 10, txt=f"Patient: {p_name}", ln=1)
    pdf.cell(200, 10, txt=f"Scan Date: {scan.scan_date.strftime('%Y-%m-%d %H:%M')}", ln=1)
    pdf.cell(200, 10, txt=f"Filename: {scan.filename}", ln=1)
    pdf.cell(200, 10, txt=f"Status: {scan.status}", ln=1)
    if scan.quality_score is not None:
        pdf.cell(200, 10, txt=f"Quality Score: {scan.quality_score:.0f}/100", ln=1)
    if scan.vessel_density is not None:
        pdf.cell(200, 10, txt=f"Vessel Density: {scan.vessel_density:.2f}", ln=1)
    if scan.confidence_score is not None:
        pdf.cell(200, 10, txt=f"Confidence: {scan.confidence_score:.2f}", ln=1)
    if scan.diagnosis_tags:
        pdf.cell(200, 10, txt=f"Tags: {scan.diagnosis_tags}", ln=1)
    if scan.follow_up_date:
        pdf.cell(200, 10, txt=f"Follow-up: {scan.follow_up_date.strftime('%Y-%m-%d')}", ln=1)
    
    pdf.ln(20)
    pdf.cell(200, 10, txt="Clinical Notes:", ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(30)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)
    if scan.suggestion_clinical:
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 7, txt=f"AI Suggestion (Clinical): {scan.suggestion_clinical}")
    if scan.notes:
        pdf.multi_cell(0, 7, txt=f"Notes: {scan.notes}")
    
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return send_file(io.BytesIO(pdf_bytes), mimetype='application/pdf', as_attachment=True, download_name=f'report_{scan.id}.pdf')


@app.route('/scan/meta/<int:scan_id>', methods=['POST'])
@login_required
def update_scan_meta(scan_id):
    scan = ScanHistory.query.get_or_404(scan_id)
    if scan.user_id != current_user.id and current_user.role != 'admin':
        return jsonify({'success': False, 'error': 'Access denied.'}), 403

    data = request.get_json() or {}
    notes = (data.get('notes') or '').strip()
    tags = (data.get('diagnosis_tags') or '').strip()
    follow_up_date = (data.get('follow_up_date') or '').strip()

    scan.notes = notes if notes else None
    scan.diagnosis_tags = tags if tags else None
    if follow_up_date:
        try:
            scan.follow_up_date = datetime.datetime.strptime(follow_up_date, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid follow-up date.'}), 400
    else:
        scan.follow_up_date = None

    db.session.commit()
    return jsonify({'success': True})

# ── Init DB ────────────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()
    ensure_sqlite_columns()
    print('Database ready -> instance/retinaseg.db')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
