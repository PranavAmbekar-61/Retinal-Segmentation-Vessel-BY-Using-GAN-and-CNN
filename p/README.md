# RetinaSeg – Retinal Vessel Segmentation System

Full-stack web application for retinal vessel segmentation using a trained U-Net model.

## Project structure

```
retinal_segmentation/
├── app.py               Flask app — routes, auth, predict API
├── database.py          SQLAlchemy models (User, ScanHistory) + helper queries
├── db_guide.py          Database utility script (inspect, seed, reset, export)
├── model.h5             Trained U-Net model (included)
├── requirements.txt     Python dependencies
├── README.md
├── instance/
│   └── retinaseg.db     SQLite database (auto-created on first run)
├── static/
│   ├── css/style.css    All styles + dark mode + responsive
│   └── js/
│       ├── theme.js     Dark/light toggle
│       └── upload.js    Upload, API call, download
└── templates/
    ├── base.html        Base HTML template
    ├── layout.html      Sidebar shell (shared by inner pages)
    ├── landing.html     Public home page
    ├── login.html       Sign in
    ├── signup.html      Create account
    ├── dashboard.html   Stats + recent scans
    ├── upload.html      Upload + result view
    ├── history.html     Paginated scan history
    └── profile.html     Update name + change password
```

## Setup & run

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Flask
python app.py
```

Open: http://127.0.0.1:5000

Database is created automatically at instance/retinaseg.db on first run.

---

## Database guide (db_guide.py)

Use this utility script to manage your database without opening SQLite manually.

```bash
python db_guide.py info       # show table summaries and row counts
python db_guide.py users      # list all registered users
python db_guide.py scans      # list all scan history records
python db_guide.py seed       # create a demo doctor account for testing
python db_guide.py reset      # drop and recreate all tables (deletes data)
python db_guide.py export     # export both tables to CSV files
```

Demo account created by seed command:
  Email    : doctor@retinaseg.com
  Password : demo1234
  Role     : doctor

---

## Database tables

### User
| Column     | Type     | Notes                          |
|------------|----------|--------------------------------|
| id         | Integer  | Primary key                    |
| name       | String   | Display name                   |
| email      | String   | Unique login email             |
| password   | String   | Bcrypt hash                    |
| role       | String   | 'doctor' or 'admin'            |
| created_at | DateTime | Auto-set to UTC now            |

### ScanHistory
| Column      | Type     | Notes                          |
|-------------|----------|--------------------------------|
| id          | Integer  | Primary key                    |
| user_id     | Integer  | Foreign key → User.id          |
| filename    | String   | Original uploaded filename     |
| scan_date   | DateTime | Auto-set to UTC now            |
| status      | String   | 'completed' or 'failed'        |
| file_size   | String   | e.g. '2.4 KB'                  |
| inference_ms| Integer  | Model run time in milliseconds |

Relationship: User 1 → many ScanHistory

---

## Inspect database in VS Code

Install extension: SQLite Viewer (by Florian Klampfer)
Then open: instance/retinaseg.db

Or via terminal:
  sqlite3 instance/retinaseg.db
  .tables
  SELECT * FROM user;
  SELECT * FROM scan_history;
  .quit

---

## Switch to MySQL or PostgreSQL (deployment)

Change one line in app.py:
  'sqlite:///retinaseg.db'                               (current)
  'mysql+pymysql://user:pass@localhost/retinaseg'        (MySQL)
  'postgresql://user:pass@localhost/retinaseg'           (PostgreSQL)

All model code stays the same.
