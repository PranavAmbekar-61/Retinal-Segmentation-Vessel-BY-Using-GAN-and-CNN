"""
db_guide.py — RetinaSeg Database Utility
==========================================
Run this script directly from VS Code terminal to inspect,
seed, reset, or export your SQLite database.

Usage:
    python db_guide.py info        → show table summaries
    python db_guide.py users       → list all users
    python db_guide.py scans       → list all scan records
    python db_guide.py seed        → create a demo doctor account
    python db_guide.py reset       → drop and recreate all tables (WARNING: deletes data)
    python db_guide.py export      → export both tables to CSV files
"""

import sys
import os
import csv
import datetime

# Make sure we can import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from database import db, User, ScanHistory


def cmd_info():
    """Show row counts and schema summary for all tables."""
    with app.app_context():
        user_count = User.query.count()
        scan_count = ScanHistory.query.count()

        print("\n" + "="*50)
        print("  RetinaSeg — Database Summary")
        print("="*50)
        print(f"\n  Database file : instance/retinaseg.db")
        print(f"\n  Table: user")
        print(f"    Rows        : {user_count}")
        print(f"    Columns     : id, name, email, password, role, created_at")
        print(f"\n  Table: scan_history")
        print(f"    Rows        : {scan_count}")
        print(f"    Columns     : id, user_id, filename, scan_date,")
        print(f"                  status, file_size, inference_ms")
        print(f"\n  Relationship  : user.id → scan_history.user_id (1 : N)")
        print("="*50 + "\n")


def cmd_users():
    """Print all users (without passwords)."""
    with app.app_context():
        users = User.query.order_by(User.id).all()
        if not users:
            print("\n  No users found. Run: python db_guide.py seed\n")
            return
        print(f"\n{'ID':<5} {'Name':<25} {'Email':<35} {'Role':<10} {'Scans':<6} {'Joined'}")
        print("-"*95)
        for u in users:
            scan_count = ScanHistory.query.filter_by(user_id=u.id).count()
            joined = u.created_at.strftime('%d %b %Y')
            print(f"{u.id:<5} {u.name:<25} {u.email:<35} {u.role:<10} {scan_count:<6} {joined}")
        print()


def cmd_scans():
    """Print all scan history records."""
    with app.app_context():
        scans = ScanHistory.query.order_by(ScanHistory.scan_date.desc()).all()
        if not scans:
            print("\n  No scan records found yet.\n")
            return
        print(f"\n{'ID':<5} {'User':<5} {'Filename':<30} {'Date':<22} {'Status':<12} {'Size':<10} {'ms'}")
        print("-"*100)
        for sc in scans:
            date = sc.scan_date.strftime('%d %b %Y, %H:%M')
            ms   = str(sc.inference_ms) if sc.inference_ms else '—'
            sz   = sc.file_size or '—'
            print(f"{sc.id:<5} {sc.user_id:<5} {sc.filename:<30} {date:<22} {sc.status:<12} {sz:<10} {ms}")
        print()


def cmd_seed():
    """Insert a demo doctor account (skips if email already exists)."""
    from flask_bcrypt import Bcrypt
    bcrypt = Bcrypt(app)

    with app.app_context():
        demo_email = 'doctor@retinaseg.com'
        if User.query.filter_by(email=demo_email).first():
            print(f"\n  Demo account already exists: {demo_email}\n")
            return

        hashed = bcrypt.generate_password_hash('demo1234').decode('utf-8')
        user = User(
            name     = 'Dr. Demo',
            email    = demo_email,
            password = hashed,
            role     = 'doctor',
        )
        db.session.add(user)

        # Add two dummy scan history records
        for i, fname in enumerate(['sample_retina_01.jpg', 'sample_retina_02.png'], 1):
            scan = ScanHistory(
                user_id      = user.id if user.id else 1,
                filename     = fname,
                status       = 'completed',
                file_size    = f'{1.2 * i:.1f} MB',
                inference_ms = 280 + i * 45,
                scan_date    = datetime.datetime.utcnow() - datetime.timedelta(days=i),
            )
            db.session.add(scan)

        db.session.commit()
        print(f"\n  Demo account created:")
        print(f"    Email    : {demo_email}")
        print(f"    Password : demo1234")
        print(f"    Role     : doctor")
        print(f"    Scans    : 2 sample records added\n")


def cmd_reset():
    """Drop all tables and recreate them (DELETES ALL DATA)."""
    confirm = input("\n  WARNING: This will delete ALL data. Type YES to confirm: ")
    if confirm.strip() != 'YES':
        print("  Cancelled.\n")
        return
    with app.app_context():
        db.drop_all()
        db.create_all()
        print("\n  Database reset complete. All tables recreated (empty).\n")


def cmd_export():
    """Export User and ScanHistory tables to CSV files."""
    with app.app_context():
        # Export users
        users = User.query.order_by(User.id).all()
        with open('export_users.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id','name','email','role','created_at'])
            for u in users:
                writer.writerow([u.id, u.name, u.email, u.role,
                                 u.created_at.strftime('%Y-%m-%d %H:%M:%S')])
        print(f"\n  Exported {len(users)} user(s) → export_users.csv")

        # Export scans
        scans = ScanHistory.query.order_by(ScanHistory.id).all()
        with open('export_scans.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id','user_id','filename','scan_date',
                             'status','file_size','inference_ms'])
            for sc in scans:
                writer.writerow([sc.id, sc.user_id, sc.filename,
                                 sc.scan_date.strftime('%Y-%m-%d %H:%M:%S'),
                                 sc.status, sc.file_size, sc.inference_ms])
        print(f"  Exported {len(scans)} scan(s)  → export_scans.csv\n")


# ── Entry point ────────────────────────────────────────────────────────────────
COMMANDS = {
    'info':   cmd_info,
    'users':  cmd_users,
    'scans':  cmd_scans,
    'seed':   cmd_seed,
    'reset':  cmd_reset,
    'export': cmd_export,
}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("\nUsage: python db_guide.py <command>")
        print("Commands:", ', '.join(COMMANDS.keys()))
        print()
    else:
        COMMANDS[sys.argv[1]]()
