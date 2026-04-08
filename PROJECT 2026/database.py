"""
database.py — RetinaSeg Database Models
========================================
All SQLAlchemy models are defined here.
Flask-SQLAlchemy is initialised in app.py and imported here via db.

Tables:
  - User         : registered accounts (Doctor / Admin)
  - ScanHistory  : one record per image upload + segmentation run

Relationship:
  User  1 ──< ScanHistory  (one user → many scans)
"""

import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


# ── User ───────────────────────────────────────────────────────────────────────
class User(db.Model, UserMixin):
    """
    Stores every registered account.

    Fields:
        id         – primary key, auto-increment integer
        name       – display name, max 100 chars
        email      – unique login identifier, max 150 chars
        password   – bcrypt hash (never store plain text)
        role       – 'doctor' or 'admin'
        created_at – UTC timestamp of registration (auto-set)
        scans      – back-reference to all ScanHistory rows for this user
    """
    __tablename__ = 'user'

    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(100), nullable=False)
    email      = db.Column(db.String(150), unique=True, nullable=False)
    password   = db.Column(db.String(200), nullable=False)
    role       = db.Column(db.String(20),  nullable=False, default='doctor')
    created_at = db.Column(db.DateTime,    nullable=False,
                           default=datetime.datetime.utcnow)

    scans = db.relationship('ScanHistory', backref='user',
                            lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<User id={self.id} email={self.email} role={self.role}>'

    def to_dict(self):
        """Return a safe dictionary (no password)."""
        return {
            'id':         self.id,
            'name':       self.name,
            'email':      self.email,
            'role':       self.role,
            'created_at': self.created_at.strftime('%d %b %Y, %H:%M'),
            'total_scans': len(self.scans),
        }


# ── ScanHistory ────────────────────────────────────────────────────────────────
class ScanHistory(db.Model):
    """
    One row is inserted every time a user uploads an image
    and the segmentation pipeline completes successfully.

    Fields:
        id           – primary key, auto-increment integer
        user_id      – foreign key → User.id
        filename     – original uploaded filename (e.g. 'retina_001.jpg')
        scan_date    – UTC timestamp when the scan was run (auto-set)
        status       – 'completed' or 'failed'
        file_size    – human-readable size string (e.g. '2.4 KB')
        inference_ms – time taken by model.predict() in milliseconds
    """
    __tablename__ = 'scan_history'

    id           = db.Column(db.Integer,  primary_key=True)
    user_id      = db.Column(db.Integer,  db.ForeignKey('user.id'),
                             nullable=False)
    filename     = db.Column(db.String(200), nullable=False)
    scan_date    = db.Column(db.DateTime,    nullable=False,
                             default=datetime.datetime.utcnow)
    status       = db.Column(db.String(30),  nullable=False,
                             default='completed')
    file_size    = db.Column(db.String(20),  nullable=True)
    inference_ms = db.Column(db.Integer,     nullable=True)

    def __repr__(self):
        return (f'<ScanHistory id={self.id} user_id={self.user_id} '
                f'file={self.filename} status={self.status}>')

    def to_dict(self):
        return {
            'id':           self.id,
            'user_id':      self.user_id,
            'filename':     self.filename,
            'scan_date':    self.scan_date.strftime('%d %b %Y, %H:%M'),
            'status':       self.status,
            'file_size':    self.file_size,
            'inference_ms': self.inference_ms,
        }


# ── Helper queries ─────────────────────────────────────────────────────────────
def get_user_stats(user_id):
    """
    Return a dict with total, completed, and pending scan counts
    for a given user_id. Used by the dashboard route.
    """
    total     = ScanHistory.query.filter_by(user_id=user_id).count()
    completed = ScanHistory.query.filter_by(
                    user_id=user_id, status='completed').count()
    pending   = total - completed
    return {'total': total, 'completed': completed, 'pending': pending}


def get_recent_scans(user_id, limit=5):
    """Return the N most recent scans for a user, newest first."""
    return (ScanHistory.query
            .filter_by(user_id=user_id)
            .order_by(ScanHistory.scan_date.desc())
            .limit(limit)
            .all())


def get_paginated_scans(user_id, page=1, per_page=10):
    """Return a paginated query object for the history page."""
    return (ScanHistory.query
            .filter_by(user_id=user_id)
            .order_by(ScanHistory.scan_date.desc())
            .paginate(page=page, per_page=per_page))


def save_scan(user_id, filename, file_size, inference_ms, status='completed'):
    """
    Create and commit a new ScanHistory record.
    Call this after every successful /predict run.
    """
    record = ScanHistory(
        user_id      = user_id,
        filename     = filename,
        status       = status,
        file_size    = file_size,
        inference_ms = inference_ms,
    )
    db.session.add(record)
    db.session.commit()
    return record
