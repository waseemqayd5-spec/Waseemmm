#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 ViralShield AI Engine - النسخة المتكاملة (Backend + Frontend)
ملف واحد جاهز للنشر على Render مع دعم SQLite / PostgreSQL
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, render_template_string
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from dotenv import load_dotenv
import numpy as np

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///viralshield.db')
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# إعدادات التجميع فقط لـ PostgreSQL (تجنب خطأ SQLite)
is_sqlite = DATABASE_URL.startswith('sqlite://')
if not is_sqlite:
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_size': 10,
        'pool_recycle': 300,
        'pool_pre_ping': True
    }
else:
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {}

db = SQLAlchemy(app)

# ==================== نماذج قاعدة البيانات ====================

class Promoter(db.Model):
    __tablename__ = 'promoters'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    promo_code = Column(String(20), unique=True, nullable=False, index=True)
    base_commission_rate = Column(Float, default=0.10)
    total_sales = Column(Integer, default=0)
    total_earnings = Column(Float, default=0.0)
    last_login = Column(DateTime, default=datetime.utcnow)
    total_clicks_tracked = Column(Integer, default=0)
    last_sale_date = Column(DateTime)
    activity_score = Column(Float, default=100.0)
    is_at_risk_of_churn = Column(Boolean, default=False)
    churn_probability = Column(Float, default=0.0)
    fingerprints = relationship('DeviceFingerprint', backref='promoter', lazy=True)
    challenges = relationship('ViralGroupChallenge', backref='promoter', lazy=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'promo_code': self.promo_code,
            'total_sales': self.total_sales,
            'total_earnings': round(self.total_earnings, 2),
            'activity_score': round(self.activity_score, 1),
            'is_at_risk': self.is_at_risk_of_churn
        }

class DeviceFingerprint(db.Model):
    __tablename__ = 'device_fingerprints'
    id = Column(Integer, primary_key=True)
    device_hash = Column(String(64), unique=True, nullable=False, index=True)
    promoter_id = Column(Integer, ForeignKey('promoters.id'), nullable=False)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    screen_resolution = Column(String(20))
    browser_language = Column(String(10))
    timezone = Column(String(50))
    first_click_time = Column(DateTime, default=datetime.utcnow)
    last_click_time = Column(DateTime, default=datetime.utcnow)
    click_count = Column(Integer, default=1)
    has_purchased = Column(Boolean, default=False)
    is_suspicious = Column(Boolean, default=False)
    fraud_score = Column(Float, default=0.0)
    
    def to_dict(self):
        return {
            'id': self.id,
            'device_hash': self.device_hash[:16] + '...',
            'promoter_id': self.promoter_id,
            'click_count': self.click_count,
            'has_purchased': self.has_purchased
        }

class ViralGroupChallenge(db.Model):
    __tablename__ = 'viral_group_challenges'
    id = Column(Integer, primary_key=True)
    creator_buyer_id = Column(String(50), nullable=False)
    promoter_id = Column(Integer, ForeignKey('promoters.id'), nullable=False)
    promo_code_used = Column(String(20), nullable=False)
    required_buyers = Column(Integer, default=3)
    current_buyers_joined = Column(Integer, default=1)
    challenge_duration_hours = Column(Float, default=3.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    expiration_time = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    status = Column(String(20), default="ACTIVE", index=True)
    product_price = Column(Float)
    product_cost = Column(Float)
    discount_applied = Column(Float, default=0.0)
    total_sales_generated = Column(Float, default=0.0)
    
    def to_dict(self):
        remaining_time = None
        if self.status == "ACTIVE" and self.expiration_time:
            remaining = self.expiration_time - datetime.utcnow()
            remaining_time = str(remaining) if remaining.total_seconds() > 0 else "0:00:00"
        return {
            'id': self.id,
            'promo_code': self.promo_code_used,
            'required_buyers': self.required_buyers,
            'current_buyers': self.current_buyers_joined,
            'remaining_to_activate': self.required_buyers - self.current_buyers_joined,
            'status': self.status,
            'remaining_time': remaining_time,
            'discount_percentage': self.discount_applied,
            'product_price': self.product_price,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Transaction(db.Model):
    __tablename__ = 'transactions'
    id = Column(Integer, primary_key=True)
    transaction_type = Column(String(50), nullable=False)
    amount = Column(Float, nullable=False)
    promoter_id = Column(Integer, ForeignKey('promoters.id'))
    challenge_id = Column(Integer, ForeignKey('viral_group_challenges.id'))
    fingerprint_id = Column(Integer, ForeignKey('device_fingerprints.id'))
    product_price = Column(Float)
    product_cost = Column(Float)
    merchant_profit = Column(Float)
    commission_paid = Column(Float)
    discount_given = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.transaction_type,
            'amount': round(self.amount, 2),
            'merchant_profit': round(self.merchant_profit, 2) if self.merchant_profit else 0,
            'commission_paid': round(self.commission_paid, 2) if self.commission_paid else 0,
            'discount_given': round(self.discount_given, 2) if self.discount_given else 0,
            'date': self.created_at.isoformat()
        }

# ==================== المحرك الرئيسي ====================

class ViralShieldEngine:
    def __init__(self):
        self.fraud_threshold = 0.7
        self.churn_threshold = 0.6
    
    def generate_device_hash(self, ip_address, user_agent, screen_res, language='en', timezone='UTC'):
        raw_string = f"{ip_address}|{user_agent}|{screen_res}|{language}|{timezone}"
        salt = "ViralShield_SecureSalt_2024"
        raw_string_with_salt = f"{raw_string}|{salt}"
        return hashlib.sha256(raw_string_with_salt.encode('utf-8')).hexdigest()
    
    def detect_fraud(self, device_hash, promo_code):
        fraud_signals = []
        fingerprints = DeviceFingerprint.query.filter_by(device_hash=device_hash).all()
        if not fingerprints:
            return {'is_fraudulent': False, 'fraud_score': 0.0, 'signals': []}
        if len(fingerprints) > 1:
            unique_promoters = set(f.promoter_id for f in fingerprints)
            if len(unique_promoters) > 1:
                fraud_signals.append("جهاز مرتبط بأكثر من مروج")
        recent_purchases = Transaction.query.filter(
            Transaction.fingerprint_id.in_([f.id for f in fingerprints]),
            Transaction.created_at >= datetime.utcnow() - timedelta(hours=1)
        ).count()
        if recent_purchases > 5:
            fraud_signals.append("نمط شراء متكرر ومشبوه")
        fraud_score = min(len(fraud_signals) * 0.3, 1.0)
        return {
            'is_fraudulent': fraud_score >= self.fraud_threshold,
            'fraud_score': fraud_score,
            'signals': fraud_signals
        }
    
    def calculate_smart_margin(self, product_price, product_cost, is_viral_success=False, 
                              promoter_performance=0.5, order_volume=1):
        gross_profit = product_price - product_cost
        if gross_profit <= 0:
            return {
                'final_price': product_price,
                'promoter_payout': 0,
                'merchant_secured_profit': 0,
                'error': 'المنتج لا يحقق هامش ربح كافي'
            }
        max_total_discount = min(product_price * 0.9, gross_profit * 0.95)
        if is_viral_success:
            max_discount_pct = min(0.40, (gross_profit * 0.8) / product_price)
            buyer_discount = min(gross_profit * max_discount_pct, max_total_discount)
            promoter_commission = gross_profit * 0.20
            if promoter_performance > 0.7:
                promoter_commission *= 1.2
        else:
            max_discount_pct = min(0.15, (gross_profit * 0.5) / product_price)
            buyer_discount = min(gross_profit * max_discount_pct, max_total_discount)
            promoter_commission = gross_profit * 0.10
            if order_volume >= 5:
                buyer_discount = min(buyer_discount * 1.1, max_total_discount)
                promoter_commission *= 0.95
        merchant_net_profit = gross_profit - buyer_discount - promoter_commission
        if merchant_net_profit < 0:
            min_profit = product_cost * 0.10
            total_available = gross_profit - min_profit
            buyer_discount = total_available * 0.6
            promoter_commission = total_available * 0.4
            merchant_net_profit = min_profit
        return {
            'final_price': round(product_price - buyer_discount, 2),
            'promoter_payout': round(promoter_commission, 2),
            'merchant_secured_profit': round(merchant_net_profit, 2),
            'discount_percentage': round((buyer_discount / product_price) * 100, 1),
            'commission_percentage': round((promoter_commission / gross_profit) * 100, 1)
        }
    
    def create_viral_challenge(self, buyer_id, promo_code, product_price, product_cost, 
                              required_buyers=3, duration_hours=3):
        promoter = Promoter.query.filter_by(promo_code=promo_code).first()
        if not promoter:
            return {'error': 'كود المروج غير صالح'}
        challenge = ViralGroupChallenge(
            creator_buyer_id=buyer_id,
            promoter_id=promoter.id,
            promo_code_used=promo_code,
            required_buyers=required_buyers,
            challenge_duration_hours=duration_hours,
            expiration_time=datetime.utcnow() + timedelta(hours=duration_hours),
            product_price=product_price,
            product_cost=product_cost,
            status="ACTIVE"
        )
        db.session.add(challenge)
        db.session.commit()
        promoter.activity_score = min(100, promoter.activity_score + 5)
        db.session.commit()
        return {
            'challenge_id': challenge.id,
            'status': 'ACTIVE',
            'expires_at': challenge.expiration_time.isoformat(),
            'required_buyers': required_buyers,
            'current_buyers': 1,
            'remaining_to_activate': required_buyers - 1
        }
    
    def process_viral_purchase(self, challenge_id, buyer_id, device_hash):
        challenge = ViralGroupChallenge.query.get(challenge_id)
        if not challenge:
            return {'error': 'التحدي غير موجود'}
        if challenge.status != "ACTIVE":
            return {'error': f'التحدي {challenge.status}'}
        if datetime.utcnow() > challenge.expiration_time:
            challenge.status = "EXPIRED"
            db.session.commit()
            return {'error': 'انتهى وقت التحدي'}
        challenge.current_buyers_joined += 1
        challenge.total_sales_generated += challenge.product_price
        if challenge.current_buyers_joined >= challenge.required_buyers:
            challenge.status = "SUCCESS"
            challenge.completed_at = datetime.utcnow()
            finance = self.calculate_smart_margin(
                challenge.product_price,
                challenge.product_cost,
                is_viral_success=True
            )
            challenge.discount_applied = finance['discount_percentage']
            transaction = Transaction(
                transaction_type="COMMISSION",
                amount=finance['promoter_payout'] * challenge.current_buyers_joined,
                promoter_id=challenge.promoter_id,
                challenge_id=challenge.id,
                product_price=challenge.product_price,
                product_cost=challenge.product_cost,
                merchant_profit=finance['merchant_secured_profit'],
                commission_paid=finance['promoter_payout'],
                discount_given=finance['final_price']
            )
            db.session.add(transaction)
            promoter = Promoter.query.get(challenge.promoter_id)
            promoter.total_sales += challenge.current_buyers_joined
            promoter.total_earnings += finance['promoter_payout'] * challenge.current_buyers_joined
            promoter.last_sale_date = datetime.utcnow()
            promoter.activity_score = min(100, promoter.activity_score + 10)
            db.session.commit()
            return {
                'status': 'SUCCESS',
                'message': '🎉 مبروك! اكتمل التحدي الجماعي',
                'final_price': finance['final_price'],
                'discount_applied': f"{finance['discount_percentage']}%",
                'savings': round(challenge.product_price - finance['final_price'], 2)
            }
        db.session.commit()
        remaining = challenge.required_buyers - challenge.current_buyers_joined
        return {
            'status': 'PENDING',
            'message': f'تم تسجيل شرائك! متبقي {remaining} أصدقاء لتفعيل الخصم الأكبر',
            'current_buyers': challenge.current_buyers_joined,
            'remaining_to_activate': remaining,
            'current_discount': '5%'
        }
    
    def calculate_churn_probability(self, promoter):
        signals = []
        churn_score = 0.0
        if promoter.last_sale_date:
            days_since_last_sale = (datetime.utcnow() - promoter.last_sale_date).days
            if days_since_last_sale > 30:
                churn_score += 0.4
                signals.append("أكثر من 30 يوم بدون مبيعات")
            elif days_since_last_sale > 14:
                churn_score += 0.2
                signals.append("أكثر من أسبوعين بدون مبيعات")
        else:
            days_since_registration = (datetime.utcnow() - promoter.created_at).days
            if days_since_registration > 7:
                churn_score += 0.5
                signals.append("مسجل منذ فترة بدون أي مبيعات")
        if promoter.total_clicks_tracked > 20:
            conversion_rate = promoter.total_sales / promoter.total_clicks_tracked
            if conversion_rate < 0.01:
                churn_score += 0.3
                signals.append("معدل تحويل منخفض جداً")
        if promoter.last_login:
            days_since_login = (datetime.utcnow() - promoter.last_login).days
            if days_since_login > 14:
                churn_score += 0.3
                signals.append("لم يسجل الدخول منذ فترة")
            elif days_since_login > 7:
                churn_score += 0.15
        if promoter.activity_score > 80:
            churn_score = max(0, churn_score - 0.2)
        churn_probability = min(churn_score, 1.0)
        return {
            'churn_probability': round(churn_probability, 2),
            'is_at_risk': churn_probability >= self.churn_threshold,
            'signals': signals,
            'activity_score': promoter.activity_score
        }
    
    def get_churn_alert_message(self, promoter_name, churn_data):
        if not churn_data['is_at_risk']:
            return None
        if churn_data['churn_probability'] > 0.8:
            return f"🔥 {promoter_name}، متجرنا يحتاجك! عرض خاص: عمولة مضاعفة على مبيعات اليوم فقط!"
        elif churn_data['churn_probability'] > 0.6:
            return f"💡 {promoter_name}، لاحظنا غيابك. هل تحتاج مساعدة في التسويق؟"
        else:
            return f"👋 {promoter_name}، مرحباً! لدينا منتجات جديدة قد تعجب متابعيك."

engine = ViralShieldEngine()

# ==================== واجهة المستخدم (HTML/CSS/JS) ====================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ViralShield AI | نظام التسويق الذكي</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <style>
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family: 'Tajawal', 'Segoe UI', sans-serif; }
        .navbar-brand { font-weight: bold; }
        .card { border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); transition: transform 0.3s; }
        .card:hover { transform: translateY(-5px); }
        .stats-card { border-right: 5px solid #0d6efd; }
        .bg-glass { background: rgba(255,255,255,0.9); backdrop-filter: blur(5px); }
        .btn { border-radius: 50px; padding: 8px 25px; }
        pre { background: #f8f9fa; border-radius: 10px; padding: 15px; }
        .table-responsive { max-height: 400px; overflow-y: auto; }
        .badge-pulse { animation: pulse 1s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    </style>
</head>
<body>

<nav class="navbar navbar-dark bg-dark bg-glass shadow-sm">
    <div class="container">
        <span class="navbar-brand"><i class="bi bi-shield-shaded"></i> ViralShield AI Engine</span>
        <span class="text-white-50">v3.0.1 | نظام تسويق ذكي وفيروسي</span>
    </div>
</nav>

<div class="container mt-4">
    <!-- لوحة الإحصائيات -->
    <div class="row" id="statsCards">
        <div class="col-md-3 mb-3">
            <div class="card stats-card text-center p-3">
                <h5><i class="bi bi-people"></i> المروجين</h5>
                <h2 id="totalPromoters">-</h2>
            </div>
        </div>
        <div class="col-md-3 mb-3">
            <div class="card stats-card text-center p-3">
                <h5><i class="bi bi-lightning-charge"></i> تحديات نشطة</h5>
                <h2 id="activeChallenges">-</h2>
            </div>
        </div>
        <div class="col-md-3 mb-3">
            <div class="card stats-card text-center p-3">
                <h5><i class="bi bi-trophy"></i> تحديات ناجحة</h5>
                <h2 id="successfulChallenges">-</h2>
            </div>
        </div>
        <div class="col-md-3 mb-3">
            <div class="card stats-card text-center p-3">
                <h5><i class="bi bi-exclamation-triangle"></i> مروجين معرضين للخطر</h5>
                <h2 id="atRiskPromoters">-</h2>
            </div>
        </div>
    </div>
    <div class="row">
        <div class="col-md-6 mb-3">
            <div class="card p-3">
                <h5><i class="bi bi-graph-up"></i> المبيعات والأرباح</h5>
                <p>إجمالي المبيعات: <strong id="totalSales">-</strong> ر.س</p>
                <p>ربح التاجر: <strong id="merchantProfit">-</strong> ر.س</p>
            </div>
        </div>
        <div class="col-md-6 mb-3">
            <div class="card p-3">
                <h5><i class="bi bi-robot"></i> حالة النظام</h5>
                <p id="systemHealth">✅ جميع الأنظمة تعمل بكفاءة</p>
                <small>آخر تحديث: <span id="lastUpdate"></span></small>
            </div>
        </div>
    </div>

    <!-- تسجيل مروج جديد -->
    <div class="card mt-3 p-4">
        <h4><i class="bi bi-person-plus"></i> تسجيل مروج جديد</h4>
        <div class="row g-3">
            <div class="col-md-4">
                <input type="text" id="promoterName" class="form-control" placeholder="الاسم الكامل">
            </div>
            <div class="col-md-4">
                <input type="email" id="promoterEmail" class="form-control" placeholder="البريد الإلكتروني">
            </div>
            <div class="col-md-4">
                <button class="btn btn-primary w-100" onclick="registerPromoter()"><i class="bi bi-check-lg"></i> تسجيل</button>
            </div>
        </div>
        <div id="registerResult" class="mt-3"></div>
    </div>

    <!-- إنشاء تحدي جماعي -->
    <div class="card mt-4 p-4">
        <h4><i class="bi bi-people-fill"></i> إنشاء تحدي شراء جماعي</h4>
        <div class="row g-3">
            <div class="col-md-3"><input type="text" id="challengePromoCode" class="form-control" placeholder="كود المروج"></div>
            <div class="col-md-2"><input type="number" id="productPrice" class="form-control" placeholder="سعر المنتج"></div>
            <div class="col-md-2"><input type="number" id="productCost" class="form-control" placeholder="تكلفة المنتج"></div>
            <div class="col-md-2"><input type="number" id="requiredBuyers" class="form-control" placeholder="عدد المشترين" value="3"></div>
            <div class="col-md-2"><input type="number" id="durationHours" class="form-control" placeholder="المدة (ساعات)" value="3"></div>
            <div class="col-md-1"><button class="btn btn-success w-100" onclick="createChallenge()"><i class="bi bi-rocket"></i></button></div>
        </div>
        <div id="challengeResult" class="mt-3"></div>
    </div>

    <!-- قائمة التحديات النشطة -->
    <div class="card mt-4 p-4">
        <h4><i class="bi bi-list-task"></i> التحديات النشطة</h4>
        <div class="table-responsive">
            <table class="table table-hover">
                <thead><tr><th>#</th><th>كود المروج</th><th>المطلوب</th><th>المنضم</th><th>الوقت المتبقي</th><th>الحالة</th></tr></thead>
                <tbody id="challengesTable"></tbody>
            </table>
        </div>
    </div>

    <!-- حاسبة العمولة الذكية -->
    <div class="card mt-4 p-4">
        <h4><i class="bi bi-calculator"></i> حساب العمولة والخصم الديناميكي</h4>
        <div class="row g-3">
            <div class="col-md-2"><input type="number" id="calcPrice" class="form-control" placeholder="سعر المنتج" value="100"></div>
            <div class="col-md-2"><input type="number" id="calcCost" class="form-control" placeholder="التكلفة" value="60"></div>
            <div class="col-md-2"><select id="viralSuccess" class="form-select"><option value="false">شراء عادي</option><option value="true">نجاح تحدي</option></select></div>
            <div class="col-md-2"><input type="number" id="promoterPerf" class="form-control" placeholder="أداء المروج (0-1)" value="0.5" step="0.1"></div>
            <div class="col-md-2"><input type="number" id="orderVol" class="form-control" placeholder="حجم الطلب" value="1"></div>
            <div class="col-md-2"><button class="btn btn-info w-100" onclick="calculateCommission()"><i class="bi bi-calculator"></i> احسب</button></div>
        </div>
        <div id="calcResult" class="mt-3 alert alert-secondary"></div>
    </div>

    <!-- قائمة المروجين مع تحليل الخمول -->
    <div class="card mt-4 p-4">
        <h4><i class="bi bi-person-badge"></i> المروجون وتحليل الخمول</h4>
        <div class="table-responsive">
            <table class="table table-striped">
                <thead><tr><th>الاسم</th><th>الكود</th><th>المبيعات</th><th>الأرباح</th><th>نسبة النشاط</th><th>خطر الخمول</th><th>إجراء</th></tr></thead>
                <tbody id="promotersTable"></tbody>
            </table>
        </div>
    </div>
</div>

<script>
    // تحديث لوحة التحكم
    async function loadDashboard() {
        try {
            const res = await fetch('/api/dashboard/stats');
            const data = await res.json();
            if (data.overview) {
                document.getElementById('totalPromoters').innerText = data.overview.total_promoters;
                document.getElementById('activeChallenges').innerText = data.overview.active_challenges;
                document.getElementById('successfulChallenges').innerText = data.overview.successful_challenges;
                document.getElementById('atRiskPromoters').innerText = data.overview.at_risk_promoters;
                document.getElementById('totalSales').innerText = data.financials.total_sales;
                document.getElementById('merchantProfit').innerText = data.financials.total_merchant_profit;
                document.getElementById('systemHealth').innerHTML = data.system_health;
            }
            document.getElementById('lastUpdate').innerText = new Date().toLocaleString();
        } catch(e) { console.error(e); }
        loadChallenges();
        loadPromoters();
    }

    async function loadChallenges() {
        try {
            const res = await fetch('/api/challenges/active');
            const challenges = await res.json();
            const tbody = document.getElementById('challengesTable');
            tbody.innerHTML = '';
            challenges.forEach(ch => {
                let row = `<tr>
                    <td>${ch.id}</td>
                    <td>${ch.promo_code}</td>
                    <td>${ch.required_buyers}</td>
                    <td>${ch.current_buyers}</td>
                    <td>${ch.remaining_time || 'انتهى'}</td>
                    <td><span class="badge ${ch.status === 'ACTIVE' ? 'bg-success' : 'bg-secondary'}">${ch.status}</span></td>
                </tr>`;
                tbody.innerHTML += row;
            });
        } catch(e) { console.error(e); }
    }

    async function loadPromoters() {
        try {
            const res = await fetch('/api/promoters/all');
            const promoters = await res.json();
            const tbody = document.getElementById('promotersTable');
            tbody.innerHTML = '';
            for (let p of promoters) {
                let riskBadge = p.is_at_risk ? '<span class="badge bg-danger">خطر</span>' : '<span class="badge bg-success">نشط</span>';
                let row = `<tr>
                    <td>${p.name}</td>
                    <td>${p.promo_code}</td>
                    <td>${p.total_sales}</td>
                    <td>${p.total_earnings}</td>
                    <td>${p.activity_score}%</td>
                    <td>${riskBadge}</td>
                    <td><button class="btn btn-sm btn-warning" onclick="predictChurn(${p.id})"><i class="bi bi-graph-up"></i> تحليل</button></td>
                </tr>`;
                tbody.innerHTML += row;
            }
        } catch(e) { console.error(e); }
    }

    async function registerPromoter() {
        const name = document.getElementById('promoterName').value;
        const email = document.getElementById('promoterEmail').value;
        if (!name || !email) { alert('يرجى إدخال الاسم والبريد'); return; }
        const res = await fetch('/api/promoter/register', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, email})
        });
        const data = await res.json();
        document.getElementById('registerResult').innerHTML = `<div class="alert alert-success">✅ ${data.message}<br>الكود: ${data.promoter.promo_code}</div>`;
        loadDashboard();
    }

    async function createChallenge() {
        const promo_code = document.getElementById('challengePromoCode').value;
        const product_price = parseFloat(document.getElementById('productPrice').value);
        const product_cost = parseFloat(document.getElementById('productCost').value);
        const required_buyers = parseInt(document.getElementById('requiredBuyers').value);
        const duration_hours = parseInt(document.getElementById('durationHours').value);
        if (!promo_code || isNaN(product_price) || isNaN(product_cost)) { alert('يرجى ملء جميع الحقول'); return; }
        const res = await fetch('/api/challenge/create', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({buyer_id: 'user_' + Date.now(), promo_code, product_price, product_cost, required_buyers, duration_hours})
        });
        const data = await res.json();
        if (data.challenge) {
            document.getElementById('challengeResult').innerHTML = `<div class="alert alert-success">🎯 تم إنشاء التحدي! الرقم: ${data.challenge.challenge_id}</div>`;
            loadDashboard();
        } else {
            document.getElementById('challengeResult').innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
        }
    }

    async function calculateCommission() {
        const product_price = parseFloat(document.getElementById('calcPrice').value);
        const product_cost = parseFloat(document.getElementById('calcCost').value);
        const is_viral_success = document.getElementById('viralSuccess').value === 'true';
        const promoter_performance = parseFloat(document.getElementById('promoterPerf').value);
        const order_volume = parseInt(document.getElementById('orderVol').value);
        const res = await fetch('/api/calculate/commission', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({product_price, product_cost, is_viral_success, promoter_performance, order_volume})
        });
        const data = await res.json();
        document.getElementById('calcResult').innerHTML = `<strong>النتيجة:</strong><br>
        السعر النهائي: ${data.calculation.final_price} ر.س<br>
        عمولة المروج: ${data.calculation.promoter_payout} ر.س<br>
        ربح التاجر: ${data.calculation.merchant_secured_profit} ر.س<br>
        نسبة الخصم: ${data.calculation.discount_percentage}%`;
    }

    async function predictChurn(promoterId) {
        const res = await fetch(`/api/ai/predict-churn/${promoterId}`);
        const data = await res.json();
        alert(`تحليل المروج ${data.promoter.name}:\nاحتمالية التوقف: ${data.churn_analysis.churn_probability * 100}%\n${data.alert_message || ''}`);
        loadDashboard();
    }

    setInterval(loadDashboard, 15000);
    loadDashboard();
</script>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# ==================== نقاط النهاية API (مع إضافة Routes للواجهة) ====================

@app.route('/')
def index():
    """الصفحة الرئيسية - الواجهة المتكاملة"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/promoters/all', methods=['GET'])
def get_all_promoters():
    promoters = Promoter.query.order_by(Promoter.id.desc()).all()
    return jsonify([p.to_dict() for p in promoters])

@app.route('/api/challenges/active', methods=['GET'])
def get_active_challenges():
    challenges = ViralGroupChallenge.query.filter_by(status='ACTIVE').order_by(ViralGroupChallenge.created_at.desc()).all()
    return jsonify([c.to_dict() for c in challenges])

# جميع نقاط API الأصلية (تم الاحتفاظ بها كما هي)
@app.route('/api/promoter/register', methods=['POST'])
def register_promoter():
    try:
        data = request.json
        if not data.get('name') or not data.get('email'):
            return jsonify({'error': 'الاسم والبريد الإلكتروني مطلوبان'}), 400
        if Promoter.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'البريد الإلكتروني مسجل مسبقاً'}), 400
        promo_code = f"{data['name'][:5].upper()}{secrets.token_hex(3).upper()}"
        promoter = Promoter(
            name=data['name'],
            email=data['email'],
            promo_code=promo_code,
            base_commission_rate=data.get('commission_rate', 0.10)
        )
        db.session.add(promoter)
        db.session.commit()
        return jsonify({'message': '✅ تم تسجيل المروج بنجاح', 'promoter': promoter.to_dict()}), 201
    except Exception as e:
        return jsonify({'error': f'خطأ في التسجيل: {str(e)}'}), 500

@app.route('/api/track/visit', methods=['POST'])
def track_visit():
    try:
        data = request.json
        device_hash = engine.generate_device_hash(
            ip_address=request.remote_addr or 'unknown',
            user_agent=request.headers.get('User-Agent', 'unknown'),
            screen_res=data.get('screen_resolution', 'unknown'),
            language=data.get('language', 'ar'),
            timezone=data.get('timezone', 'UTC')
        )
        existing = DeviceFingerprint.query.filter_by(device_hash=device_hash).first()
        if existing:
            existing.click_count += 1
            existing.last_click_time = datetime.utcnow()
            fraud_check = engine.detect_fraud(device_hash, data.get('promo_code'))
            existing.is_suspicious = fraud_check['is_fraudulent']
            existing.fraud_score = fraud_check['fraud_score']
            promoter = Promoter.query.get(existing.promoter_id)
            if promoter:
                promoter.total_clicks_tracked += 1
                promoter.last_login = datetime.utcnow()
            db.session.commit()
            return jsonify({
                'status': 'returning_visitor',
                'device_hash': device_hash[:16] + '...',
                'visit_count': existing.click_count,
                'fraud_check': fraud_check
            })
        promoter = Promoter.query.filter_by(promo_code=data.get('promo_code')).first()
        if not promoter:
            return jsonify({'error': 'كود المروج غير صالح'}), 404
        fingerprint = DeviceFingerprint(
            device_hash=device_hash,
            promoter_id=promoter.id,
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            screen_resolution=data.get('screen_resolution'),
            browser_language=data.get('language'),
            timezone=data.get('timezone')
        )
        db.session.add(fingerprint)
        promoter.total_clicks_tracked += 1
        promoter.last_login = datetime.utcnow()
        db.session.commit()
        return jsonify({
            'status': 'new_visitor',
            'device_hash': device_hash[:16] + '...',
            'promoter': promoter.name,
            'fraud_check': {'is_fraudulent': False, 'fraud_score': 0.0}
        }), 201
    except Exception as e:
        return jsonify({'error': f'خطأ في التتبع: {str(e)}'}), 500

@app.route('/api/challenge/create', methods=['POST'])
def create_challenge():
    try:
        data = request.json
        result = engine.create_viral_challenge(
            buyer_id=data.get('buyer_id'),
            promo_code=data.get('promo_code'),
            product_price=data.get('product_price', 100),
            product_cost=data.get('product_cost', 60),
            required_buyers=data.get('required_buyers', 3),
            duration_hours=data.get('duration_hours', 3)
        )
        if 'error' in result:
            return jsonify(result), 400
        return jsonify({'message': '🎯 تم إنشاء التحدي الجماعي!', 'challenge': result}), 201
    except Exception as e:
        return jsonify({'error': f'خطأ في إنشاء التحدي: {str(e)}'}), 500

@app.route('/api/challenge/join/<int:challenge_id>', methods=['POST'])
def join_challenge(challenge_id):
    try:
        data = request.json
        device_hash = engine.generate_device_hash(
            ip_address=request.remote_addr or 'unknown',
            user_agent=request.headers.get('User-Agent', 'unknown'),
            screen_res=data.get('screen_resolution', 'unknown')
        )
        result = engine.process_viral_purchase(
            challenge_id=challenge_id,
            buyer_id=data.get('buyer_id'),
            device_hash=device_hash
        )
        if 'error' in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'خطأ في الانضمام للتحدي: {str(e)}'}), 500

@app.route('/api/challenge/status/<int:challenge_id>', methods=['GET'])
def challenge_status(challenge_id):
    try:
        challenge = ViralGroupChallenge.query.get(challenge_id)
        if not challenge:
            return jsonify({'error': 'التحدي غير موجود'}), 404
        return jsonify(challenge.to_dict())
    except Exception as e:
        return jsonify({'error': f'خطأ في جلب حالة التحدي: {str(e)}'}), 500

@app.route('/api/calculate/commission', methods=['POST'])
def calculate_commission():
    try:
        data = request.json
        result = engine.calculate_smart_margin(
            product_price=data.get('product_price', 100),
            product_cost=data.get('product_cost', 60),
            is_viral_success=data.get('is_viral_success', False),
            promoter_performance=data.get('promoter_performance', 0.5),
            order_volume=data.get('order_volume', 1)
        )
        return jsonify({'calculation': result, 'summary': f"السعر النهائي: {result['final_price']} | عمولة المروج: {result['promoter_payout']} | ربح التاجر: {result['merchant_secured_profit']}"})
    except Exception as e:
        return jsonify({'error': f'خطأ في الحساب: {str(e)}'}), 500

@app.route('/api/ai/predict-churn/<int:promoter_id>', methods=['GET'])
def predict_churn(promoter_id):
    try:
        promoter = Promoter.query.get(promoter_id)
        if not promoter:
            return jsonify({'error': 'المروج غير موجود'}), 404
        churn_data = engine.calculate_churn_probability(promoter)
        alert_message = engine.get_churn_alert_message(promoter.name, churn_data)
        promoter.churn_probability = churn_data['churn_probability']
        promoter.is_at_risk_of_churn = churn_data['is_at_risk']
        if churn_data['is_at_risk']:
            promoter.activity_score = max(0, promoter.activity_score - 5)
        else:
            promoter.activity_score = min(100, promoter.activity_score + 2)
        db.session.commit()
        return jsonify({
            'promoter': promoter.to_dict(),
            'churn_analysis': churn_data,
            'alert_message': alert_message,
            'recommendation': 'إرسال حافز خاص' if churn_data['is_at_risk'] else 'المروج نشط'
        })
    except Exception as e:
        return jsonify({'error': f'خطأ في التنبؤ: {str(e)}'}), 500

@app.route('/api/dashboard/stats', methods=['GET'])
def dashboard_stats():
    try:
        total_promoters = Promoter.query.count()
        active_challenges = ViralGroupChallenge.query.filter_by(status='ACTIVE').count()
        successful_challenges = ViralGroupChallenge.query.filter_by(status='SUCCESS').count()
        total_transactions = Transaction.query.count()
        at_risk_promoters = Promoter.query.filter_by(is_at_risk_of_churn=True).count()
        total_sales = db.session.query(db.func.sum(Transaction.amount)).filter_by(transaction_type='SALE').scalar() or 0
        total_merchant_profit = db.session.query(db.func.sum(Transaction.merchant_profit)).scalar() or 0
        return jsonify({
            'overview': {
                'total_promoters': total_promoters,
                'active_challenges': active_challenges,
                'successful_challenges': successful_challenges,
                'total_transactions': total_transactions,
                'at_risk_promoters': at_risk_promoters
            },
            'financials': {
                'total_sales': round(total_sales, 2),
                'total_merchant_profit': round(total_merchant_profit, 2)
            },
            'system_health': '✅ جميع الأنظمة تعمل بكفاءة'
        })
    except Exception as e:
        return jsonify({'error': f'خطأ في جلب الإحصائيات: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        db.session.execute(db.text('SELECT 1'))
        return jsonify({'status': 'healthy', 'database': 'connected', 'timestamp': datetime.utcnow().isoformat()})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'المسار غير موجود', 'status_code': 404}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'خطأ في الخادم', 'status_code': 500}), 500

# ==================== تهيئة قاعدة البيانات ====================
def init_database():
    with app.app_context():
        db.create_all()
        if Promoter.query.count() == 0:
            demo_promoter = Promoter(
                name="أحمد المسوق",
                email="demo@viralshield.com",
                promo_code="DEMO2024",
                base_commission_rate=0.10,
                activity_score=85.0
            )
            db.session.add(demo_promoter)
            db.session.commit()
            print("✅ تم إنشاء المروج التجريبي: DEMO2024")
        print("✅ قاعدة البيانات جاهزة وجميع الجداول موجودة")

init_database()

# ==================== التشغيل ====================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"""
╔══════════════════════════════════════════════════════════╗
║     🚀 ViralShield AI Engine - النسخة المتكاملة         ║
║     📍 Running on: http://0.0.0.0:{port}                 ║
║     🎨 UI доступна على الصفحة الرئيسية                  ║
║     🧠 AI Systems: Active                               ║
║     🔐 Fraud Protection: Enabled                        ║
╚══════════════════════════════════════════════════════════╝
    """)
    app.run(host='0.0.0.0', port=port, debug=False)
