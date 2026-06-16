#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 ViralShield AI Engine - النسخة المتكاملة (Backend + Frontend)
ملف واحد جاهز للنشر على Render مع دعم SQLite / PostgreSQL
تم إضافة:
- نظام الروابط القصيرة والتتبع المتقدم (رقم 2)
- متجر منتجات وفرق تسويقية (رقم 5)
- واجهة جذابة بألوان ذهبي، أسود، أبيض، أصفر
- ✅ المشاركة التلقائية على وسائل التواصل الاجتماعي (رقم 2)
- ✅ نظام الكوبونات الديناميكية للمشترين (رقم 3)
- ✅ العمولة المتدرجة (Tiered Commission) (رقم 7)
- ✅ تحليل سلوك العميل (Customer Journey) (رقم 11)
- ✅ تم إصلاح مشكلة الضغط على كود المروج نهائياً
- إضافة تذييل الصفحة: "إعداد وتصميم م/ وسيم الحميدي"
"""

import os
import hashlib
import secrets
import string
from datetime import datetime, timedelta
from functools import wraps
import json

from flask import Flask, request, jsonify, render_template_string, redirect, url_for, abort
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
    level = Column(String(20), default="برونز")
    points = Column(Integer, default=0)
    is_merchant = Column(Boolean, default=False)
    # العمولة المتدرجة (رقم 7)
    monthly_sales = Column(Integer, default=0)
    tier = Column(String(20), default="برونز")
    fingerprints = relationship('DeviceFingerprint', backref='promoter', lazy=True)
    challenges = relationship('ViralGroupChallenge', backref='promoter', lazy=True)
    products = relationship('Product', backref='merchant', lazy=True)
    short_links = relationship('ShortLink', backref='promoter', lazy=True)
    teams_created = relationship('Team', backref='creator', lazy=True)
    team_memberships = relationship('TeamMember', backref='promoter', lazy=True)
    # كوبونات (رقم 3)
    coupons = relationship('Coupon', backref='promoter', lazy=True)
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
            'is_at_risk': self.is_at_risk_of_churn,
            'level': self.level,
            'points': self.points,
            'is_merchant': self.is_merchant,
            'monthly_sales': self.monthly_sales,
            'tier': self.tier
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
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=True)
    
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
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'team_id': self.team_id
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
    product_id = Column(Integer, ForeignKey('products.id'), nullable=True)
    short_link_id = Column(Integer, ForeignKey('short_links.id'), nullable=True)
    # رحلة العميل (رقم 11)
    customer_id = Column(String(100), nullable=True)
    session_id = Column(String(100), nullable=True)
    step = Column(String(50), default="click")
    
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.transaction_type,
            'amount': round(self.amount, 2),
            'merchant_profit': round(self.merchant_profit, 2) if self.merchant_profit else 0,
            'commission_paid': round(self.commission_paid, 2) if self.commission_paid else 0,
            'discount_given': round(self.discount_given, 2) if self.discount_given else 0,
            'date': self.created_at.isoformat(),
            'customer_id': self.customer_id,
            'step': self.step
        }

# ---------- النماذج الجديدة ----------

class Product(db.Model):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    price = Column(Float, nullable=False)
    cost = Column(Float, nullable=False)
    merchant_id = Column(Integer, ForeignKey('promoters.id'), nullable=False)
    image_url = Column(String(500))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    short_links = relationship('ShortLink', backref='product', lazy=True)
    transactions = relationship('Transaction', backref='product', lazy=True)
    coupons = relationship('Coupon', backref='product', lazy=True)  # رقم 3
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'price': round(self.price, 2),
            'cost': round(self.cost, 2),
            'merchant_id': self.merchant_id,
            'merchant_name': self.merchant.name if self.merchant else None,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat()
        }

class ShortLink(db.Model):
    __tablename__ = 'short_links'
    id = Column(Integer, primary_key=True)
    code = Column(String(20), unique=True, nullable=False, index=True)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=False)
    promoter_id = Column(Integer, ForeignKey('promoters.id'), nullable=False)
    clicks_count = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    clicks = relationship('Click', backref='short_link', lazy=True)
    transactions = relationship('Transaction', backref='short_link', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'code': self.code,
            'product_id': self.product_id,
            'product_name': self.product.name if self.product else None,
            'promoter_id': self.promoter_id,
            'promoter_name': self.promoter.name if self.promoter else None,
            'clicks_count': self.clicks_count,
            'conversions': self.conversions,
            'conversion_rate': round((self.conversions / self.clicks_count * 100) if self.clicks_count > 0 else 0, 2),
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active,
            'full_url': f"/r/{self.code}"
        }

class Click(db.Model):
    __tablename__ = 'clicks'
    id = Column(Integer, primary_key=True)
    short_link_id = Column(Integer, ForeignKey('short_links.id'), nullable=False)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    referer = Column(String(500))
    timestamp = Column(DateTime, default=datetime.utcnow)
    converted = Column(Boolean, default=False)
    converted_at = Column(DateTime)
    device_type = Column(String(20))
    browser = Column(String(50))
    country = Column(String(50))
    # رحلة العميل (رقم 11)
    customer_id = Column(String(100), nullable=True)
    session_id = Column(String(100), nullable=True)
    step = Column(String(50), default="click")
    page_visited = Column(String(200))
    time_spent = Column(Integer, default=0)  # ثواني
    exited_at = Column(DateTime)
    
    def to_dict(self):
        return {
            'id': self.id,
            'short_link_id': self.short_link_id,
            'ip': self.ip_address,
            'timestamp': self.timestamp.isoformat(),
            'converted': self.converted,
            'converted_at': self.converted_at.isoformat() if self.converted_at else None,
            'device_type': self.device_type,
            'browser': self.browser,
            'country': self.country,
            'customer_id': self.customer_id,
            'step': self.step,
            'page_visited': self.page_visited,
            'time_spent': self.time_spent
        }

class Team(db.Model):
    __tablename__ = 'teams'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    created_by = Column(Integer, ForeignKey('promoters.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    members = relationship('TeamMember', backref='team', lazy=True)
    challenges = relationship('ViralGroupChallenge', backref='team', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_by': self.created_by,
            'creator_name': self.creator.name if self.creator else None,
            'created_at': self.created_at.isoformat(),
            'members_count': len(self.members),
            'is_active': self.is_active
        }

class TeamMember(db.Model):
    __tablename__ = 'team_members'
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    promoter_id = Column(Integer, ForeignKey('promoters.id'), nullable=False)
    joined_at = Column(DateTime, default=datetime.utcnow)
    role = Column(String(20), default="member")
    
    def to_dict(self):
        return {
            'id': self.id,
            'team_id': self.team_id,
            'promoter_id': self.promoter_id,
            'promoter_name': self.promoter.name if self.promoter else None,
            'joined_at': self.joined_at.isoformat(),
            'role': self.role
        }

# ---------- نماذج الميزات الجديدة ----------

# 3. نظام الكوبونات الديناميكية
class Coupon(db.Model):
    __tablename__ = 'coupons'
    id = Column(Integer, primary_key=True)
    code = Column(String(30), unique=True, nullable=False, index=True)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=False)
    promoter_id = Column(Integer, ForeignKey('promoters.id'), nullable=False)
    discount_percentage = Column(Float, default=0.0)
    discount_amount = Column(Float, default=0.0)
    is_percentage = Column(Boolean, default=True)  # True نسبة مئوية، False قيمة ثابتة
    is_used = Column(Boolean, default=False)
    used_by = Column(String(100))  # معرف العميل
    used_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    max_uses = Column(Integer, default=1)
    use_count = Column(Integer, default=0)
    
    def to_dict(self):
        return {
            'id': self.id,
            'code': self.code,
            'product_id': self.product_id,
            'product_name': self.product.name if self.product else None,
            'promoter_id': self.promoter_id,
            'promoter_name': self.promoter.name if self.promoter else None,
            'discount_percentage': self.discount_percentage,
            'discount_amount': self.discount_amount,
            'is_percentage': self.is_percentage,
            'is_used': self.is_used,
            'used_by': self.used_by,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'created_at': self.created_at.isoformat(),
            'max_uses': self.max_uses,
            'use_count': self.use_count
        }

# 11. تحليل سلوك العميل - جدول إضافي للجلسات
class CustomerJourney(db.Model):
    __tablename__ = 'customer_journeys'
    id = Column(Integer, primary_key=True)
    customer_id = Column(String(100), nullable=False, index=True)
    session_id = Column(String(100), nullable=False, index=True)
    short_link_id = Column(Integer, ForeignKey('short_links.id'), nullable=False)
    promoter_id = Column(Integer, ForeignKey('promoters.id'), nullable=False)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=False)
    first_click = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    total_time_seconds = Column(Integer, default=0)
    pages_visited = Column(Text, default='[]')  # JSON array
    steps = Column(Text, default='[]')  # JSON array من الخطوات
    converted = Column(Boolean, default=False)
    converted_at = Column(DateTime)
    dropped_at = Column(DateTime)  # وقت التخلي عن الشراء
    drop_reason = Column(String(200))
    
    def to_dict(self):
        return {
            'id': self.id,
            'customer_id': self.customer_id,
            'session_id': self.session_id,
            'short_link_id': self.short_link_id,
            'promoter_id': self.promoter_id,
            'promoter_name': self.promoter.name if self.promoter else None,
            'product_id': self.product_id,
            'product_name': self.product.name if self.product else None,
            'first_click': self.first_click.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'total_time_seconds': self.total_time_seconds,
            'pages_visited': json.loads(self.pages_visited) if self.pages_visited else [],
            'steps': json.loads(self.steps) if self.steps else [],
            'converted': self.converted,
            'converted_at': self.converted_at.isoformat() if self.converted_at else None,
            'dropped_at': self.dropped_at.isoformat() if self.dropped_at else None,
            'drop_reason': self.drop_reason
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
    
    # 7. العمولة المتدرجة (Tiered Commission)
    def get_commission_rate(self, promoter):
        """حساب نسبة العمولة بناءً على المبيعات الشهرية"""
        if promoter.monthly_sales >= 100:
            return 0.25  # 25% - البلاتيني
        elif promoter.monthly_sales >= 50:
            return 0.20  # 20% - الذهبي
        elif promoter.monthly_sales >= 20:
            return 0.15  # 15% - الفضي
        elif promoter.monthly_sales >= 5:
            return 0.12  # 12% - البرونزي
        else:
            return 0.10  # 10% - الأساسي
    
    def calculate_smart_margin(self, product_price, product_cost, is_viral_success=False, 
                              promoter_performance=0.5, order_volume=1, promoter=None):
        gross_profit = product_price - product_cost
        if gross_profit <= 0:
            return {
                'final_price': product_price,
                'promoter_payout': 0,
                'merchant_secured_profit': 0,
                'error': 'المنتج لا يحقق هامش ربح كافي'
            }
        # حساب نسبة العمولة المتدرجة
        if promoter:
            commission_rate = self.get_commission_rate(promoter)
        else:
            commission_rate = 0.10
        max_total_discount = min(product_price * 0.9, gross_profit * 0.95)
        if is_viral_success:
            max_discount_pct = min(0.40, (gross_profit * 0.8) / product_price)
            buyer_discount = min(gross_profit * max_discount_pct, max_total_discount)
            promoter_commission = gross_profit * commission_rate
            if promoter_performance > 0.7:
                promoter_commission *= 1.2
        else:
            max_discount_pct = min(0.15, (gross_profit * 0.5) / product_price)
            buyer_discount = min(gross_profit * max_discount_pct, max_total_discount)
            promoter_commission = gross_profit * commission_rate * 0.8
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
            'commission_percentage': round((promoter_commission / gross_profit) * 100, 1),
            'commission_rate_used': round(commission_rate * 100, 1)
        }
    
    def create_viral_challenge(self, buyer_id, promo_code, product_price, product_cost, 
                              required_buyers=3, duration_hours=3, team_id=None):
        promoter = Promoter.query.filter_by(promo_code=promo_code).first()
        if not promoter:
            return {'error': 'كود المروج غير صالح'}
        if team_id:
            team = Team.query.get(team_id)
            if not team:
                return {'error': 'الفريق غير موجود'}
            membership = TeamMember.query.filter_by(team_id=team_id, promoter_id=promoter.id).first()
            if not membership:
                return {'error': 'المروج ليس عضواً في هذا الفريق'}
        challenge = ViralGroupChallenge(
            creator_buyer_id=buyer_id,
            promoter_id=promoter.id,
            promo_code_used=promo_code,
            required_buyers=required_buyers,
            challenge_duration_hours=duration_hours,
            expiration_time=datetime.utcnow() + timedelta(hours=duration_hours),
            product_price=product_price,
            product_cost=product_cost,
            team_id=team_id,
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
            'remaining_to_activate': required_buyers - 1,
            'team_id': team_id
        }
    
    def process_viral_purchase(self, challenge_id, buyer_id, device_hash, customer_id=None):
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
            promoter = Promoter.query.get(challenge.promoter_id)
            finance = self.calculate_smart_margin(
                challenge.product_price,
                challenge.product_cost,
                is_viral_success=True,
                promoter=promoter
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
                discount_given=finance['final_price'],
                customer_id=customer_id,
                step="purchase"
            )
            db.session.add(transaction)
            promoter.total_sales += challenge.current_buyers_joined
            promoter.monthly_sales += challenge.current_buyers_joined  # للعمولة المتدرجة
            promoter.total_earnings += finance['promoter_payout'] * challenge.current_buyers_joined
            promoter.last_sale_date = datetime.utcnow()
            promoter.activity_score = min(100, promoter.activity_score + 10)
            promoter.points += challenge.current_buyers_joined * 5
            self.update_promoter_level(promoter)
            db.session.commit()
            # 3. إنشاء كوبون للمشتري
            coupon = self.create_coupon_for_buyer(promoter, challenge.product_id, buyer_id)
            return {
                'status': 'SUCCESS',
                'message': '🎉 مبروك! اكتمل التحدي الجماعي',
                'final_price': finance['final_price'],
                'discount_applied': f"{finance['discount_percentage']}%",
                'savings': round(challenge.product_price - finance['final_price'], 2),
                'coupon_code': coupon.code if coupon else None
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
    
    # 3. إنشاء كوبون للمشتري
    def create_coupon_for_buyer(self, promoter, product_id, buyer_id):
        """إنشاء كوبون خصم للمشتري بعد اكتمال التحدي"""
        import random
        code = f"VIP-{promoter.promo_code[:4]}-{secrets.token_hex(3).upper()}"
        while Coupon.query.filter_by(code=code).first():
            code = f"VIP-{promoter.promo_code[:4]}-{secrets.token_hex(3).upper()}"
        # خصم عشوائي بين 5% و 15%
        discount = random.randint(5, 15)
        coupon = Coupon(
            code=code,
            product_id=product_id,
            promoter_id=promoter.id,
            discount_percentage=discount,
            is_percentage=True,
            expires_at=datetime.utcnow() + timedelta(days=30),
            max_uses=1
        )
        db.session.add(coupon)
        db.session.commit()
        return coupon
    
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
            conversion_rate = promoter.total_sales / promoter.total_clicks_tracked if promoter.total_clicks_tracked > 0 else 0
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
    
    def update_promoter_level(self, promoter):
        if promoter.points >= 500:
            promoter.level = "ذهبي"
        elif promoter.points >= 200:
            promoter.level = "فضي"
        elif promoter.points >= 50:
            promoter.level = "برونزي"
        else:
            promoter.level = "برونز"
        # تحديث المستوى (tier) بناءً على المبيعات الشهرية
        if promoter.monthly_sales >= 100:
            promoter.tier = "بلاتيني"
        elif promoter.monthly_sales >= 50:
            promoter.tier = "ذهبي"
        elif promoter.monthly_sales >= 20:
            promoter.tier = "فضي"
        elif promoter.monthly_sales >= 5:
            promoter.tier = "برونزي"
        else:
            promoter.tier = "أساسي"
        db.session.commit()

    # 11. تتبع رحلة العميل
    def track_customer_journey(self, customer_id, session_id, short_link_id, product_id, 
                              promoter_id, page_visited, step, time_spent=0):
        """تسجيل خطوة في رحلة العميل"""
        journey = CustomerJourney.query.filter_by(
            customer_id=customer_id, 
            session_id=session_id
        ).first()
        
        if not journey:
            # إنشاء رحلة جديدة
            journey = CustomerJourney(
                customer_id=customer_id,
                session_id=session_id,
                short_link_id=short_link_id,
                promoter_id=promoter_id,
                product_id=product_id,
                first_click=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                pages_visited=json.dumps([]),
                steps=json.dumps([])
            )
            db.session.add(journey)
        
        # تحديث البيانات
        pages = json.loads(journey.pages_visited) if journey.pages_visited else []
        if page_visited and page_visited not in pages:
            pages.append(page_visited)
        journey.pages_visited = json.dumps(pages)
        
        steps = json.loads(journey.steps) if journey.steps else []
        steps.append({
            'step': step,
            'timestamp': datetime.utcnow().isoformat(),
            'page': page_visited,
            'time_spent': time_spent
        })
        journey.steps = json.dumps(steps)
        journey.last_activity = datetime.utcnow()
        journey.total_time_seconds += time_spent
        
        db.session.commit()
        return journey

engine = ViralShieldEngine()

# ==================== واجهة المستخدم ====================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ViralShield AI | نظام التسويق الذكي</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --gold: #FFD700;
            --gold-dark: #D4A017;
            --black: #000000;
            --white: #FFFFFF;
            --yellow: #FFC107;
            --gray: #2C2C2C;
        }
        body {
            background: linear-gradient(135deg, #1a1a1a 0%, #0d0d0d 100%);
            font-family: 'Tajawal', 'Segoe UI', sans-serif;
            color: var(--white);
        }
        .navbar {
            background: rgba(0,0,0,0.9) !important;
            border-bottom: 2px solid var(--gold);
        }
        .navbar-brand, .navbar-brand i {
            color: var(--gold) !important;
            font-weight: bold;
        }
        .card {
            background: rgba(20,20,20,0.9);
            backdrop-filter: blur(10px);
            border: 1px solid var(--gold-dark);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(255,215,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
            color: var(--white);
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(255,215,0,0.2);
        }
        .card-header {
            background: linear-gradient(135deg, var(--gold-dark), var(--gold));
            color: var(--black);
            font-weight: bold;
            border-radius: 20px 20px 0 0 !important;
            border-bottom: none;
        }
        .card-header h5, .card-header i {
            color: var(--black);
        }
        .btn-gold {
            background: linear-gradient(135deg, var(--gold), var(--yellow));
            color: var(--black);
            border: none;
            font-weight: bold;
            border-radius: 50px;
            padding: 8px 25px;
            transition: all 0.3s;
        }
        .btn-gold:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px var(--gold);
            color: var(--black);
        }
        .btn-outline-gold {
            background: transparent;
            color: var(--gold);
            border: 2px solid var(--gold);
            border-radius: 50px;
            padding: 8px 25px;
            font-weight: bold;
        }
        .btn-outline-gold:hover {
            background: var(--gold);
            color: var(--black);
        }
        .stats-card {
            background: rgba(0,0,0,0.7);
            border: 1px solid var(--gold-dark);
            border-radius: 15px;
            padding: 15px;
            text-align: center;
        }
        .stats-card h2 {
            color: var(--gold);
        }
        .text-gold {
            color: var(--gold);
        }
        .table {
            color: var(--white);
        }
        .table thead th {
            background: var(--gold-dark);
            color: var(--black);
            border-bottom: none;
        }
        .table tbody tr {
            border-bottom: 1px solid #444;
        }
        .table tbody tr:hover {
            background: rgba(255,215,0,0.05);
        }
        .badge-gold {
            background: var(--gold);
            color: var(--black);
        }
        .form-control, .form-select {
            background: #1a1a1a;
            border: 1px solid #444;
            color: var(--white);
            border-radius: 10px;
        }
        .form-control:focus, .form-select:focus {
            background: #1a1a1a;
            border-color: var(--gold);
            box-shadow: 0 0 0 0.2rem rgba(255,215,0,0.25);
            color: var(--white);
        }
        .form-control::placeholder {
            color: #888;
        }
        .alert-gold {
            background: rgba(255,215,0,0.15);
            border: 1px solid var(--gold);
            color: var(--gold);
            border-radius: 10px;
        }
        .alert-success {
            background: rgba(40,167,69,0.2);
            border-color: #28a745;
            color: #28a745;
        }
        .alert-danger {
            background: rgba(220,53,69,0.2);
            border-color: #dc3545;
            color: #dc3545;
        }
        .section-title {
            border-right: 5px solid var(--gold);
            padding-right: 15px;
            margin-bottom: 20px;
            color: var(--gold);
        }
        .gold-divider {
            height: 2px;
            background: linear-gradient(90deg, var(--gold), transparent);
            margin: 20px 0;
        }
        .chart-container {
            background: rgba(0,0,0,0.5);
            border-radius: 15px;
            padding: 15px;
            border: 1px solid var(--gold-dark);
        }
        footer {
            border-top: 1px solid var(--gold-dark);
            margin-top: 30px;
            padding: 15px 0;
            color: var(--gold);
            text-align: center;
            font-size: 0.9rem;
        }
        .promo-code-safe {
            color: var(--gold);
            font-weight: bold;
            cursor: default;
            user-select: text;
            pointer-events: none;
            display: inline-block;
        }
        .short-link-safe {
            color: var(--gold);
            font-weight: bold;
            cursor: pointer;
            text-decoration: underline;
        }
        .short-link-safe:hover {
            color: var(--yellow);
        }
        .no-click {
            pointer-events: none !important;
        }
        /* عمولة متدرجة */
        .tier-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.8rem;
        }
        .tier-platinum { background: #e5e4e2; color: #000; }
        .tier-gold { background: #FFD700; color: #000; }
        .tier-silver { background: #C0C0C0; color: #000; }
        .tier-bronze { background: #CD7F32; color: #fff; }
        .tier-basic { background: #666; color: #fff; }
        /* رحلة العميل */
        .journey-step {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 15px;
            margin: 2px;
            font-size: 0.75rem;
        }
        .step-click { background: #2196F3; color: #fff; }
        .step-view { background: #4CAF50; color: #fff; }
        .step-cart { background: #FF9800; color: #fff; }
        .step-checkout { background: #9C27B0; color: #fff; }
        .step-purchase { background: #FFD700; color: #000; }
        .step-drop { background: #f44336; color: #fff; }
    </style>
</head>
<body>

<nav class="navbar navbar-expand-lg navbar-dark">
    <div class="container">
        <span class="navbar-brand"><i class="bi bi-shield-shaded"></i> ViralShield AI</span>
        <span class="text-gold">v3.0.4 | نظام متكامل</span>
    </div>
</nav>

<div class="container mt-4">
    <!-- لوحة الإحصائيات -->
    <div class="row" id="statsCards">
        <div class="col-md-3 mb-3">
            <div class="stats-card">
                <h5><i class="bi bi-people text-gold"></i> المروجين</h5>
                <h2 id="totalPromoters">-</h2>
            </div>
        </div>
        <div class="col-md-3 mb-3">
            <div class="stats-card">
                <h5><i class="bi bi-lightning-charge text-gold"></i> تحديات نشطة</h5>
                <h2 id="activeChallenges">-</h2>
            </div>
        </div>
        <div class="col-md-3 mb-3">
            <div class="stats-card">
                <h5><i class="bi bi-trophy text-gold"></i> تحديات ناجحة</h5>
                <h2 id="successfulChallenges">-</h2>
            </div>
        </div>
        <div class="col-md-3 mb-3">
            <div class="stats-card">
                <h5><i class="bi bi-exclamation-triangle text-gold"></i> مروجين معرضين للخطر</h5>
                <h2 id="atRiskPromoters">-</h2>
            </div>
        </div>
    </div>

    <!-- الميزات الجديدة: العمولة المتدرجة والمشاركة -->
    <div class="row">
        <div class="col-md-6 mb-3">
            <div class="card p-3">
                <h5 class="text-gold"><i class="bi bi-bar-chart-fill"></i> العمولة المتدرجة</h5>
                <p>معدل العمولة الحالي: <strong id="commissionRateDisplay" class="text-gold">10%</strong></p>
                <p>المستوى: <span id="tierDisplay" class="tier-badge tier-basic">أساسي</span></p>
                <small>كلما زادت مبيعاتك الشهرية، ترتفع نسبة عمولتك!</small>
            </div>
        </div>
        <div class="col-md-6 mb-3">
            <div class="card p-3">
                <h5 class="text-gold"><i class="bi bi-share-fill"></i> مشاركة اجتماعية</h5>
                <div class="d-flex gap-2 flex-wrap">
                    <button class="btn btn-gold btn-sm" onclick="shareOnSocial('twitter')"><i class="bi bi-twitter"></i> تويتر</button>
                    <button class="btn btn-gold btn-sm" onclick="shareOnSocial('facebook')"><i class="bi bi-facebook"></i> فيسبوك</button>
                    <button class="btn btn-gold btn-sm" onclick="shareOnSocial('whatsapp')"><i class="bi bi-whatsapp"></i> واتساب</button>
                    <button class="btn btn-gold btn-sm" onclick="shareOnSocial('telegram')"><i class="bi bi-telegram"></i> تيليجرام</button>
                </div>
                <div id="shareResult" class="mt-2"></div>
            </div>
        </div>
    </div>

    <!-- الروابط القصيرة والمنتجات -->
    <div class="row">
        <div class="col-md-6 mb-3">
            <div class="card p-3">
                <h5 class="text-gold"><i class="bi bi-link-45deg"></i> روابط قصيرة</h5>
                <p>إجمالي النقرات: <strong id="totalClicks" class="text-gold">-</strong></p>
                <p>التحويلات: <strong id="totalConversions" class="text-gold">-</strong></p>
                <p>معدل التحويل: <strong id="conversionRate" class="text-gold">-</strong>%</p>
            </div>
        </div>
        <div class="col-md-6 mb-3">
            <div class="card p-3">
                <h5 class="text-gold"><i class="bi bi-box-seam"></i> المنتجات</h5>
                <p>إجمالي المنتجات: <strong id="totalProducts" class="text-gold">-</strong></p>
                <p>المنتجات النشطة: <strong id="activeProducts" class="text-gold">-</strong></p>
            </div>
        </div>
    </div>

    <!-- رسوم بيانية -->
    <div class="row">
        <div class="col-md-6 mb-3">
            <div class="card p-3">
                <h5 class="text-gold"><i class="bi bi-bar-chart"></i> النقرات اليومية</h5>
                <div class="chart-container">
                    <canvas id="clicksChart"></canvas>
                </div>
            </div>
        </div>
        <div class="col-md-6 mb-3">
            <div class="card p-3">
                <h5 class="text-gold"><i class="bi bi-pie-chart"></i> توزيع الأجهزة</h5>
                <div class="chart-container">
                    <canvas id="deviceChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <!-- رحلة العميل (جديد رقم 11) -->
    <div class="card mt-4 p-4">
        <h5 class="text-gold"><i class="bi bi-diagram-3"></i> رحلة العميل (Customer Journey)</h5>
        <div class="table-responsive">
            <table class="table">
                <thead><tr><th>العميل</th><th>المنتج</th><th>المروج</th><th>الخطوات</th><th>الوقت (ث)</th><th>الحالة</th></tr></thead>
                <tbody id="journeyTable"></tbody>
            </table>
        </div>
    </div>

    <!-- الكوبونات (جديد رقم 3) -->
    <div class="card mt-4 p-4">
        <h5 class="text-gold"><i class="bi bi-ticket-perforated"></i> الكوبونات الديناميكية</h5>
        <div class="table-responsive">
            <table class="table">
                <thead><tr><th>الكود</th><th>المنتج</th><th>المروج</th><th>الخصم</th><th>الحالة</th><th>انتهاء</th></tr></thead>
                <tbody id="couponsTable"></tbody>
            </table>
        </div>
    </div>

    <!-- قسم المنتجات -->
    <div class="card mt-4 p-4">
        <h5 class="text-gold"><i class="bi bi-box"></i> إدارة المنتجات</h5>
        <div class="row g-3">
            <div class="col-md-3">
                <input type="text" id="productName" class="form-control" placeholder="اسم المنتج">
            </div>
            <div class="col-md-3">
                <textarea id="productDesc" class="form-control" placeholder="وصف المنتج" rows="1"></textarea>
            </div>
            <div class="col-md-2">
                <input type="number" id="productPrice" class="form-control" placeholder="سعر البيع">
            </div>
            <div class="col-md-2">
                <input type="number" id="productCost" class="form-control" placeholder="التكلفة">
            </div>
            <div class="col-md-2">
                <button class="btn btn-gold w-100" onclick="addProduct()"><i class="bi bi-plus-lg"></i> إضافة</button>
            </div>
        </div>
        <div id="productResult" class="mt-3"></div>
        <div class="table-responsive mt-3">
            <table class="table">
                <thead><tr><th>#</th><th>الاسم</th><th>السعر</th><th>التكلفة</th><th>التاجر</th><th>الحالة</th></tr></thead>
                <tbody id="productsTable"></tbody>
            </table>
        </div>
    </div>

    <!-- قسم الروابط القصيرة -->
    <div class="card mt-4 p-4">
        <h5 class="text-gold"><i class="bi bi-link"></i> إنشاء رابط قصير</h5>
        <div class="row g-3">
            <div class="col-md-4">
                <select id="shortProductSelect" class="form-select">
                    <option value="">اختر منتجاً</option>
                </select>
            </div>
            <div class="col-md-4">
                <input type="text" id="promoterCodeForShort" class="form-control" placeholder="كود المروج">
            </div>
            <div class="col-md-4">
                <button class="btn btn-gold w-100" onclick="createShortLink()"><i class="bi bi-link-45deg"></i> إنشاء رابط</button>
            </div>
        </div>
        <div id="shortLinkResult" class="mt-3"></div>
        <div class="table-responsive mt-3">
            <table class="table">
                <thead><tr><th>الكود</th><th>المنتج</th><th>المروج</th><th>النقرات</th><th>التحويلات</th><th>نسبة التحويل</th><th>الرابط</th></tr></thead>
                <tbody id="shortLinksTable"></tbody>
            </table>
        </div>
    </div>

    <!-- قسم الفرق -->
    <div class="card mt-4 p-4">
        <h5 class="text-gold"><i class="bi bi-people-fill"></i> الفرق التسويقية</h5>
        <div class="row g-3">
            <div class="col-md-4">
                <input type="text" id="teamName" class="form-control" placeholder="اسم الفريق">
            </div>
            <div class="col-md-4">
                <textarea id="teamDesc" class="form-control" placeholder="وصف الفريق" rows="1"></textarea>
            </div>
            <div class="col-md-4">
                <button class="btn btn-gold w-100" onclick="createTeam()"><i class="bi bi-plus-circle"></i> إنشاء فريق</button>
            </div>
        </div>
        <div id="teamResult" class="mt-3"></div>
        <div class="table-responsive mt-3">
            <table class="table">
                <thead><tr><th>#</th><th>اسم الفريق</th><th>المنشئ</th><th>الأعضاء</th><th>الحالة</th><th>انضم</th></tr></thead>
                <tbody id="teamsTable"></tbody>
            </table>
        </div>
    </div>

    <!-- باقي الأقسام -->
    <div class="row mt-4">
        <div class="col-md-6">
            <div class="card p-3">
                <h5 class="text-gold"><i class="bi bi-person-plus"></i> تسجيل مروج جديد</h5>
                <div class="row g-3">
                    <div class="col-md-5">
                        <input type="text" id="promoterName" class="form-control" placeholder="الاسم الكامل">
                    </div>
                    <div class="col-md-5">
                        <input type="email" id="promoterEmail" class="form-control" placeholder="البريد الإلكتروني">
                    </div>
                    <div class="col-md-2">
                        <button class="btn btn-gold w-100" onclick="registerPromoter()"><i class="bi bi-check-lg"></i></button>
                    </div>
                </div>
                <div id="registerResult" class="mt-3"></div>
            </div>
        </div>
        <div class="col-md-6">
            <div class="card p-3">
                <h5 class="text-gold"><i class="bi bi-rocket"></i> إنشاء تحدي جماعي</h5>
                <div class="row g-2">
                    <div class="col-md-3"><input type="text" id="challengePromoCode" class="form-control" placeholder="كود المروج"></div>
                    <div class="col-md-2"><input type="number" id="challengePrice" class="form-control" placeholder="سعر المنتج"></div>
                    <div class="col-md-2"><input type="number" id="challengeCost" class="form-control" placeholder="التكلفة"></div>
                    <div class="col-md-2"><input type="number" id="requiredBuyers" class="form-control" placeholder="عدد المشترين" value="3"></div>
                    <div class="col-md-2"><input type="number" id="durationHours" class="form-control" placeholder="المدة" value="3"></div>
                    <div class="col-md-1"><button class="btn btn-gold w-100" onclick="createChallenge()"><i class="bi bi-rocket"></i></button></div>
                </div>
                <div id="challengeResult" class="mt-3"></div>
            </div>
        </div>
    </div>

    <!-- قائمة التحديات النشطة -->
    <div class="card mt-4 p-4">
        <h5 class="text-gold"><i class="bi bi-list-task"></i> التحديات النشطة</h5>
        <div class="table-responsive">
            <table class="table">
                <thead><tr><th>#</th><th>كود المروج</th><th>المطلوب</th><th>المنضم</th><th>الوقت المتبقي</th><th>الحالة</th></tr></thead>
                <tbody id="challengesTable"></tbody>
            </table>
        </div>
    </div>

    <!-- قائمة المروجين -->
    <div class="card mt-4 p-4">
        <h5 class="text-gold"><i class="bi bi-person-badge"></i> المروجون وتحليل الخمول</h5>
        <div class="table-responsive">
            <table class="table">
                <thead><tr><th>الاسم</th><th>الكود</th><th>المبيعات</th><th>الشهرية</th><th>الأرباح</th><th>المستوى</th><th>الدرجة</th><th>نسبة النشاط</th><th>خطر الخمول</th><th>إجراء</th></tr></thead>
                <tbody id="promotersTable"></tbody>
            </table>
        </div>
    </div>

    <!-- حاسبة العمولة الذكية -->
    <div class="card mt-4 p-4">
        <h5 class="text-gold"><i class="bi bi-calculator"></i> حساب العمولة والخصم الديناميكي</h5>
        <div class="row g-3">
            <div class="col-md-2"><input type="number" id="calcPrice" class="form-control" placeholder="سعر المنتج" value="100"></div>
            <div class="col-md-2"><input type="number" id="calcCost" class="form-control" placeholder="التكلفة" value="60"></div>
            <div class="col-md-2"><select id="viralSuccess" class="form-select"><option value="false">شراء عادي</option><option value="true">نجاح تحدي</option></select></div>
            <div class="col-md-2"><input type="number" id="promoterPerf" class="form-control" placeholder="أداء المروج (0-1)" value="0.5" step="0.1"></div>
            <div class="col-md-2"><input type="number" id="orderVol" class="form-control" placeholder="حجم الطلب" value="1"></div>
            <div class="col-md-2"><button class="btn btn-gold w-100" onclick="calculateCommission()"><i class="bi bi-calculator"></i> احسب</button></div>
        </div>
        <div id="calcResult" class="mt-3 alert alert-gold"></div>
    </div>
</div>

<!-- تذييل الصفحة -->
<footer>
    إعداد وتصميم م/ وسيم الحميدي &copy; 2026
</footer>

<script>
    // ===== دوال مساعدة =====
    async function fetchAPI(url, options={}) {
        const res = await fetch(url, options);
        return await res.json();
    }

    // ===== تحميل لوحة التحكم =====
    async function loadDashboard() {
        try {
            const data = await fetchAPI('/api/dashboard/stats');
            if (data.overview) {
                document.getElementById('totalPromoters').innerText = data.overview.total_promoters;
                document.getElementById('activeChallenges').innerText = data.overview.active_challenges;
                document.getElementById('successfulChallenges').innerText = data.overview.successful_challenges;
                document.getElementById('atRiskPromoters').innerText = data.overview.at_risk_promoters;
                document.getElementById('totalClicks').innerText = data.overview.total_clicks || 0;
                document.getElementById('totalConversions').innerText = data.overview.total_conversions || 0;
                document.getElementById('conversionRate').innerText = data.overview.conversion_rate || 0;
                document.getElementById('totalProducts').innerText = data.overview.total_products || 0;
                document.getElementById('activeProducts').innerText = data.overview.active_products || 0;
            }
            if (data.promoter_stats) {
                document.getElementById('commissionRateDisplay').innerText = data.promoter_stats.commission_rate + '%';
                document.getElementById('tierDisplay').innerText = data.promoter_stats.tier;
                document.getElementById('tierDisplay').className = 'tier-badge tier-' + data.promoter_stats.tier.toLowerCase();
            }
        } catch(e) { console.error(e); }
        loadChallenges();
        loadPromoters();
        loadProducts();
        loadShortLinks();
        loadTeams();
        loadCharts();
        loadProductSelect();
        loadJourneys();
        loadCoupons();
    }

    // ===== رحلة العميل (جديد) =====
    async function loadJourneys() {
        try {
            const journeys = await fetchAPI('/api/journeys');
            const tbody = document.getElementById('journeyTable');
            tbody.innerHTML = '';
            journeys.forEach(j => {
                const steps = j.steps || [];
                let stepHtml = steps.map(s => {
                    let cls = 'step-' + (s.step || 'click');
                    return `<span class="journey-step ${cls}">${s.step || 'نقرة'}</span>`;
                }).join(' ');
                let status = j.converted ? '✅ تحول' : (j.dropped_at ? '❌ انسحاب' : '⏳ قيد التقدم');
                let row = `<tr>
                    <td>${j.customer_id}</td>
                    <td>${j.product_name}</td>
                    <td>${j.promoter_name}</td>
                    <td>${stepHtml}</td>
                    <td>${j.total_time_seconds}</td>
                    <td>${status}</td>
                </tr>`;
                tbody.innerHTML += row;
            });
        } catch(e) { console.error(e); }
    }

    // ===== الكوبونات (جديد) =====
    async function loadCoupons() {
        try {
            const coupons = await fetchAPI('/api/coupons');
            const tbody = document.getElementById('couponsTable');
            tbody.innerHTML = '';
            coupons.forEach(c => {
                let status = c.is_used ? 'مستخدم' : 'نشط';
                let row = `<tr>
                    <td><span class="text-gold">${c.code}</span></td>
                    <td>${c.product_name}</td>
                    <td>${c.promoter_name}</td>
                    <td>${c.is_percentage ? c.discount_percentage + '%' : c.discount_amount + ' ر.س'}</td>
                    <td>${status}</td>
                    <td>${c.expires_at ? new Date(c.expires_at).toLocaleDateString('ar-SA') : 'لا ينتهي'}</td>
                </tr>`;
                tbody.innerHTML += row;
            });
        } catch(e) { console.error(e); }
    }

    // ===== التحديات =====
    async function loadChallenges() {
        try {
            const challenges = await fetchAPI('/api/challenges/active');
            const tbody = document.getElementById('challengesTable');
            tbody.innerHTML = '';
            challenges.forEach(ch => {
                let row = `<tr>
                    <td>${ch.id}</td>
                    <td>${ch.promo_code}</td>
                    <td>${ch.required_buyers}</td>
                    <td>${ch.current_buyers}</td>
                    <td>${ch.remaining_time || 'انتهى'}</td>
                    <td><span class="badge bg-gold text-black">${ch.status}</span></td>
                </tr>`;
                tbody.innerHTML += row;
            });
        } catch(e) { console.error(e); }
    }

    // ===== المروجين مع العمولة المتدرجة =====
    async function loadPromoters() {
        try {
            const promoters = await fetchAPI('/api/promoters/all');
            const tbody = document.getElementById('promotersTable');
            tbody.innerHTML = '';
            for (let p of promoters) {
                let riskBadge = p.is_at_risk ? '<span class="badge bg-danger">خطر</span>' : '<span class="badge bg-success">نشط</span>';
                let tierClass = 'tier-' + (p.tier || 'basic').toLowerCase();
                let row = `<tr>
                    <td>${p.name}</td>
                    <td><span class="promo-code-safe">${p.promo_code}</span></td>
                    <td>${p.total_sales}</td>
                    <td>${p.monthly_sales || 0}</td>
                    <td>${p.total_earnings}</td>
                    <td><span class="badge bg-gold text-black">${p.level}</span></td>
                    <td><span class="tier-badge ${tierClass}">${p.tier || 'أساسي'}</span></td>
                    <td>${p.activity_score}%</td>
                    <td>${riskBadge}</td>
                    <td><button class="btn btn-sm btn-outline-gold" onclick="predictChurn(${p.id})"><i class="bi bi-graph-up"></i></button></td>
                </tr>`;
                tbody.innerHTML += row;
            }
        } catch(e) { console.error(e); }
    }

    // ===== المنتجات =====
    async function loadProducts() {
        try {
            const products = await fetchAPI('/api/products');
            const tbody = document.getElementById('productsTable');
            tbody.innerHTML = '';
            products.forEach(p => {
                let row = `<tr>
                    <td>${p.id}</td>
                    <td>${p.name}</td>
                    <td>${p.price}</td>
                    <td>${p.cost}</td>
                    <td>${p.merchant_name || p.merchant_id}</td>
                    <td>${p.is_active ? 'نشط' : 'غير نشط'}</td>
                </tr>`;
                tbody.innerHTML += row;
            });
        } catch(e) { console.error(e); }
    }

    async function addProduct() {
        const name = document.getElementById('productName').value;
        const desc = document.getElementById('productDesc').value;
        const price = parseFloat(document.getElementById('productPrice').value);
        const cost = parseFloat(document.getElementById('productCost').value);
        if (!name || isNaN(price) || isNaN(cost)) { alert('يرجى ملء جميع الحقول'); return; }
        const data = await fetchAPI('/api/product/create', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, description: desc, price, cost, merchant_id: 1})
        });
        if (data.product) {
            document.getElementById('productResult').innerHTML = `<div class="alert alert-success">✅ تم إضافة المنتج</div>`;
            loadDashboard();
        } else {
            document.getElementById('productResult').innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
        }
    }

    // ===== الروابط القصيرة =====
    async function loadShortLinks() {
        try {
            const links = await fetchAPI('/api/shortlinks');
            const tbody = document.getElementById('shortLinksTable');
            tbody.innerHTML = '';
            links.forEach(link => {
                let row = `<tr>
                    <td>${link.code}</td>
                    <td>${link.product_name}</td>
                    <td>${link.promoter_name}</td>
                    <td>${link.clicks_count}</td>
                    <td>${link.conversions}</td>
                    <td>${link.conversion_rate}%</td>
                    <td><a href="/r/${link.code}" target="_blank" class="short-link-safe">/r/${link.code}</a></td>
                </tr>`;
                tbody.innerHTML += row;
            });
        } catch(e) { console.error(e); }
    }

    async function createShortLink() {
        const product_id = document.getElementById('shortProductSelect').value;
        const promo_code = document.getElementById('promoterCodeForShort').value;
        if (!product_id || !promo_code) { alert('يرجى اختيار منتج وإدخال كود المروج'); return; }
        const data = await fetchAPI('/api/shortlink/create', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({product_id, promo_code})
        });
        if (data.short_link) {
            document.getElementById('shortLinkResult').innerHTML = `<div class="alert alert-success">✅ رابط قصير: <a href="/r/${data.short_link.code}" target="_blank" class="text-gold">/r/${data.short_link.code}</a></div>`;
            loadDashboard();
        } else {
            document.getElementById('shortLinkResult').innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
        }
    }

    async function loadProductSelect() {
        try {
            const products = await fetchAPI('/api/products');
            const sel = document.getElementById('shortProductSelect');
            sel.innerHTML = '<option value="">اختر منتجاً</option>';
            products.forEach(p => {
                sel.innerHTML += `<option value="${p.id}">${p.name}</option>`;
            });
        } catch(e) { console.error(e); }
    }

    // ===== الفرق =====
    async function loadTeams() {
        try {
            const teams = await fetchAPI('/api/teams');
            const tbody = document.getElementById('teamsTable');
            tbody.innerHTML = '';
            teams.forEach(t => {
                let row = `<tr>
                    <td>${t.id}</td>
                    <td>${t.name}</td>
                    <td>${t.creator_name}</td>
                    <td>${t.members_count}</td>
                    <td>${t.is_active ? 'نشط' : 'غير نشط'}</td>
                    <td><button class="btn btn-sm btn-gold" onclick="joinTeam(${t.id})"><i class="bi bi-person-plus"></i> انضم</button></td>
                </tr>`;
                tbody.innerHTML += row;
            });
        } catch(e) { console.error(e); }
    }

    async function createTeam() {
        const name = document.getElementById('teamName').value;
        const desc = document.getElementById('teamDesc').value;
        if (!name) { alert('يرجى إدخال اسم الفريق'); return; }
        const data = await fetchAPI('/api/team/create', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, description: desc, promoter_id: 1})
        });
        if (data.team) {
            document.getElementById('teamResult').innerHTML = `<div class="alert alert-success">✅ تم إنشاء الفريق</div>`;
            loadDashboard();
        } else {
            document.getElementById('teamResult').innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
        }
    }

    async function joinTeam(teamId) {
        const data = await fetchAPI(`/api/team/join/${teamId}`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({promoter_id: 1})
        });
        if (data.success) {
            alert('تم الانضمام للفريق');
            loadDashboard();
        } else {
            alert(data.error);
        }
    }

    // ===== الرسوم البيانية =====
    async function loadCharts() {
        try {
            const stats = await fetchAPI('/api/analytics/charts');
            const ctx1 = document.getElementById('clicksChart').getContext('2d');
            new Chart(ctx1, {
                type: 'bar',
                data: {
                    labels: stats.daily_labels || ['اليوم'],
                    datasets: [{
                        label: 'النقرات',
                        data: stats.daily_clicks || [0],
                        backgroundColor: 'rgba(255,215,0,0.6)',
                        borderColor: '#FFD700',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { labels: { color: '#fff' } }
                    },
                    scales: {
                        x: { ticks: { color: '#fff' } },
                        y: { ticks: { color: '#fff' } }
                    }
                }
            });
            const ctx2 = document.getElementById('deviceChart').getContext('2d');
            new Chart(ctx2, {
                type: 'pie',
                data: {
                    labels: stats.device_labels || ['محطة'],
                    datasets: [{
                        data: stats.device_data || [1],
                        backgroundColor: ['#FFD700', '#FFC107', '#D4A017', '#FF8F00']
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { labels: { color: '#fff' } }
                    }
                }
            });
        } catch(e) { console.error(e); }
    }

    // ===== المشاركة الاجتماعية (رقم 2) =====
    function shareOnSocial(platform) {
        const url = encodeURIComponent(window.location.href);
        const text = encodeURIComponent('🎯 اكتشف أفضل العروض والتحديات على ViralShield AI! انضم الآن واستفد من خصومات رائعة!');
        let shareUrl = '';
        switch(platform) {
            case 'twitter':
                shareUrl = `https://twitter.com/intent/tweet?text=${text}&url=${url}`;
                break;
            case 'facebook':
                shareUrl = `https://www.facebook.com/sharer/sharer.php?u=${url}&quote=${text}`;
                break;
            case 'whatsapp':
                shareUrl = `https://api.whatsapp.com/send?text=${text}%20${url}`;
                break;
            case 'telegram':
                shareUrl = `https://t.me/share/url?url=${url}&text=${text}`;
                break;
            default:
                alert('منصة غير مدعومة');
                return;
        }
        window.open(shareUrl, '_blank', 'width=600,height=500');
        document.getElementById('shareResult').innerHTML = `<div class="alert alert-success">✅ تم فتح نافذة المشاركة على ${platform}</div>`;
    }

    // ===== باقي الدوال =====
    async function registerPromoter() {
        const name = document.getElementById('promoterName').value;
        const email = document.getElementById('promoterEmail').value;
        if (!name || !email) { alert('يرجى إدخال الاسم والبريد'); return; }
        const data = await fetchAPI('/api/promoter/register', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, email})
        });
        document.getElementById('registerResult').innerHTML = `<div class="alert alert-success">✅ ${data.message}<br>الكود: ${data.promoter.promo_code}</div>`;
        loadDashboard();
    }

    async function createChallenge() {
        const promo_code = document.getElementById('challengePromoCode').value;
        const product_price = parseFloat(document.getElementById('challengePrice').value);
        const product_cost = parseFloat(document.getElementById('challengeCost').value);
        const required_buyers = parseInt(document.getElementById('requiredBuyers').value);
        const duration_hours = parseInt(document.getElementById('durationHours').value);
        if (!promo_code || isNaN(product_price) || isNaN(product_cost)) { alert('يرجى ملء جميع الحقول'); return; }
        const data = await fetchAPI('/api/challenge/create', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({buyer_id: 'user_' + Date.now(), promo_code, product_price, product_cost, required_buyers, duration_hours})
        });
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
        const data = await fetchAPI('/api/calculate/commission', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({product_price, product_cost, is_viral_success, promoter_performance, order_volume})
        });
        document.getElementById('calcResult').innerHTML = `<strong>النتيجة:</strong><br>
        السعر النهائي: ${data.calculation.final_price} ر.س<br>
        عمولة المروج: ${data.calculation.promoter_payout} ر.س (${data.calculation.commission_rate_used}%)<br>
        ربح التاجر: ${data.calculation.merchant_secured_profit} ر.س<br>
        نسبة الخصم: ${data.calculation.discount_percentage}%`;
    }

    async function predictChurn(promoterId) {
        const data = await fetchAPI(`/api/ai/predict-churn/${promoterId}`);
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

# ==================== نقاط النهاية API ====================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# ---------- صفحة المنتج ----------
@app.route('/product/<int:product_id>')
def product_page(product_id):
    product = Product.query.get(product_id)
    if not product:
        return "المنتج غير موجود", 404
    return f"""
    <!DOCTYPE html>
    <html dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{product.name}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
        <style>
            body {{ background: #1a1a1a; color: #fff; font-family: 'Tajawal', sans-serif; }}
            .product-card {{ background: rgba(20,20,20,0.9); border: 1px solid #D4A017; border-radius: 20px; padding: 30px; margin-top: 50px; }}
            .text-gold {{ color: #FFD700; }}
            .btn-gold {{ background: #FFD700; color: #000; border: none; border-radius: 50px; padding: 10px 30px; font-weight: bold; }}
            .btn-gold:hover {{ background: #FFC107; color: #000; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="product-card">
                <h1 class="text-gold">{product.name}</h1>
                <p class="text-white-50">{product.description or 'وصف المنتج'}</p>
                <h2 class="text-gold">السعر: {product.price} ر.س</h2>
                <button class="btn-gold mt-3" onclick="alert('تم إضافة المنتج للسلة!')"><i class="bi bi-cart-plus"></i> أضف للسلة</button>
                <br><br>
                <a href="/" class="text-gold"><i class="bi bi-arrow-right"></i> العودة للرئيسية</a>
            </div>
        </div>
        <footer class="text-center text-gold mt-5" style="border-top:1px solid #D4A017; padding:15px;">
            إعداد وتصميم م/ وسيم الحميدي &copy; 2026
        </footer>
        <script>
            // تتبع رحلة العميل (جديد)
            const customerId = 'c' + Date.now();
            const sessionId = 's' + Date.now() + Math.random().toString(36).substr(2, 5);
            fetch('/api/journey/track', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{
                    customer_id: customerId,
                    session_id: sessionId,
                    short_link_id: 1,
                    product_id: {product.id},
                    promoter_id: 1,
                    page_visited: 'product_page',
                    step: 'view',
                    time_spent: 5
                }})
            }});
        </script>
    </body>
    </html>
    """

# ---------- نقاط API الحالية ----------
@app.route('/api/promoters/all', methods=['GET'])
def get_all_promoters():
    promoters = Promoter.query.order_by(Promoter.id.desc()).all()
    return jsonify([p.to_dict() for p in promoters])

@app.route('/api/challenges/active', methods=['GET'])
def get_active_challenges():
    challenges = ViralGroupChallenge.query.filter_by(status='ACTIVE').order_by(ViralGroupChallenge.created_at.desc()).all()
    return jsonify([c.to_dict() for c in challenges])

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
            duration_hours=data.get('duration_hours', 3),
            team_id=data.get('team_id')
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
            device_hash=device_hash,
            customer_id=data.get('customer_id')
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
        # جلب المروج (استخدام بيانات افتراضية أو من الطلب)
        promoter = Promoter.query.filter_by(promo_code=data.get('promo_code', 'PROMO2024')).first()
        result = engine.calculate_smart_margin(
            product_price=data.get('product_price', 100),
            product_cost=data.get('product_cost', 60),
            is_viral_success=data.get('is_viral_success', False),
            promoter_performance=data.get('promoter_performance', 0.5),
            order_volume=data.get('order_volume', 1),
            promoter=promoter
        )
        return jsonify({'calculation': result})
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
        total_clicks = db.session.query(db.func.sum(ShortLink.clicks_count)).scalar() or 0
        total_conversions = db.session.query(db.func.sum(ShortLink.conversions)).scalar() or 0
        conversion_rate = round((total_conversions / total_clicks * 100) if total_clicks > 0 else 0, 2)
        total_products = Product.query.count()
        active_products = Product.query.filter_by(is_active=True).count()
        # إحصاءات العمولة المتدرجة
        promoter = Promoter.query.first()
        commission_rate = engine.get_commission_rate(promoter) * 100 if promoter else 10
        tier = promoter.tier if promoter else "أساسي"
        return jsonify({
            'overview': {
                'total_promoters': total_promoters,
                'active_challenges': active_challenges,
                'successful_challenges': successful_challenges,
                'total_transactions': total_transactions,
                'at_risk_promoters': at_risk_promoters,
                'total_clicks': total_clicks,
                'total_conversions': total_conversions,
                'conversion_rate': conversion_rate,
                'total_products': total_products,
                'active_products': active_products
            },
            'financials': {
                'total_sales': round(total_sales, 2),
                'total_merchant_profit': round(total_merchant_profit, 2)
            },
            'promoter_stats': {
                'commission_rate': round(commission_rate, 1),
                'tier': tier
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

# ---------- نقاط API الجديدة ----------
@app.route('/api/products', methods=['GET'])
def list_products():
    products = Product.query.filter_by(is_active=True).all()
    return jsonify([p.to_dict() for p in products])

@app.route('/api/product/create', methods=['POST'])
def create_product():
    try:
        data = request.json
        merchant = Promoter.query.get(data.get('merchant_id'))
        if not merchant or not merchant.is_merchant:
            return jsonify({'error': 'التاجر غير موجود أو غير مفعل'}), 400
        product = Product(
            name=data['name'],
            description=data.get('description', ''),
            price=data['price'],
            cost=data['cost'],
            merchant_id=merchant.id
        )
        db.session.add(product)
        db.session.commit()
        return jsonify({'message': 'تم إضافة المنتج', 'product': product.to_dict()}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/shortlinks', methods=['GET'])
def list_shortlinks():
    links = ShortLink.query.order_by(ShortLink.created_at.desc()).all()
    return jsonify([link.to_dict() for link in links])

@app.route('/api/shortlink/create', methods=['POST'])
def create_shortlink():
    try:
        data = request.json
        product = Product.query.get(data['product_id'])
        if not product:
            return jsonify({'error': 'المنتج غير موجود'}), 404
        promoter = Promoter.query.filter_by(promo_code=data['promo_code']).first()
        if not promoter:
            return jsonify({'error': 'كود المروج غير صالح'}), 404
        code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        while ShortLink.query.filter_by(code=code).first():
            code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        short_link = ShortLink(
            code=code,
            product_id=product.id,
            promoter_id=promoter.id
        )
        db.session.add(short_link)
        db.session.commit()
        return jsonify({'message': 'تم إنشاء الرابط القصير', 'short_link': short_link.to_dict()}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/r/<code>')
def redirect_shortlink(code):
    short_link = ShortLink.query.filter_by(code=code, is_active=True).first()
    if not short_link:
        return "⚠️ الرابط غير صالح أو منتهي الصلاحية", 404
    # تسجيل النقرة مع رحلة العميل
    customer_id = request.args.get('cid', f"c{datetime.utcnow().timestamp()}")
    session_id = request.args.get('sid', f"s{datetime.utcnow().timestamp()}{secrets.token_hex(3)}")
    click = Click(
        short_link_id=short_link.id,
        ip_address=request.remote_addr,
        user_agent=request.headers.get('User-Agent'),
        referer=request.headers.get('Referer'),
        device_type='mobile' if 'Mobile' in request.headers.get('User-Agent', '') else 'desktop',
        browser=request.headers.get('User-Agent', '')[:50],
        customer_id=customer_id,
        session_id=session_id,
        step='click',
        page_visited='short_link_redirect'
    )
    db.session.add(click)
    short_link.clicks_count += 1
    db.session.commit()
    # تتبع رحلة العميل
    engine.track_customer_journey(
        customer_id=customer_id,
        session_id=session_id,
        short_link_id=short_link.id,
        product_id=short_link.product_id,
        promoter_id=short_link.promoter_id,
        page_visited='redirect',
        step='click',
        time_spent=0
    )
    return redirect(url_for('product_page', product_id=short_link.product_id) + f'?cid={customer_id}&sid={session_id}')

@app.route('/api/click/convert', methods=['POST'])
def convert_click():
    try:
        data = request.json
        click = Click.query.get(data['click_id'])
        if not click:
            return jsonify({'error': 'النقرة غير موجودة'}), 404
        click.converted = True
        click.converted_at = datetime.utcnow()
        short_link = ShortLink.query.get(click.short_link_id)
        if short_link:
            short_link.conversions += 1
        # تحديث رحلة العميل
        journey = CustomerJourney.query.filter_by(customer_id=click.customer_id, session_id=click.session_id).first()
        if journey:
            journey.converted = True
            journey.converted_at = datetime.utcnow()
        db.session.commit()
        return jsonify({'message': 'تم تحديث النقرة إلى محولة'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/teams', methods=['GET'])
def list_teams():
    teams = Team.query.filter_by(is_active=True).all()
    return jsonify([t.to_dict() for t in teams])

@app.route('/api/team/create', methods=['POST'])
def create_team():
    try:
        data = request.json
        promoter = Promoter.query.get(data['promoter_id'])
        if not promoter:
            return jsonify({'error': 'المروج غير موجود'}), 404
        team = Team(
            name=data['name'],
            description=data.get('description', ''),
            created_by=promoter.id
        )
        db.session.add(team)
        db.session.commit()
        member = TeamMember(team_id=team.id, promoter_id=promoter.id, role='leader')
        db.session.add(member)
        db.session.commit()
        return jsonify({'message': 'تم إنشاء الفريق', 'team': team.to_dict()}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/team/join/<int:team_id>', methods=['POST'])
def join_team(team_id):
    try:
        data = request.json
        promoter = Promoter.query.get(data['promoter_id'])
        if not promoter:
            return jsonify({'error': 'المروج غير موجود'}), 404
        team = Team.query.get(team_id)
        if not team:
            return jsonify({'error': 'الفريق غير موجود'}), 404
        existing = TeamMember.query.filter_by(team_id=team_id, promoter_id=promoter.id).first()
        if existing:
            return jsonify({'error': 'أنت بالفعل عضو في هذا الفريق'}), 400
        member = TeamMember(team_id=team_id, promoter_id=promoter.id)
        db.session.add(member)
        db.session.commit()
        return jsonify({'success': True, 'message': 'تم الانضمام للفريق'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/charts', methods=['GET'])
def chart_data():
    try:
        daily_clicks = []
        labels = []
        for i in range(6, -1, -1):
            day = datetime.utcnow().date() - timedelta(days=i)
            start = datetime(day.year, day.month, day.day)
            end = start + timedelta(days=1)
            count = Click.query.filter(Click.timestamp >= start, Click.timestamp < end).count()
            daily_clicks.append(count)
            labels.append(day.strftime('%Y-%m-%d'))
        mobile_count = Click.query.filter(Click.device_type == 'mobile').count()
        desktop_count = Click.query.filter(Click.device_type == 'desktop').count()
        other_count = Click.query.filter(Click.device_type.notin_(['mobile', 'desktop'])).count()
        return jsonify({
            'daily_labels': labels,
            'daily_clicks': daily_clicks,
            'device_labels': ['جوال', 'حاسوب', 'أخرى'],
            'device_data': [mobile_count, desktop_count, other_count]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------- نقاط API للميزات الجديدة ----------

# 3. الكوبونات
@app.route('/api/coupons', methods=['GET'])
def list_coupons():
    coupons = Coupon.query.order_by(Coupon.created_at.desc()).all()
    return jsonify([c.to_dict() for c in coupons])

@app.route('/api/coupon/validate', methods=['POST'])
def validate_coupon():
    try:
        data = request.json
        coupon = Coupon.query.filter_by(code=data['code'], is_used=False).first()
        if not coupon:
            return jsonify({'valid': False, 'error': 'كوبون غير صالح أو مستخدم'}), 404
        if coupon.expires_at and datetime.utcnow() > coupon.expires_at:
            return jsonify({'valid': False, 'error': 'انتهت صلاحية الكوبون'}), 400
        if coupon.use_count >= coupon.max_uses:
            return jsonify({'valid': False, 'error': 'تم استخدام الكوبون الحد الأقصى'}), 400
        return jsonify({
            'valid': True,
            'coupon': coupon.to_dict(),
            'discount': coupon.discount_percentage if coupon.is_percentage else coupon.discount_amount
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/coupon/use', methods=['POST'])
def use_coupon():
    try:
        data = request.json
        coupon = Coupon.query.filter_by(code=data['code']).first()
        if not coupon:
            return jsonify({'error': 'كوبون غير صالح'}), 404
        if coupon.is_used:
            return jsonify({'error': 'الكوبون مستخدم مسبقاً'}), 400
        if coupon.expires_at and datetime.utcnow() > coupon.expires_at:
            return jsonify({'error': 'انتهت صلاحية الكوبون'}), 400
        coupon.is_used = True
        coupon.used_by = data.get('used_by')
        coupon.used_at = datetime.utcnow()
        coupon.use_count += 1
        db.session.commit()
        return jsonify({'message': 'تم استخدام الكوبون بنجاح', 'coupon': coupon.to_dict()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 11. رحلة العميل
@app.route('/api/journeys', methods=['GET'])
def list_journeys():
    journeys = CustomerJourney.query.order_by(CustomerJourney.first_click.desc()).limit(50).all()
    return jsonify([j.to_dict() for j in journeys])

@app.route('/api/journey/track', methods=['POST'])
def track_journey():
    try:
        data = request.json
        journey = engine.track_customer_journey(
            customer_id=data['customer_id'],
            session_id=data['session_id'],
            short_link_id=data.get('short_link_id', 1),
            product_id=data['product_id'],
            promoter_id=data.get('promoter_id', 1),
            page_visited=data.get('page_visited', 'unknown'),
            step=data.get('step', 'view'),
            time_spent=data.get('time_spent', 0)
        )
        return jsonify({'success': True, 'journey': journey.to_dict()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/journey/analytics', methods=['GET'])
def journey_analytics():
    try:
        # إحصائيات سريعة عن رحلة العميل
        total_journeys = CustomerJourney.query.count()
        converted = CustomerJourney.query.filter_by(converted=True).count()
        dropped = CustomerJourney.query.filter(CustomerJourney.dropped_at.isnot(None)).count()
        conversion_rate = round((converted / total_journeys * 100) if total_journeys > 0 else 0, 2)
        # أكثر خطوة يتخلى فيها العميل
        steps_data = {}
        journeys = CustomerJourney.query.all()
        for j in journeys:
            steps = json.loads(j.steps) if j.steps else []
            for s in steps:
                step_name = s.get('step', 'unknown')
                steps_data[step_name] = steps_data.get(step_name, 0) + 1
        return jsonify({
            'total_journeys': total_journeys,
            'converted': converted,
            'dropped': dropped,
            'conversion_rate': conversion_rate,
            'step_frequency': steps_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== تهيئة قاعدة البيانات ====================
def init_database():
    with app.app_context():
        db.create_all()
        if Promoter.query.count() == 0:
            demo_merchant = Promoter(
                name="أحمد التاجر",
                email="merchant@viralshield.com",
                promo_code="MERCHANT2024",
                base_commission_rate=0.10,
                activity_score=85.0,
                is_merchant=True,
                points=1000,
                level="ذهبي",
                monthly_sales=60,
                tier="ذهبي"
            )
            db.session.add(demo_merchant)
            demo_promoter = Promoter(
                name="سارة المسوقة",
                email="promoter@viralshield.com",
                promo_code="PROMO2024",
                base_commission_rate=0.15,
                activity_score=90.0,
                points=200,
                level="فضي",
                monthly_sales=25,
                tier="فضي"
            )
            db.session.add(demo_promoter)
            db.session.commit()
            product = Product(
                name="هاتف ذكي X100",
                description="هاتف متطور بمواصفات عالية",
                price=999.0,
                cost=700.0,
                merchant_id=demo_merchant.id
            )
            db.session.add(product)
            db.session.commit()
            # إنشاء كوبون تجريبي
            coupon = Coupon(
                code="TEST-VIP-2024",
                product_id=product.id,
                promoter_id=demo_promoter.id,
                discount_percentage=15,
                is_percentage=True,
                expires_at=datetime.utcnow() + timedelta(days=30),
                max_uses=5
            )
            db.session.add(coupon)
            db.session.commit()
            print("✅ تم إنشاء المستخدمين التجريبيين والمنتجات والكوبونات")
        print("✅ قاعدة البيانات جاهزة وجميع الجداول موجودة")

init_database()

# ==================== التشغيل ====================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"""
╔══════════════════════════════════════════════════════════╗
║     🚀 ViralShield AI Engine - النسخة المتكاملة         ║
║     📍 Running on: http://0.0.0.0:{port}                 ║
║     ✅ تم إضافة الميزات الجديدة:                          ║
║     🔹 المشاركة الاجتماعية (رقم 2)                      ║
║     🔹 الكوبونات الديناميكية (رقم 3)                    ║
║     🔹 العمولة المتدرجة (رقم 7)                         ║
║     🔹 رحلة العميل (رقم 11)                             ║
║     © إعداد وتصميم م/ وسيم الحميدي                      ║
╚══════════════════════════════════════════════════════════╝
    """)
    app.run(host='0.0.0.0', port=port, debug=False)
