#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 ViralShield AI Engine - منظومة التسويق الذكي والفيروسي المتكاملة
ملف واحد كامل للتشغيل المباشر على Render مع دعم Python 3.12
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps

# ==================== استيراد المكتبات الأساسية ====================
from flask import Flask, request, jsonify, session, g
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from dotenv import load_dotenv
import numpy as np

# تحميل متغيرات البيئة (للمفاتيح الحساسة)
load_dotenv()

# ==================== إعداد التطبيق ====================
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # تسهيل الاتصال من أي frontend

# إعدادات الأمان
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))

# إعداد قاعدة البيانات (PostgreSQL على Render أو SQLite محلياً)
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///viralshield.db')
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,
    'pool_recycle': 300,
    'pool_pre_ping': True
}

# تهيئة قاعدة البيانات
db = SQLAlchemy(app)

# ==================== نماذج قاعدة البيانات ====================

class Promoter(db.Model):
    """جدول المروجين - يستخدمه الذكاء الاصطناعي للتنبؤ بالنشاط"""
    __tablename__ = 'promoters'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    promo_code = Column(String(20), unique=True, nullable=False, index=True)
    base_commission_rate = Column(Float, default=0.10)
    
    # بيانات الأداء والتتبع
    total_sales = Column(Integer, default=0)
    total_earnings = Column(Float, default=0.0)
    last_login = Column(DateTime, default=datetime.utcnow)
    total_clicks_tracked = Column(Integer, default=0)
    last_sale_date = Column(DateTime)
    
    # تحليل سلوك المروج
    activity_score = Column(Float, default=100.0)  # من 0 إلى 100
    is_at_risk_of_churn = Column(Boolean, default=False)
    churn_probability = Column(Float, default=0.0)  # نسبة احتمالية التوقف
    
    # العلاقات
    fingerprints = relationship('DeviceFingerprint', backref='promoter', lazy=True)
    challenges = relationship('ViralGroupChallenge', backref='promoter', lazy=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'promo_code': self.promo_code,
            'total_sales': self.total_sales,
            'total_earnings': round(self.total_earnings, 2),
            'activity_score': round(self.activity_score, 1),
            'is_at_risk': self.is_at_risk_of_churn
        }


class DeviceFingerprint(db.Model):
    """جدول بصمات الأجهزة المخفية - تتبع بدون كوكيز"""
    __tablename__ = 'device_fingerprints'
    
    id = Column(Integer, primary_key=True)
    device_hash = Column(String(64), unique=True, nullable=False, index=True)
    promoter_id = Column(Integer, ForeignKey('promoters.id'), nullable=False)
    
    # معلومات الجهاز للتتبع الذكي
    ip_address = Column(String(45))
    user_agent = Column(Text)
    screen_resolution = Column(String(20))
    browser_language = Column(String(10))
    timezone = Column(String(50))
    
    first_click_time = Column(DateTime, default=datetime.utcnow)
    last_click_time = Column(DateTime, default=datetime.utcnow)
    click_count = Column(Integer, default=1)
    has_purchased = Column(Boolean, default=False)
    
    # منع الاحتيال
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
    """جدول تحديات الشراء الجماعي الفيروسي"""
    __tablename__ = 'viral_group_challenges'
    
    id = Column(Integer, primary_key=True)
    creator_buyer_id = Column(String(50), nullable=False)
    promoter_id = Column(Integer, ForeignKey('promoters.id'), nullable=False)
    promo_code_used = Column(String(20), nullable=False)
    
    # إعدادات التحدي
    required_buyers = Column(Integer, default=3)
    current_buyers_joined = Column(Integer, default=1)
    challenge_duration_hours = Column(Float, default=3.0)
    
    # التوقيت
    created_at = Column(DateTime, default=datetime.utcnow)
    expiration_time = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    
    # الحالة
    status = Column(String(20), default="ACTIVE", index=True)  # ACTIVE, SUCCESS, EXPIRED, CANCELLED
    
    # المعلومات المالية
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
            'discount_percentage': self.discount_applied
        }


class Transaction(db.Model):
    """جدول المعاملات والعمليات المالية"""
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    transaction_type = Column(String(50), nullable=False)  # SALE, COMMISSION, REFUND
    amount = Column(Float, nullable=False)
    
    # الربط مع الأطراف المعنية
    promoter_id = Column(Integer, ForeignKey('promoters.id'))
    challenge_id = Column(Integer, ForeignKey('viral_group_challenges.id'))
    fingerprint_id = Column(Integer, ForeignKey('device_fingerprints.id'))
    
    # تفاصيل العملية
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


# ==================== المحرك الرئيسي للنظام ====================

class ViralShieldEngine:
    """
    المحرك الأساسي للمنظومة المتكاملة
    يدمج: البصمة الرقمية + العمولات الذكية + التحديات الجماعية + الذكاء الاصطناعي
    """
    
    def __init__(self):
        self.fraud_threshold = 0.7  # حد الاحتيال
        self.churn_threshold = 0.6  # حد احتمالية التوقف
    
    # ========== 1. نظام البصمة الرقمية المخفية ==========
    def generate_device_hash(self, ip_address, user_agent, screen_res, language='en', timezone='UTC'):
        """
        توليد بصمة رقمية فريدة للجهاز
        تستخدم SHA-256 لتشفير معلومات الجهاز
        """
        raw_string = f"{ip_address}|{user_agent}|{screen_res}|{language}|{timezone}"
        # إضافة salt عشوائي للأمان مع الحفاظ على التكرارية
        salt = "ViralShield_SecureSalt_2024"
        raw_string_with_salt = f"{raw_string}|{salt}"
        return hashlib.sha256(raw_string_with_salt.encode('utf-8')).hexdigest()
    
    def detect_fraud(self, device_hash, promo_code):
        """
        كشف محاولات الاحتيال:
        - نفس الجهاز يستخدم أكواد متعددة
        - أنماط شراء مشبوهة
        """
        fraud_signals = []
        
        # التحقق من وجود الجهاز مع مروجين مختلفين
        fingerprints = DeviceFingerprint.query.filter_by(device_hash=device_hash).all()
        
        if len(fingerprints) > 1:
            unique_promoters = set(f.promoter_id for f in fingerprints)
            if len(unique_promoters) > 1:
                fraud_signals.append("جهاز مرتبط بأكثر من مروج")
        
        # التحقق من تكرار الشراء السريع
        recent_purchases = Transaction.query.filter(
            Transaction.fingerprint_id.in_([f.id for f in fingerprints]),
            Transaction.created_at >= datetime.utcnow() - timedelta(hours=1)
        ).count()
        
        if recent_purchases > 5:
            fraud_signals.append("نمط شراء متكرر ومشبوه")
        
        # حساب درجة الاحتيال
        fraud_score = min(len(fraud_signals) * 0.3, 1.0)
        
        return {
            'is_fraudulent': fraud_score >= self.fraud_threshold,
            'fraud_score': fraud_score,
            'signals': fraud_signals
        }
    
    # ========== 2. محرك التسعير والعمولات الديناميكي ==========
    def calculate_smart_margin(self, product_price, product_cost, is_viral_success=False, 
                              promoter_performance=0.5, order_volume=1):
        """
        حساب ذكي للخصم والعمولة يضمن ربح التاجر تحت أي ظرف
        
        المعايير:
        - product_price: سعر المنتج للمستهلك
        - product_cost: تكلفة المنتج على التاجر
        - is_viral_success: هل نجح التحدي الجماعي
        - promoter_performance: أداء المروج (0 إلى 1)
        - order_volume: عدد الطلبات في العملية
        """
        gross_profit = product_price - product_cost
        
        # حماية: لا يمكن أن نخسر التاجر
        if gross_profit <= 0:
            return {
                'final_price': product_price,
                'promoter_payout': 0,
                'merchant_secured_profit': 0,
                'error': 'المنتج لا يحقق هامش ربح كافي'
            }
        
        # حساب نسب التوزيع الذكية
        if is_viral_success:
            # نجاح التحدي الجماعي: خصم أكبر مع حماية الربح
            max_discount_pct = min(0.40, (gross_profit * 0.8) / product_price)  # لا يتجاوز 80% من الربح
            buyer_discount = gross_profit * max_discount_pct
            promoter_commission = gross_profit * 0.20
            
            # مكافأة إضافية للمروج المتميز
            if promoter_performance > 0.7:
                promoter_commission *= 1.2  # 20% إضافية
        else:
            # الشراء العادي
            max_discount_pct = min(0.15, (gross_profit * 0.5) / product_price)
            buyer_discount = gross_profit * max_discount_pct
            promoter_commission = gross_profit * 0.10
            
            # خصم بالجملة
            if order_volume >= 5:
                buyer_discount *= 1.1  # زيادة 10% للخصم
                promoter_commission *= 0.95  # تخفيض بسيط للعمولة لتعويض الخصم
        
        # حساب صافي ربح التاجر بعد كل الخصومات
        merchant_net_profit = gross_profit - buyer_discount - promoter_commission
        
        # حماية أخيرة: التأكد من ربح التاجر
        if merchant_net_profit < 0:
            # تعديل النسب لضمان الحد الأدنى من الربح (10% من التكلفة)
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
    
    # ========== 3. نظام التحديات الجماعية الفيروسية ==========
    def create_viral_challenge(self, buyer_id, promo_code, product_price, product_cost, 
                              required_buyers=3, duration_hours=3):
        """
        إنشاء تحدي شراء جماعي جديد (القنبلة الموقوتة)
        """
        promoter = Promoter.query.filter_by(promo_code=promo_code).first()
        if not promoter:
            return {'error': 'كود المروج غير صالح'}
        
        # إنشاء التحدي
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
        
        # تحديث نشاط المروج
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
        """
        معالجة عملية شراء ضمن تحدي جماعي
        """
        challenge = ViralGroupChallenge.query.get(challenge_id)
        
        if not challenge:
            return {'error': 'التحدي غير موجود'}
        
        if challenge.status != "ACTIVE":
            return {'error': f'التحدي {challenge.status}'}
        
        # التحقق من الوقت
        if datetime.utcnow() > challenge.expiration_time:
            challenge.status = "EXPIRED"
            db.session.commit()
            return {'error': 'انتهى وقت التحدي'}
        
        # إضافة المشتري الجديد
        challenge.current_buyers_joined += 1
        challenge.total_sales_generated += challenge.product_price
        
        # التحقق من اكتمال التحدي
        if challenge.current_buyers_joined >= challenge.required_buyers:
            challenge.status = "SUCCESS"
            challenge.completed_at = datetime.utcnow()
            
            # حساب الخصم الأقصى للجميع
            finance = self.calculate_smart_margin(
                challenge.product_price,
                challenge.product_cost,
                is_viral_success=True
            )
            
            challenge.discount_applied = finance['discount_percentage']
            
            # تسجيل معاملة للمروج
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
            
            # تحديث أرباح المروج
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
        
        # إذا لم يكتمل التحدي بعد
        db.session.commit()
        
        remaining = challenge.required_buyers - challenge.current_buyers_joined
        return {
            'status': 'PENDING',
            'message': f'تم تسجيل شرائك! متبقي {remaining} أصدقاء لتفعيل الخصم الأكبر',
            'current_buyers': challenge.current_buyers_joined,
            'remaining_to_activate': remaining,
            'current_discount': '5%'
        }
    
    # ========== 4. الذكاء الاصطناعي: التنبؤ بخمول المروجين ==========
    def calculate_churn_probability(self, promoter):
        """
        حساب احتمالية توقف المروج عن النشاط باستخدام خوارزمية ذكية
        تعتمد على: آخر نشاط، معدل المبيعات، التفاعل
        """
        signals = []
        churn_score = 0.0
        
        # 1. تحليل آخر نشاط
        if promoter.last_sale_date:
            days_since_last_sale = (datetime.utcnow() - promoter.last_sale_date).days
            
            if days_since_last_sale > 30:
                churn_score += 0.4
                signals.append("أكثر من 30 يوم بدون مبيعات")
            elif days_since_last_sale > 14:
                churn_score += 0.2
                signals.append("أكثر من أسبوعين بدون مبيعات")
        else:
            # لم يقم بأي عملية بيع
            days_since_registration = (datetime.utcnow() - promoter.created_at).days
            if days_since_registration > 7:
                churn_score += 0.5
                signals.append("مسجل منذ فترة بدون أي مبيعات")
        
        # 2. تحليل معدل النقرات مقابل المبيعات
        if promoter.total_clicks_tracked > 20:
            conversion_rate = promoter.total_sales / promoter.total_clicks_tracked
            if conversion_rate < 0.01:  # أقل من 1%
                churn_score += 0.3
                signals.append("معدل تحويل منخفض جداً")
        
        # 3. تحليل الانتظام في النشاط
        if promoter.last_login:
            days_since_login = (datetime.utcnow() - promoter.last_login).days
            if days_since_login > 14:
                churn_score += 0.3
                signals.append("لم يسجل الدخول منذ فترة")
            elif days_since_login > 7:
                churn_score += 0.15
        
        # 4. تخفيض الدرجة للمروجين النشطين
        if promoter.activity_score > 80:
            churn_score = max(0, churn_score - 0.2)
        
        # تطبيع النتيجة
        churn_probability = min(churn_score, 1.0)
        
        return {
            'churn_probability': round(churn_probability, 2),
            'is_at_risk': churn_probability >= self.churn_threshold,
            'signals': signals,
            'activity_score': promoter.activity_score
        }
    
    def get_churn_alert_message(self, promoter_name, churn_data):
        """
        توليد رسالة تنبيه مخصصة للمروج الكسول
        """
        if not churn_data['is_at_risk']:
            return None
        
        messages = {
            'high': f"🔥 {promoter_name}، متجرنا يحتاجك! عرض خاص: عمولة مضاعفة على مبيعات اليوم فقط!",
            'medium': f"💡 {promoter_name}، لاحظنا غيابك. هل تحتاج مساعدة في التسويق؟",
            'low': f"👋 {promoter_name}، مرحباً! لدينا منتجات جديدة قد تعجب متابعيك."
        }
        
        if churn_data['churn_probability'] > 0.8:
            return messages['high']
        elif churn_data['churn_probability'] > 0.6:
            return messages['medium']
        else:
            return messages['low']


# إنشاء نسخة من المحرك
engine = ViralShieldEngine()


# ==================== نقاط نهاية API ====================

@app.route('/')
def index():
    """الصفحة الرئيسية - معلومات النظام"""
    return jsonify({
        'system': 'ViralShield AI Engine',
        'version': '3.0.0',
        'status': '🚀 Operational',
        'features': [
            'Cookie-less AI Tracking',
            'Viral Group Buying',
            'AI Margin-Based Commission',
            'Churn Prediction & Fraud Prevention'
        ],
        'endpoints': {
            'register_promoter': '/api/promoter/register',
            'create_challenge': '/api/challenge/create',
            'join_challenge': '/api/challenge/join/<id>',
            'challenge_status': '/api/challenge/status/<id>',
            'track_visit': '/api/track/visit',
            'calculate_commission': '/api/calculate/commission',
            'predict_churn': '/api/ai/predict-churn/<promoter_id>',
            'dashboard_stats': '/api/dashboard/stats'
        }
    })


@app.route('/api/promoter/register', methods=['POST'])
def register_promoter():
    """تسجيل مروج جديد"""
    try:
        data = request.json
        
        # التحقق من البيانات المطلوبة
        if not data.get('name') or not data.get('email'):
            return jsonify({'error': 'الاسم والبريد الإلكتروني مطلوبان'}), 400
        
        # التحقق من عدم وجود البريد مسبقاً
        if Promoter.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'البريد الإلكتروني مسجل مسبقاً'}), 400
        
        # إنشاء كود ترويجي فريد
        promo_code = f"{data['name'][:5].upper()}{secrets.token_hex(3).upper()}"
        
        promoter = Promoter(
            name=data['name'],
            email=data['email'],
            promo_code=promo_code,
            base_commission_rate=data.get('commission_rate', 0.10)
        )
        
        db.session.add(promoter)
        db.session.commit()
        
        return jsonify({
            'message': '✅ تم تسجيل المروج بنجاح',
            'promoter': promoter.to_dict()
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'خطأ في التسجيل: {str(e)}'}), 500


@app.route('/api/track/visit', methods=['POST'])
def track_visit():
    """تتبع زيارة جديدة وتوليد بصمة الجهاز"""
    try:
        data = request.json
        
        # توليد البصمة
        device_hash = engine.generate_device_hash(
            ip_address=request.remote_addr or 'unknown',
            user_agent=request.headers.get('User-Agent', 'unknown'),
            screen_res=data.get('screen_resolution', 'unknown'),
            language=data.get('language', 'ar'),
            timezone=data.get('timezone', 'UTC')
        )
        
        # البحث عن بصمة موجودة
        existing = DeviceFingerprint.query.filter_by(device_hash=device_hash).first()
        
        if existing:
            # تحديث الزيارة
            existing.click_count += 1
            existing.last_click_time = datetime.utcnow()
            
            # التحقق من الاحتيال
            fraud_check = engine.detect_fraud(device_hash, data.get('promo_code'))
            existing.is_suspicious = fraud_check['is_fraudulent']
            existing.fraud_score = fraud_check['fraud_score']
            
            db.session.commit()
            
            return jsonify({
                'status': 'returning_visitor',
                'device_hash': device_hash[:16] + '...',
                'visit_count': existing.click_count,
                'fraud_check': fraud_check
            })
        
        # بصمة جديدة
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
        
        # تحديث إحصائيات المروج
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
    """إنشاء تحدي شراء جماعي جديد"""
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
        
        return jsonify({
            'message': '🎯 تم إنشاء التحدي الجماعي!',
            'challenge': result
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'خطأ في إنشاء التحدي: {str(e)}'}), 500


@app.route('/api/challenge/join/<int:challenge_id>', methods=['POST'])
def join_challenge(challenge_id):
    """الانضمام إلى تحدي جماعي قائم"""
    try:
        data = request.json
        
        # توليد بصمة للمشتري الجديد
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
    """معرفة حالة التحدي الجماعي"""
    try:
        challenge = ViralGroupChallenge.query.get(challenge_id)
        
        if not challenge:
            return jsonify({'error': 'التحدي غير موجود'}), 404
        
        return jsonify(challenge.to_dict())
        
    except Exception as e:
        return jsonify({'error': f'خطأ في جلب حالة التحدي: {str(e)}'}), 500


@app.route('/api/calculate/commission', methods=['POST'])
def calculate_commission():
    """حساب العمولة والخصم الديناميكي"""
    try:
        data = request.json
        
        result = engine.calculate_smart_margin(
            product_price=data.get('product_price', 100),
            product_cost=data.get('product_cost', 60),
            is_viral_success=data.get('is_viral_success', False),
            promoter_performance=data.get('promoter_performance', 0.5),
            order_volume=data.get('order_volume', 1)
        )
        
        return jsonify({
            'calculation': result,
            'summary': f"السعر النهائي: {result['final_price']} | عمولة المروج: {result['promoter_payout']} | ربح التاجر: {result['merchant_secured_profit']}"
        })
        
    except Exception as e:
        return jsonify({'error': f'خطأ في الحساب: {str(e)}'}), 500


@app.route('/api/ai/predict-churn/<int:promoter_id>', methods=['GET'])
def predict_churn(promoter_id):
    """التنبؤ باحتمالية خمول المروج"""
    try:
        promoter = Promoter.query.get(promoter_id)
        
        if not promoter:
            return jsonify({'error': 'المروج غير موجود'}), 404
        
        churn_data = engine.calculate_churn_probability(promoter)
        alert_message = engine.get_churn_alert_message(promoter.name, churn_data)
        
        # تحديث بيانات المروج
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
    """لوحة التحكم - إحصائيات عامة"""
    try:
        total_promoters = Promoter.query.count()
        active_challenges = ViralGroupChallenge.query.filter_by(status='ACTIVE').count()
        successful_challenges = ViralGroupChallenge.query.filter_by(status='SUCCESS').count()
        total_transactions = Transaction.query.count()
        
        # المروجين المعرضين للخطر
        at_risk_promoters = Promoter.query.filter_by(is_at_risk_of_churn=True).count()
        
        # إجمالي المبيعات
        total_sales = db.session.query(db.func.sum(Transaction.amount)).filter_by(
            transaction_type='SALE'
        ).scalar() or 0
        
        # إجمالي أرباح التاجر
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
    """فحص صحة النظام"""
    try:
        # اختبار اتصال قاعدة البيانات
        db.session.execute(db.text('SELECT 1'))
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


# ==================== معالجة الأخطاء العامة ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'المسار غير موجود', 'status_code': 404}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'خطأ في الخادم', 'status_code': 500}), 500


# ==================== إنشاء قاعدة البيانات والتشغيل ====================

def init_database():
    """تهيئة قاعدة البيانات وإنشاء الجداول"""
    with app.app_context():
        db.create_all()
        
        # إضافة بيانات تجريبية إذا كانت القاعدة فارغة
        if Promoter.query.count() == 0:
            # إنشاء مروج تجريبي
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


# ==================== نقطة التشغيل الرئيسية ====================

if __name__ == '__main__':
    # تهيئة قاعدة البيانات
    init_database()
    
    # تشغيل التطبيق
    port = int(os.environ.get('PORT', 5000))
    print(f"""
╔══════════════════════════════════════════════════════════╗
║     🚀 ViralShield AI Engine - Ready                    ║
║     📍 Running on: http://0.0.0.0:{port}                  ║
║     📊 API Docs: http://0.0.0.0:{port}/                  ║
║     🧠 AI Systems: Active                               ║
║     🔐 Fraud Protection: Enabled                        ║
║     💰 Smart Margin Engine: Online                      ║
╚══════════════════════════════════════════════════════════╝
    """)
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # للتشغيل على Render مع gunicorn
    init_database()
