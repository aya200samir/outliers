# -*- coding: utf-8 -*-
"""
واجهة المستخدم لتحليل الأحكام القضائية وكشف الشذوذ
مستوحاة من تصميم Mizan AI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
import os
from io import StringIO
warnings.filterwarnings('ignore')

# ==================== مكتبات التعلم الآلي ====================
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)

# محاولة استيراد SHAP للتفسير
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ==================== إعدادات الصفحة ====================
st.set_page_config(
    page_title="عدالة⚖️ - نظام تحليل الأحكام القضائية",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS مخصص - مستوحى من Mizan AI ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');
    * { font-family: 'Cairo', sans-serif; }
    
    /* Header */
    .header {
        background: linear-gradient(135deg, #0a3147, #1a4b6d);
        color: white;
        padding: 2rem;
        border-radius: 0 0 30px 30px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,20,40,0.3);
    }
    .header h1 { 
        font-size: 3rem; 
        font-weight: 900; 
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .header p { 
        font-size: 1.2rem; 
        opacity: 0.9;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 20px;
        padding: 1.8rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #eaeef2;
        transition: all 0.3s ease;
    }
    .card:hover {
        box-shadow: 0 15px 35px rgba(26,75,109,0.1);
        transform: translateY(-3px);
    }
    .card-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a4b6d;
        margin-bottom: 1.2rem;
        border-bottom: 2px solid #eaeef2;
        padding-bottom: 0.7rem;
    }
    
    /* Metric Cards */
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fbff, #ffffff);
        border-radius: 18px;
        padding: 1.2rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.03);
        text-align: center;
        flex: 1 1 180px;
        border: 1px solid #dde5ed;
        transition: all 0.3s;
    }
    .metric-card:hover {
        border-color: #1a4b6d;
        box-shadow: 0 8px 20px rgba(26,75,109,0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #0a3147;
        line-height: 1.2;
    }
    .metric-label {
        color: #5f6b7a;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Badges */
    .badge-normal {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 700;
        display: inline-block;
        border-right: 4px solid #28a745;
    }
    .badge-anomaly {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 700;
        display: inline-block;
        border-right: 4px solid #dc3545;
    }
    .badge-warning {
        background: linear-gradient(135deg, #fff3cd, #ffeeba);
        color: #856404;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 700;
        display: inline-block;
        border-right: 4px solid #ffc107;
    }
    
    /* Alert Boxes */
    .alert-success {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-right: 8px solid #28a745;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1.2rem 0;
        color: #155724;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(40,167,69,0.1);
    }
    .alert-danger {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-right: 8px solid #dc3545;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1.2rem 0;
        color: #721c24;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(220,53,69,0.1);
    }
    .alert-warning {
        background: linear-gradient(135deg, #fff3cd, #ffeeba);
        border-right: 8px solid #ffc107;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1.2rem 0;
        color: #856404;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(255,193,7,0.1);
    }
    .alert-info {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        border-right: 8px solid #17a2b8;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1.2rem 0;
        color: #0c5460;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(23,162,184,0.1);
    }
    
    /* Feature Importance */
    .feature-bar {
        height: 8px;
        background: linear-gradient(90deg, #1a4b6d, #4a90e2);
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1a4b6d, #2c5f8a);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        width: 100%;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(26,75,109,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2c5f8a, #1a4b6d);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(26,75,109,0.4);
    }
    
    /* Sidebar */
    .css-1d391kg, .css-12ttj6m {
        background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
    }
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 900;
        color: #1a4b6d;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #eaeef2;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #0a3147, #1a4b6d);
        color: white;
        padding: 2rem;
        border-radius: 30px 30px 0 0;
        margin-top: 4rem;
        text-align: center;
        box-shadow: 0 -10px 30px rgba(0,0,0,0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 12px 12px 0 0;
        padding: 0.8rem 1.8rem;
        font-weight: 700;
        color: #5f6b7a;
        border: 1px solid #eaeef2;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a4b6d, #2c5f8a);
        color: white !important;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #1a4b6d, transparent);
        margin: 2rem 0;
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #1a4b6d;
        cursor: help;
    }
</style>
""", unsafe_allow_html=True)

# ==================== تهيئة حالة الجلسة ====================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_pack' not in st.session_state:
    st.session_state.model_pack = None

# ==================== توليد بيانات تجريبية (محاكاة) ====================
def generate_sample_crime_data(n_samples=5000):
    """
    توليد بيانات جرائم تجريبية للمحاكاة
    """
    np.random.seed(42)
    
    crime_types = ['سرقة', 'سطو', 'نشل', 'احتيال', 'اختلاس']
    descriptions = {
        'سرقة': ['سرقة بالإكراه', 'سرقة سيارة', 'سرقة منزل', 'سرقة محل'],
        'سطو': ['سطو مسلح', 'سطو بنك', 'سطو منزل'],
        'نشل': ['نشل في المواصلات', 'نشل في السوق', 'نشل محفظة'],
        'احتيال': ['احتيال مالي', 'تزوير', 'انتحال شخصية'],
        'اختلاس': ['اختلاس أموال عامة', 'اختلاس من شركة']
    }
    locations = ['شارع', 'منزل', 'بنك', 'متجر', 'مواصلات عامة', 'مول تجاري']
    districts = ['المنطقة الشمالية', 'المنطقة الجنوبية', 'المنطقة الشرقية', 'المنطقة الغربية', 'الوسطى']
    judges = ['القاضي أحمد', 'القاضي محمد', 'القاضي فاطمة', 'القاضي سارة', 'القاضي خالد']
    
    data = []
    for i in range(n_samples):
        crime_type = np.random.choice(crime_types)
        desc = np.random.choice(descriptions[crime_type])
        location = np.random.choice(locations)
        district = np.random.choice(districts)
        judge = np.random.choice(judges)
        domestic = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # الأدلة (رقمية)
        evidence_strength = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.15, 0.3, 0.25, 0.2])
        
        # القرار (قبض/لم يقبض) مع بعض التحيزات
        if crime_type in ['سطو', 'احتيال'] and evidence_strength >= 4:
            arrest = 1
        elif crime_type == 'سرقة' and domestic == 1:
            arrest = np.random.choice([0, 1], p=[0.4, 0.6])
        elif location == 'بنك' and evidence_strength >= 3:
            arrest = 1
        elif judge == 'القاضي أحمد' and crime_type == 'نشل':
            arrest = np.random.choice([0, 1], p=[0.8, 0.2])  # متساهل مع النشل
        else:
            arrest = np.random.choice([0, 1], p=[0.45, 0.55])
        
        # شذوذ متعمد (حالات مشبوهة)
        if i % 97 == 0:  # كل 97 حالة نضيف شذوذ
            if evidence_strength >= 4 and crime_type == 'سطو':
                arrest = 0  # سطو قوي بدون قبض
            elif judge == 'القاضي خالد' and crime_type == 'اختلاس':
                arrest = 0  # القاضي خالد متساهل مع الاختلاس
        
        data.append({
            'نوع_الجريمة': crime_type,
            'الوصف': desc,
            'المكان': location,
            'المنطقة': district,
            'محلي': domestic,
            'القاضي': judge,
            'قوة_الأدلة': evidence_strength,
            'تم_القبض': arrest
        })
    
    return pd.DataFrame(data)

# ==================== دالة MCAS (محاكاة) ====================
def mcas_score(y_true, y_pred, lambda1=1, lambda2=1):
    """
    محاكاة لمقياس MCAS
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    css_plus = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    css_minus = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0
    
    cfs = 0.5 * (
        (fp / (tp + tn + fp) if (tp + tn + fp) > 0 else 0) +
        (fn / (tp + tn + fn) if (tp + tn + fn) > 0 else 0)
    )
    
    mcas = (lambda1 * (css_plus - cfs) + lambda2 * (css_minus - cfs)) / (lambda1 + lambda2)
    return max(0, min(1, mcas))

# ==================== تدريب النموذج ====================
def train_model(df):
    """
    تدريب نموذج RandomForest مع تجهيز البيانات
    """
    # اختيار الأعمدة المهمة
    feature_cols = ['قوة_الأدلة', 'محلي']
    categorical_cols = ['نوع_الجريمة', 'الوصف', 'المكان', 'المنطقة', 'القاضي']
    
    df_encoded = df.copy()
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col + '_code'] = le.fit_transform(df_encoded[col])
        encoders[col] = le
        feature_cols.append(col + '_code')
    
    X = df_encoded[feature_cols]
    y = df_encoded['تم_القبض']
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # تدريب النموذج
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # التنبؤ
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # حساب المقاييس
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'mcas': mcas_score(y_test, y_pred)
    }
    
    return {
        'model': model,
        'encoders': encoders,
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols,
        'metrics': metrics,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'df_encoded': df_encoded
    }

# ==================== كشف الشذوذ ====================
def detect_anomalies(model_pack, df, threshold=0.8):
    """
    اكتشاف الحالات الشاذة (Outliers)
    """
    model = model_pack['model']
    encoders = model_pack['encoders']
    feature_cols = model_pack['feature_cols']
    categorical_cols = model_pack['categorical_cols']
    
    df_encoded = df.copy()
    for col in categorical_cols:
        if col in encoders:
            df_encoded[col + '_code'] = encoders[col].transform(df_encoded[col])
    
    X_all = df_encoded[feature_cols]
    probabilities = model.predict_proba(X_all)[:, 1]
    
    # تحديد الشذوذ: احتمال قبض عالٍ ولكن لم يتم القبض فعلياً
    anomalies = df[(probabilities >= threshold) & (df['تم_القبض'] == 0)]
    
    # إضافة احتمالية الشذوذ للنتائج
    anomaly_indices = anomalies.index
    anomaly_probs = probabilities[anomaly_indices]
    anomalies = anomalies.copy()
    anomalies['احتمالية_الشذوذ'] = anomaly_probs
    
    return anomalies, probabilities

# ==================== تحليل أهمية الميزات ====================
def get_feature_importance(model_pack, feature_names_ar):
    """
    استخراج أهمية الميزات
    """
    model = model_pack['model']
    importances = model.feature_importances_
    feature_names = model_pack['feature_cols']
    
    # ترجمة أسماء الميزات
    name_mapping = {
        'قوة_الأدلة': 'قوة الأدلة',
        'محلي': 'محلي/دولي',
        'نوع_الجريمة_code': 'نوع الجريمة',
        'الوصف_code': 'وصف الجريمة',
        'المكان_code': 'المكان',
        'المنطقة_code': 'المنطقة',
        'القاضي_code': 'القاضي'
    }
    
    feature_names_ar = [name_mapping.get(f, f) for f in feature_names]
    
    # ترتيب حسب الأهمية
    indices = np.argsort(importances)[::-1]
    
    result = []
    for i in indices:
        result.append({
            'الميزة': feature_names_ar[i],
            'الأهمية': importances[i]
        })
    
    return result

# ==================== الصفحة الرئيسية ====================
def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>⚖️ عدالة - نظام تحليل الأحكام القضائية</h1>
        <p>كشف الأنماط الطبيعية وتحليل الحالات الشاذة باستخدام الذكاء الاصطناعي</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🔍 لوحة التحكم</div>', unsafe_allow_html=True)
        
        st.markdown("### 📂 البيانات")
        data_source = st.radio(
            "مصدر البيانات",
            ["📊 بيانات تجريبية", "📁 رفع ملف CSV"],
            index=0
        )
        
        if data_source == "📁 رفع ملف CSV":
            uploaded_file = st.file_uploader("اختر ملف CSV", type=['csv'])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.success(f"✅ تم تحميل {len(df)} سجل")
                except Exception as e:
                    st.error(f"خطأ في قراءة الملف: {e}")
            else:
                st.info("يرجى رفع ملف CSV")
        else:
            if st.button("🔄 توليد بيانات تجريبية"):
                with st.spinner("جاري توليد البيانات..."):
                    df = generate_sample_crime_data(5000)
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                st.success("✅ تم توليد 5000 حالة بنجاح")
        
        st.markdown("---")
        
        st.markdown("### ⚙️ إعدادات النموذج")
        threshold = st.slider(
            "عتبة كشف الشذوذ",
            min_value=0.5,
            max_value=0.95,
            value=0.8,
            step=0.05,
            help="كلما زادت القيمة، قل عدد الحالات المشبوهة (أكثر دقة)"
        )
        
        n_estimators = st.slider(
            "عدد الأشجار",
            min_value=50,
            max_value=300,
            value=150,
            step=50
        )
        
        if st.button("🧠 تدريب النموذج", type="primary"):
            if st.session_state.data_loaded and st.session_state.df is not None:
                with st.spinner("جاري تدريب النموذج..."):
                    # تحديث معلمات النموذج
                    model_pack = train_model(st.session_state.df)
                    st.session_state.model_pack = model_pack
                    st.session_state.model_trained = True
                st.success("✅ تم تدريب النموذج بنجاح")
            else:
                st.warning("⚠️ يرجى تحميل البيانات أولاً")
        
        st.markdown("---")
        st.markdown("### ℹ️ عن النظام")
        st.markdown("""
        **الإصدار:** 1.0.0  
        **التقنيات:**  
        - Random Forest  
        - تحليل الشذوذ  
        - تفسير النتائج  
        
        **الفلسفة:**  
        - كشف الأنماط غير الطبيعية  
        - تحليل أسباب الشذوذ  
        - دعم القرار للخبراء
        """)
    
    # المحتوى الرئيسي
    if not st.session_state.data_loaded:
        st.info("👈 يرجى تحميل البيانات أو توليدها من القائمة الجانبية")
        
        # عرض شرح النظام
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="card">
                <div class="card-title">📊 تحليل البيانات</div>
                <p>استكشاف البيانات وفهم الأنماط المخفية في الأحكام القضائية.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="card">
                <div class="card-title">🧠 كشف الشذوذ</div>
                <p>اكتشاف الحالات التي تخرج عن النمط الطبيعي باستخدام الذكاء الاصطناعي.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="card">
                <div class="card-title">🔍 تحليل الأسباب</div>
                <p>فهم العوامل التي أدت إلى الشذوذ (القاضي؟ الأدلة؟ المكان؟).</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # عرض البيانات
    df = st.session_state.df
    
    st.markdown("## 📊 نظرة عامة على البيانات")
    
    # إحصائيات سريعة
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">إجمالي الحالات</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        arrest_rate = df['تم_القبض'].mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{arrest_rate:.1f}%</div>
            <div class="metric-label">نسبة القبض</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['نوع_الجريمة'].nunique()}</div>
            <div class="metric-label">أنواع الجرائم</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['القاضي'].nunique()}</div>
            <div class="metric-label">عدد القضاة</div>
        </div>
        """, unsafe_allow_html=True)
    
    # تبويبات
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 استكشاف البيانات", 
        "🧠 النموذج والتقييم", 
        "🚨 كشف الشذوذ",
        "📈 تحليل الأسباب",
        "⚖️ نظام القرار"
    ])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📋 عينة من البيانات</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 توزيع الجرائم حسب النوع</div>', unsafe_allow_html=True)
            crime_dist = df['نوع_الجريمة'].value_counts().reset_index()
            crime_dist.columns = ['نوع الجريمة', 'العدد']
            fig = px.pie(crime_dist, values='العدد', names='نوع الجريمة', 
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 معدلات القبض حسب نوع الجريمة</div>', unsafe_allow_html=True)
            arrest_by_crime = df.groupby('نوع_الجريمة')['تم_القبض'].mean().reset_index()
            arrest_by_crime.columns = ['نوع الجريمة', 'معدل القبض']
            fig = px.bar(arrest_by_crime, x='نوع الجريمة', y='معدل القبض',
                         color='معدل القبض', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # النموذج والتقييم
    with tab2:
        if not st.session_state.model_trained:
            st.warning("⚠️ يرجى تدريب النموذج أولاً من القائمة الجانبية")
        else:
            model_pack = st.session_state.model_pack
            metrics = model_pack['metrics']
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 مقاييس أداء النموذج</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['accuracy']*100:.1f}%</div>
                    <div class="metric-label">الدقة</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['precision']*100:.1f}%</div>
                    <div class="metric-label">الدقة (Precision)</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['recall']*100:.1f}%</div>
                    <div class="metric-label">الاستدعاء</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['f1']*100:.1f}%</div>
                    <div class="metric-label">F1 Score</div>
                </div>
                """, unsafe_allow_html=True)
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['mcas']*100:.1f}%</div>
                    <div class="metric-label">MCAS</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">📊 مصفوفة الارتباك</div>', unsafe_allow_html=True)
                cm = confusion_matrix(model_pack['y_test'], model_pack['y_pred'])
                fig = px.imshow(cm, text_auto=True, 
                                x=['لم يقبض', 'قبض'], y=['لم يقبض', 'قبض'],
                                color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">📈 منحنى ROC</div>', unsafe_allow_html=True)
                fpr, tpr, _ = roc_curve(model_pack['y_test'], model_pack['y_proba'])
                roc_auc = auc(fpr, tpr)
                
                fig = px.area(x=fpr, y=tpr, title=f'AUC = {roc_auc:.3f}',
                              labels={'x': 'معدل الإيجابيات الكاذبة', 'y': 'معدل الإيجابيات الحقيقية'})
                fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # كشف الشذوذ
    with tab3:
        if not st.session_state.model_trained:
            st.warning("⚠️ يرجى تدريب النموذج أولاً")
        else:
            model_pack = st.session_state.model_pack
            
            with st.spinner("جاري كشف الحالات الشاذة..."):
                anomalies, probs = detect_anomalies(model_pack, df, threshold)
            
            st.markdown(f"""
            <div class="card">
                <div class="card-title">🚨 نتائج كشف الشذوذ</div>
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-value">{len(anomalies):,}</div>
                        <div class="metric-label">حالة مشبوهة</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(anomalies)/len(df)*100:.2f}%</div>
                        <div class="metric-label">نسبة الشذوذ</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            if len(anomalies) > 0:
                st.markdown(f"""
                <div class="alert-warning">
                    ⚠️ تم اكتشاف {len(anomalies)} حالة لا تتبع النمط الطبيعي.
                    هذه الحالات تحتاج إلى مراجعة دقيقة من قبل الخبراء.
                </div>
                """, unsafe_allow_html=True)
                
                # عرض الحالات الشاذة
                st.markdown('<div class="card-title">📋 الحالات المشبوهة</div>', unsafe_allow_html=True)
                display_cols = ['نوع_الجريمة', 'الوصف', 'المكان', 'المنطقة', 'القاضي', 
                               'قوة_الأدلة', 'احتمالية_الشذوذ']
                st.dataframe(anomalies[display_cols].head(20), use_container_width=True)
                
                # تحليل الشذوذ حسب القاضي
                st.markdown('<div class="card-title">👨‍⚖️ تحليل الشذوذ حسب القاضي</div>', unsafe_allow_html=True)
                judge_anomalies = anomalies.groupby('القاضي').size().reset_index(name='عدد_الحالات')
                fig = px.bar(judge_anomalies, x='القاضي', y='عدد_الحالات',
                             color='عدد_الحالات', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("""
                <div class="alert-success">
                    ✅ لم يتم العثور على حالات شاذة بالمعايير الحالية.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # تحليل الأسباب
    with tab4:
        if not st.session_state.model_trained:
            st.warning("⚠️ يرجى تدريب النموذج أولاً")
        else:
            model_pack = st.session_state.model_pack
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🔍 أهم العوامل المؤثرة في القرار</div>', unsafe_allow_html=True)
            
            feature_importance = get_feature_importance(model_pack, [])
            
            for f in feature_importance:
                st.markdown(f"""
                <div style="margin: 1rem 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span><strong>{f['الميزة']}</strong></span>
                        <span>{f['الأهمية']*100:.1f}%</span>
                    </div>
                    <div class="feature-bar" style="width: {f['الأهمية']*100}%;"></div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # تحليل منطقي
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🧠 تحليل منطقي</div>', unsafe_allow_html=True)
            
            top_feature = feature_importance[0]['الميزة']
            st.markdown(f"""
            <div class="alert-info">
                <strong>🔎 الميزة الأكثر تأثيراً هي "{top_feature}"</strong><br><br>
                هذا يعني أن النظام يعتبر أن هذا العامل هو الأهم في تحديد ما إذا كانت القضية طبيعية أم لا.
                عند وجود حالات شاذة تتعلق بهذه الميزة (مثل نوع جريمة معين أو قاضٍ معين)،
                فإن ذلك يستدعي تدقيقاً إضافياً.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # نظام القرار
    with tab5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">⚖️ نظام القرار الهجين</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0f7ff, #ffffff); padding: 1.5rem; border-radius: 15px;">
            <h4>آلية العمل:</h4>
            <ul>
                <li><span class="badge-normal">✅ منطقة آمنة (ثقة ≥ 80%)</span> - قرار آلي مع تفسير</li>
                <li><span class="badge-anomaly">❌ منطقة شاذة (ثقة ≤ 20%)</span> - رفض آلي مع تفسير</li>
                <li><span class="badge-warning">⚠️ منطقة رمادية</span> - تحويل للمراجعة البشرية</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<br>', unsafe_allow_html=True)
        
        if st.session_state.model_trained:
            model_pack = st.session_state.model_pack
            
            col1, col2 = st.columns(2)
            with col1:
                crime_type = st.selectbox("نوع الجريمة", df['نوع_الجريمة'].unique())
                location = st.selectbox("المكان", df['المكان'].unique())
                district = st.selectbox("المنطقة", df['المنطقة'].unique())
                evidence = st.slider("قوة الأدلة (1-5)", 1, 5, 3)
            
            with col2:
                judge = st.selectbox("القاضي", df['القاضي'].unique())
                domestic = st.checkbox("جريمة محلية (Domestic)")
            
            if st.button("🔮 تحليل القضية", use_container_width=True):
                # تجهيز بيانات الإدخال
                input_data = {
                    'قوة_الأدلة': evidence,
                    'محلي': 1 if domestic else 0,
                    'نوع_الجريمة': crime_type,
                    'الوصف': df[df['نوع_الجريمة'] == crime_type]['الوصف'].iloc[0],
                    'المكان': location,
                    'المنطقة': district,
                    'القاضي': judge
                }
                
                # تحويل البيانات
                input_df = pd.DataFrame([input_data])
                for col in model_pack['categorical_cols']:
                    if col in model_pack['encoders']:
                        input_df[col + '_code'] = model_pack['encoders'][col].transform(input_df[col])
                
                # التنبؤ
                feature_cols = model_pack['feature_cols']
                X_input = input_df[feature_cols]
                prob = model_pack['model'].predict_proba(X_input)[0][1]
                
                # عرض النتيجة
                st.markdown('<hr>', unsafe_allow_html=True)
                
                if prob >= 0.8:
                    st.markdown(f"""
                    <div class="alert-success">
                        <h4>✅ قرار آلي: قبض متوقع</h4>
                        <p>نسبة الثقة: {prob*100:.1f}%</p>
                        <p>القضية واضحة وتتبع النمط الطبيعي.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif prob <= 0.2:
                    st.markdown(f"""
                    <div class="alert-danger">
                        <h4>❌ قرار آلي: لا يتوقع قبض</h4>
                        <p>نسبة الثقة: {(1-prob)*100:.1f}%</p>
                        <p>القضية واضحة وتتبع النمط الطبيعي للرفض.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-warning">
                        <h4>⚠️ يحتاج مراجعة بشرية</h4>
                        <p>نسبة الثقة: {prob*100:.1f}%</p>
                        <p>القضية في المنطقة الرمادية. يرجى عرضها على خبير.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # تقرير للمراجعة
                    with st.expander("📋 تقرير المراجعة", expanded=True):
                        st.markdown("""
                        **نقاط المراجعة:**
                        1. هل الأدلة كافية رغم عدم تطابقها مع النمط؟
                        2. هل هناك ظروف خاصة بالقضية؟
                        3. هل القاضي له سوابق مع هذا النوع؟
                        """)
        else:
            st.info("👈 يرجى تدريب النموذج أولاً من القائمة الجانبية")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # تحليل إضافي للشذوذ
    if st.session_state.model_trained and len(anomalies) > 0:
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown("## 📊 تحليل متقدم للشذوذ")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📍 توزيع الشذوذ حسب المنطقة</div>', unsafe_allow_html=True)
            anomaly_by_district = anomalies['المنطقة'].value_counts().reset_index()
            anomaly_by_district.columns = ['المنطقة', 'عدد الحالات']
            fig = px.pie(anomaly_by_district, values='عدد الحالات', names='المنطقة',
                         color_discrete_sequence=px.colors.sequential.Reds_r)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">👨‍⚖️ أكثر القضاة شذوذاً</div>', unsafe_allow_html=True)
            judge_counts = anomalies['القاضي'].value_counts().head(5).reset_index()
            judge_counts.columns = ['القاضي', 'عدد الحالات']
            fig = px.bar(judge_counts, x='القاضي', y='عدد الحالات',
                         color='عدد الحالات', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>⚖️ نظام عدالة لتحليل الأحكام القضائية | الإصدار 1.0.0</p>
        <p>مبني على تقنيات التعلم الآلي وتحليل الشذوذ</p>
        <p style="opacity:0.7; font-size:0.9rem;">© 2026 - جميع الحقوق محفوظة</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
