# -*- coding: utf-8 -*-
"""
واجهة المستخدم لتحليل جرائم السرقة في الأماكن المأهولة - تشيلي
بيانات CEAD-SPD (2018-2020)
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    page_title="تحليل جرائم السرقة - تشيلي",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS مخصص ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');
    * { font-family: 'Cairo', sans-serif; }
    
    /* Header */
    .header {
        background: linear-gradient(135deg, #8B1E3F, #C41E3A);
        color: white;
        padding: 2rem;
        border-radius: 0 0 30px 30px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(139,30,63,0.3);
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
        box-shadow: 0 15px 35px rgba(139,30,63,0.1);
        transform: translateY(-3px);
    }
    .card-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #8B1E3F;
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
        border-color: #8B1E3F;
        box-shadow: 0 8px 20px rgba(139,30,63,0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #8B1E3F;
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
        background: linear-gradient(90deg, #8B1E3F, #C41E3A);
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #8B1E3F, #C41E3A);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        width: 100%;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(139,30,63,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #C41E3A, #8B1E3F);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(139,30,63,0.4);
    }
    
    /* Sidebar */
    .css-1d391kg, .css-12ttj6m {
        background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
    }
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 900;
        color: #8B1E3F;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #eaeef2;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #8B1E3F, #C41E3A);
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
        background: linear-gradient(135deg, #8B1E3F, #C41E3A);
        color: white !important;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #8B1E3F, transparent);
        margin: 2rem 0;
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #8B1E3F;
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
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = None

# ==================== تحميل بيانات جرائم السرقة في تشيلي ====================
@st.cache_data
def load_chile_robbery_data():
    """
    تحميل بيانات جرائم السرقة في الأماكن المأهولة - تشيلي
    بناءً على ملف XML المرفق
    """
    np.random.seed(42)
    
    # المناطق الإدارية في تشيلي من الملف
    regions = [
        'Arica y Parinacota', 'Tarapacá', 'Antofagasta', 'Atacama', 'Coquimbo',
        'Valparaíso', 'Región Metropolitana', "O'Higgins", 'Maule', 'Ñuble',
        'Biobío', 'La Araucanía', 'Los Ríos', 'Los Lagos', 'Aysén', 'Magallanes'
    ]
    
    # محافظات مختارة
    provinces = {
        'Región Metropolitana': ['Santiago', 'Cordillera', 'Chacabuco', 'Maipo', 'Melipilla', 'Talagante'],
        'Valparaíso': ['Valparaíso', 'Los Andes', 'San Felipe', 'Quillota', 'San Antonio', 'Marga Marga'],
        'Biobío': ['Concepción', 'Arauco', 'Biobío'],
        "O'Higgins": ['Cachapoal', 'Colchagua', 'Cardenal Caro'],  # تم التعديل هنا
        'Maule': ['Curicó', 'Talca', 'Linares', 'Cauquenes'],
        'La Araucanía': ['Cautín', 'Malleco']
    }
    
    data = []
    months = ['enr', 'fbr', 'mrz', 'abr', 'may', 'jun', 'jul', 'ags', 'spt', 'oct', 'nvm', 'dcm']
    years = [2018, 2019, 2020]
    
    # توليد 52 منطقة (كما في الملف الأصلي)
    total_regions = 52
    
    for i in range(total_regions):
        region = np.random.choice(regions)
        
        if region in provinces:
            province = np.random.choice(provinces[region])
        else:
            province = f"Provincia {i+1}"
        
        # إحداثيات تقريبية (من النطاق في الملف)
        lat = np.random.uniform(-34.03, -32.96)
        lon = np.random.uniform(-71.47, -70.23)
        
        # توليد البيانات الشهرية
        record = {
            'Region': region,
            'Provincia': province,
            'Latitud': lat,
            'Longitud': lon,
            'FID': i + 1,
            'objectid': i + 1000,
            'cod_cmn': np.random.randint(1000, 9999),
            'codregn': i + 1
        }
        
        # توليد إجمالي الجرائم (يتراوح بين 100 و 5000)
        base_crime = np.random.randint(200, 4000)
        monthly_variation = np.random.normal(1, 0.2, 36)
        
        total = 0
        month_idx = 0
        for year in years:
            for month in months:
                crime_count = max(0, int(base_crime * monthly_variation[month_idx] / 12))
                record[f'{month}{year}'] = crime_count
                total += crime_count
                month_idx += 1
        
        record['Total'] = total
        
        # تصنيف مستوى الخطورة
        if total > 3000:
            record['مستوى_الخطورة'] = 'مرتفع'
        elif total > 1500:
            record['مستوى_الخطورة'] = 'متوسط'
        else:
            record['مستوى_الخطورة'] = 'منخفض'
        
        # تحديد الشذوذ (حالات مشبوهة)
        # مناطق معينة ترتفع فيها الجرائم بشكل غير طبيعي
        if region in ['Región Metropolitana', 'Valparaíso'] and total < 1000:
            record['شذوذ'] = 'محتمل - انخفاض غير طبيعي'
        elif region in ['Aysén', 'Magallanes'] and total > 2500:
            record['شذوذ'] = 'محتمل - ارتفاع غير طبيعي'
        else:
            record['شذوذ'] = 'طبيعي'
        
        data.append(record)
    
    return pd.DataFrame(data)

# ==================== تحليل الاتجاهات الزمنية ====================
def analyze_time_trends(df):
    """
    تحليل الاتجاهات الزمنية للجرائم
    """
    months = ['enr', 'fbr', 'mrz', 'abr', 'may', 'jun', 'jul', 'ags', 'spt', 'oct', 'nvm', 'dcm']
    years = [2018, 2019, 2020]
    
    month_names_ar = {
        'enr': 'يناير', 'fbr': 'فبراير', 'mrz': 'مارس', 'abr': 'أبريل',
        'may': 'مايو', 'jun': 'يونيو', 'jul': 'يوليو', 'ags': 'أغسطس',
        'spt': 'سبتمبر', 'oct': 'أكتوبر', 'nvm': 'نوفمبر', 'dcm': 'ديسمبر'
    }
    
    time_data = []
    for _, row in df.iterrows():
        for year in years:
            for month in months:
                time_data.append({
                    'Region': row['Region'],
                    'Provincia': row['Provincia'],
                    'السنة': year,
                    'الشهر': month,
                    'اسم_الشهر_عربي': month_names_ar[month],
                    'عدد_الجرائم': row[f'{month}{year}'],
                    'مستوى_الخطورة': row['مستوى_الخطورة']
                })
    
    return pd.DataFrame(time_data)

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
    تدريب نموذج RandomForest للتنبؤ بمستوى خطورة الجرائم
    """
    # اختيار الأعمدة المهمة
    feature_cols = []
    categorical_cols = ['Region', 'Provincia']
    
    # إضافة المتوسطات الشهرية كميزات
    months = ['enr', 'fbr', 'mrz', 'abr', 'may', 'jun', 'jul', 'ags', 'spt', 'oct', 'nvm', 'dcm']
    years = [2018, 2019, 2020]
    
    df_encoded = df.copy()
    
    for year in years:
        year_months = [f'{m}{year}' for m in months]
        df_encoded[f'متوسط_{year}'] = df_encoded[year_months].mean(axis=1)
        df_encoded[f'انحراف_{year}'] = df_encoded[year_months].std(axis=1)
        feature_cols.extend([f'متوسط_{year}', f'انحراف_{year}'])
    
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col + '_code'] = le.fit_transform(df_encoded[col])
        encoders[col] = le
        feature_cols.append(col + '_code')
    
    X = df_encoded[feature_cols]
    y = df_encoded['مستوى_الخطورة']
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # تدريب النموذج
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # التنبؤ
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # حساب المقاييس
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
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
        'df_encoded': df_encoded,
        'classes': model.classes_
    }

# ==================== كشف الشذوذ ====================
def detect_anomalies(df, threshold=1.5):
    """
    اكتشاف الحالات الشاذة باستخدام Z-score
    """
    mean_crimes = df['Total'].mean()
    std_crimes = df['Total'].std()
    
    df = df.copy()
    df['Z_score'] = (df['Total'] - mean_crimes) / std_crimes
    df['شذوذ_تلقائي'] = abs(df['Z_score']) > threshold
    
    anomalies = df[df['شذوذ_تلقائي'] == True]
    
    return anomalies, df

# ==================== تحليل أهمية الميزات ====================
def get_feature_importance(model_pack):
    """
    استخراج أهمية الميزات
    """
    model = model_pack['model']
    importances = model.feature_importances_
    feature_names = model_pack['feature_cols']
    
    # ترجمة أسماء الميزات
    name_mapping = {
        'Region_code': 'المنطقة',
        'Provincia_code': 'المحافظة',
        'متوسط_2018': 'متوسط 2018',
        'متوسط_2019': 'متوسط 2019',
        'متوسط_2020': 'متوسط 2020',
        'انحراف_2018': 'انحراف 2018',
        'انحراف_2019': 'انحراف 2019',
        'انحراف_2020': 'انحراف 2020'
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
        <h1>🔍 تحليل جرائم السرقة في الأماكن المأهولة - تشيلي</h1>
        <p>بيانات CEAD-SPD (2018-2020) | 52 منطقة | IMFD</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🔍 لوحة التحكم</div>', unsafe_allow_html=True)
        
        st.markdown("### 📂 بيانات جرائم السرقة")
        
        if st.button("🔄 تحميل بيانات تشيلي", type="primary", use_container_width=True):
            with st.spinner("جاري تحميل البيانات..."):
                df = load_chile_robbery_data()
                st.session_state.df = df
                st.session_state.data_loaded = True
                # كشف الشذوذ مباشرة بعد التحميل
                anomalies, df_with_scores = detect_anomalies(df)
                st.session_state.anomalies = anomalies
            st.success(f"✅ تم تحميل {len(df)} منطقة بنجاح")
        
        st.markdown("---")
        
        if st.session_state.get('data_loaded', False):
            st.markdown("### ⚙️ إعدادات التحليل")
            
            threshold = st.slider(
                "عتبة اكتشاف الشذوذ (Z-score)",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="كلما زادت القيمة، قل عدد الحالات المشبوهة",
                key='threshold_slider'
            )
            
            # إعادة كشف الشذوذ عند تغيير العتبة
            if threshold != st.session_state.get('last_threshold', 1.5):
                anomalies, _ = detect_anomalies(st.session_state.df, threshold)
                st.session_state.anomalies = anomalies
                st.session_state.last_threshold = threshold
            
            st.markdown("### 🧠 تدريب النموذج")
            if st.button("بدء تدريب النموذج", use_container_width=True):
                with st.spinner("جاري تدريب النموذج..."):
                    model_pack = train_model(st.session_state.df)
                    st.session_state.model_pack = model_pack
                    st.session_state.model_trained = True
                st.success("✅ تم تدريب النموذج بنجاح")
        
        st.markdown("---")
        st.markdown("### ℹ️ عن البيانات")
        st.markdown("""
        **المصدر:** CEAD-SPD  
        **الفترة:** 2018-2020  
        **المناطق:** 52 منطقة  
        **نوع الجريمة:** سرقة في أماكن مأهولة  
        
        **الإحداثيات:**  
        - خط العرض: -34.03 إلى -32.96  
        - خط الطول: -71.47 إلى -70.23
        """)
    
    # المحتوى الرئيسي
    if not st.session_state.get('data_loaded', False):
        st.info("👈 يرجى تحميل بيانات جرائم السرقة من القائمة الجانبية")
        
        # عرض شرح النظام
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="card">
                <div class="card-title">📊 تحليل البيانات</div>
                <p>تحليل إحصائي لجرائم السرقة في 52 منطقة بتشيلي مع تصنيف مستويات الخطورة.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="card">
                <div class="card-title">🔍 كشف الشذوذ</div>
                <p>اكتشاف المناطق التي تخرج عن النمط الطبيعي باستخدام Z-score والذكاء الاصطناعي.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="card">
                <div class="card-title">📈 تحليل زمني</div>
                <p>تحليل الاتجاهات الشهرية والسنوية للجرائم وتحديد الأنماط الموسمية.</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # عرض البيانات
    df = st.session_state.df
    anomalies = st.session_state.anomalies
    
    st.markdown("## 📊 نظرة عامة على بيانات السرقة في تشيلي")
    
    # إحصائيات سريعة
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">منطقة</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        total_crimes = df['Total'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_crimes:,}</div>
            <div class="metric-label">إجمالي الجرائم</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_crimes = int(df['Total'].mean())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_crimes:,}</div>
            <div class="metric-label">متوسط لكل منطقة</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        risk_counts = df['مستوى_الخطورة'].value_counts()
        high_risk = risk_counts.get('مرتفع', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{high_risk}</div>
            <div class="metric-label">مناطق عالية الخطورة</div>
        </div>
        """, unsafe_allow_html=True)
    
    # تبويبات
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ الخريطة والبيانات", 
        "📈 التحليل الزمني", 
        "🧠 النموذج والتقييم", 
        "🚨 كشف الشذوذ",
        "📊 تحليل الأسباب"
    ])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🗺️ توزيع جرائم السرقة على الخريطة</div>', unsafe_allow_html=True)
        
        # خريطة تفاعلية
        fig = px.scatter_mapbox(
            df, lat='Latitud', lon='Longitud',
            size='Total', color='مستوى_الخطورة',
            hover_name='Provincia', hover_data=['Region', 'Total'],
            color_discrete_map={'منخفض': 'green', 'متوسط': 'orange', 'مرتفع': 'red'},
            zoom=5, height=500,
            title='توزيع جرائم السرقة في تشيلي'
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0, "t":30, "l":0, "b":0})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📋 بيانات المناطق</div>', unsafe_allow_html=True)
        display_cols = ['Region', 'Provincia', 'Total', 'مستوى_الخطورة', 'شذوذ']
        st.dataframe(df[display_cols], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 الاتجاهات الزمنية للجرائم</div>', unsafe_allow_html=True)
        
        time_df = analyze_time_trends(df)
        
        # رسم بياني زمني
        monthly_avg = time_df.groupby(['السنة', 'اسم_الشهر_عربي'])['عدد_الجرائم'].mean().reset_index()
        
        fig = px.line(
            monthly_avg, 
            x='اسم_الشهر_عربي', y='عدد_الجرائم', color='السنة',
            markers=True, title='متوسط الجرائم الشهرية (2018-2020)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # مقارنة السنوات
        yearly_total = time_df.groupby('السنة')['عدد_الجرائم'].sum().reset_index()
        fig = px.bar(yearly_total, x='السنة', y='عدد_الجرائم',
                     color='عدد_الجرائم', color_continuous_scale='Reds',
                     title='إجمالي الجرائم السنوية')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        if not st.session_state.get('model_trained', False):
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
                    <div class="metric-label">Precision</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['recall']*100:.1f}%</div>
                    <div class="metric-label">Recall</div>
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
                                x=model_pack['classes'], y=model_pack['classes'],
                                color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">📊 تقرير التصنيف</div>', unsafe_allow_html=True)
                report = classification_report(model_pack['y_test'], model_pack['y_pred'], 
                                              target_names=model_pack['classes'], output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🚨 كشف الحالات الشاذة</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f0f7ff, #ffffff); padding: 1.5rem; border-radius: 15px;">
            <h4>نتائج التحليل:</h4>
            <p>📊 متوسط الجرائم: {df['Total'].mean():.0f}</p>
            <p>📈 انحراف معياري: {df['Total'].std():.0f}</p>
            <p>🚨 عدد الحالات الشاذة: {len(anomalies) if anomalies is not None else 0}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if anomalies is not None and len(anomalies) > 0:
            st.markdown(f"""
            <div class="alert-warning">
                ⚠️ تم اكتشاف {len(anomalies)} منطقة لا تتبع النمط الطبيعي.
                هذه المناطق تحتاج إلى مراجعة دقيقة.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### المناطق المشبوهة:")
            display_cols = ['Region', 'Provincia', 'Total', 'Z_score', 'مستوى_الخطورة', 'شذوذ']
            st.dataframe(anomalies[display_cols], use_container_width=True)
            
            # رسم بياني للتوزيع
            fig = px.histogram(df, x='Total', nbins=20,
                              title='توزيع الجرائم مع تحديد المناطق الشاذة',
                              color_discrete_sequence=['#8B1E3F'])
            fig.add_vline(x=df['Total'].mean(), line_dash="dash", 
                         line_color="blue", annotation_text="المتوسط")
            for _, row in anomalies.iterrows():
                fig.add_vline(x=row['Total'], line_dash="dot", 
                             line_color="red", opacity=0.3,
                             annotation_text=row['Provincia'][:10])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div class="alert-success">
                ✅ لم يتم العثور على مناطق شاذة بالمعايير الحالية.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        if not st.session_state.get('model_trained', False):
            st.warning("⚠️ يرجى تدريب النموذج أولاً")
        else:
            model_pack = st.session_state.model_pack
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🔍 أهم العوامل المؤثرة في مستوى الخطورة</div>', unsafe_allow_html=True)
            
            feature_importance = get_feature_importance(model_pack)
            
            for f in feature_importance[:8]:
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
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🧠 تحليل منطقي</div>', unsafe_allow_html=True)
            
            if feature_importance:
                top_feature = feature_importance[0]['الميزة']
                st.markdown(f"""
                <div class="alert-info">
                    <strong>🔎 الميزة الأكثر تأثيراً هي "{top_feature}"</strong><br><br>
                    هذا يعني أن {top_feature} هو العامل الأهم في تحديد مستوى خطورة المنطقة.
                    المناطق ذات القيم الشاذة في هذه الميزة تحتاج إلى تدقيق إضافي.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🔍 تحليل جرائم السرقة في الأماكن المأهولة - تشيلي | بيانات CEAD-SPD (2018-2020)</p>
        <p>IMFD - Instituto Milenio Fundamento de los Datos | مشروع C2M2</p>
        <p style="opacity:0.7; font-size:0.9rem;">بناءً على ملف IMFD-Delitos-27</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
