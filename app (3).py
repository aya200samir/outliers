# -*- coding: utf-8 -*-
"""
واجهة المستخدم لتحليل جرائم السرقة في الأماكن المأهولة - تشيلي
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import warnings
warnings.filterwarnings('ignore')

# ==================== مكتبات التعلم الآلي ====================
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, mean_squared_error, r2_score)

# ==================== إعدادات الصفحة ====================
st.set_page_config(
    page_title="تحليل جرائم السرقة - تشيلي",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS مخصص ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');
    * { font-family: 'Cairo', sans-serif; }
    
    .header {
        background: linear-gradient(135deg, #922B21, #CB4335);
        color: white;
        padding: 2rem;
        border-radius: 0 0 30px 30px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .header h1 { 
        font-size: 3rem; 
        font-weight: 900; 
        margin-bottom: 0.5rem;
    }
    .header p { 
        font-size: 1.2rem; 
        opacity: 0.9;
    }
    
    .card {
        background: white;
        border-radius: 20px;
        padding: 1.8rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #eaeef2;
    }
    .card-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #CB4335;
        margin-bottom: 1.2rem;
        border-bottom: 2px solid #eaeef2;
        padding-bottom: 0.7rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8fbff, #ffffff);
        border-radius: 18px;
        padding: 1.2rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.03);
        text-align: center;
        border: 1px solid #dde5ed;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #CB4335;
    }
    
    .footer {
        background: linear-gradient(135deg, #922B21, #CB4335);
        color: white;
        padding: 2rem;
        border-radius: 30px 30px 0 0;
        margin-top: 4rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==================== دوال معالجة البيانات المكانية ====================

@st.cache_data
def load_chile_crime_data():
    """
    محاكاة لبيانات جرائم تشيلي (نظراً لعدم توفر ملف Shapefile الفعلي)
    """
    # مناطق تشيلي
    regions = [
        'Arica y Parinacota', 'Tarapacá', 'Antofagasta', 'Atacama', 'Coquimbo',
        'Valparaíso', 'Metropolitana', "O'Higgins", 'Maule', 'Ñuble',
        'Biobío', 'La Araucanía', 'Los Ríos', 'Los Lagos', 'Aysén', 'Magallanes'
    ]
    
    # محافظات مختارة
    provinces = {
        'Metropolitana': ['Santiago', 'Cordillera', 'Chacabuco', 'Maipo', 'Melipilla', 'Talagante'],
        'Valparaíso': ['Valparaíso', 'Los Andes', 'San Felipe', 'Quillota', 'San Antonio'],
        'Biobío': ['Concepción', 'Arauco', 'Biobío']
    }
    
    data = []
    np.random.seed(42)
    
    # توليد بيانات لـ 52 منطقة (كما في الملف الأصلي)
    for i in range(52):
        region = np.random.choice(regions)
        
        if region in provinces:
            province = np.random.choice(provinces[region])
        else:
            province = f"Provincia {i+1}"
        
        # بيانات شهرية للسنوات 2018-2020
        months = ['enr', 'fbr', 'mrz', 'abr', 'may', 'jun', 'jul', 'ags', 'spt', 'oct', 'nvm', 'dcm']
        
        record = {
            'Region': region,
            'Provincia': province,
            'Latitud': -33.0 + np.random.randn() * 2,
            'Longitud': -70.0 + np.random.randn() * 2,
            'Total': np.random.randint(100, 5000)
        }
        
        # إضافة بيانات شهرية
        base_crime = record['Total'] / 36  # متوسط شهري
        for year in [2018, 2019, 2020]:
            for month in months:
                variation = np.random.normal(1, 0.3)
                record[f'{month}{year}'] = max(0, int(base_crime * variation))
        
        # تصنيف الخطورة (للتحدي: target)
        if record['Total'] > 3000:
            record['مستوى_الخطورة'] = 'مرتفع'
        elif record['Total'] > 1500:
            record['مستوى_الخطورة'] = 'متوسط'
        else:
            record['مستوى_الخطورة'] = 'منخفض'
        
        data.append(record)
    
    return pd.DataFrame(data)

def prepare_time_series_data(df):
    """
    تحويل البيانات الشهرية إلى صيغة مناسبة للتحليل الزمني
    """
    months = ['enr', 'fbr', 'mrz', 'abr', 'may', 'jun', 'jul', 'ags', 'spt', 'oct', 'nvm', 'dcm']
    years = [2018, 2019, 2020]
    
    time_data = []
    for _, row in df.iterrows():
        for year in years:
            for i, month in enumerate(months):
                time_data.append({
                    'Region': row['Region'],
                    'Provincia': row['Provincia'],
                    'السنة': year,
                    'الشهر': i + 1,
                    'اسم_الشهر': month,
                    'عدد_الجرائم': row[f'{month}{year}'],
                    'Total': row['Total'],
                    'مستوى_الخطورة': row['مستوى_الخطورة']
                })
    
    return pd.DataFrame(time_data)

# ==================== تدريب النموذج ====================
def train_crime_model(df):
    """
    تدريب نموذج للتنبؤ بمستوى الخطورة
    """
    feature_cols = ['Total']
    categorical_cols = ['Region', 'Provincia']
    
    df_encoded = df.copy()
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col + '_code'] = le.fit_transform(df_encoded[col])
        encoders[col] = le
        feature_cols.append(col + '_code')
    
    # إضافة إحصائيات موسمية
    months_cols = [col for col in df.columns if any(m in col for m in ['enr', 'fbr', 'mrz'])]
    if months_cols:
        df_encoded['متوسط_الربع_الأول'] = df[months_cols[:3]].mean(axis=1)
        df_encoded['انحراف_الربع_الأول'] = df[months_cols[:3]].std(axis=1)
        feature_cols.extend(['متوسط_الربع_الأول', 'انحراف_الربع_الأول'])
    
    X = df_encoded[feature_cols]
    y = df_encoded['مستوى_الخطورة']
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # تدريب النموذج
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # التنبؤ
    y_pred = model.predict(X_test)
    
    # حساب المقاييس
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    return {
        'model': model,
        'encoders': encoders,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_importance': dict(zip(feature_cols, model.feature_importances_))
    }

# ==================== الصفحة الرئيسية ====================
def main():
    st.markdown("""
    <div class="header">
        <h1>🗺️ تحليل جرائم السرقة في الأماكن المأهولة - تشيلي</h1>
        <p>IMFD - بيانات CEAD-SPD (2018-2020)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🔍 لوحة التحكم</div>', unsafe_allow_html=True)
        
        st.markdown("### 📂 البيانات")
        data_source = st.radio(
            "مصدر البيانات",
            ["📊 بيانات محاكاة (Chile)", "📁 رفع Shapefile"],
            index=0
        )
        
        if data_source == "📁 رفع Shapefile":
            uploaded_file = st.file_uploader("ارفع ملف .shp", type=['shp'])
            if uploaded_file:
                st.info("ملف Shapefile يحتاج إلى ملفات .shx, .dbf, .prj أيضاً")
        else:
            if st.button("🔄 تحميل بيانات تشيلي"):
                with st.spinner("جاري تحميل البيانات..."):
                    df = load_chile_crime_data()
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                st.success("✅ تم تحميل 52 منطقة بنجاح")
        
        if st.session_state.get('data_loaded', False):
            st.markdown("### 🧠 تدريب النموذج")
            if st.button("بدء التدريب", type="primary"):
                with st.spinner("جاري تدريب النموذج..."):
                    model_pack = train_crime_model(st.session_state.df)
                    st.session_state.model_pack = model_pack
                    st.session_state.model_trained = True
                st.success("✅ تم تدريب النموذج بنجاح")
    
    # المحتوى الرئيسي
    if not st.session_state.get('data_loaded', False):
        st.info("👈 يرجى تحميل البيانات من القائمة الجانبية")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="card">
                <div class="card-title">📊 عن البيانات</div>
                <p>جرائم السرقة في الأماكن المأهولة (robbery in inhabited place) في تشيلي</p>
                <p><strong>المصدر:</strong> CEAD-SPD</p>
                <p><strong>الفترة:</strong> 2018-2020</p>
                <p><strong>عدد المناطق:</strong> 52 منطقة</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <div class="card-title">🗺️ نطاق البيانات</div>
                <p><strong>خط العرض:</strong> -34.03 إلى -32.96</p>
                <p><strong>خط الطول:</strong> -71.47 إلى -70.23</p>
                <p><strong>نظام الإحداثيات:</strong> WGS 1984 Web Mercator</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # عرض البيانات
    df = st.session_state.df
    
    st.markdown("## 📊 نظرة عامة على البيانات")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">منطقة</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['Total'].sum():,}</div>
            <div class="metric-label">إجمالي الجرائم</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['Region'].nunique()}</div>
            <div class="metric-label">منطقة إدارية</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        risk_dist = df['مستوى_الخطورة'].value_counts()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{risk_dist.get('مرتفع', 0)}</div>
            <div class="metric-label">مناطق خطرة</div>
        </div>
        """, unsafe_allow_html=True)
    
    # تبويبات
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ الخريطة", "📈 تحليل زمني", "🧠 النموذج", "🔍 كشف الشذوذ"
    ])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🗺️ توزيع الجرائم على الخريطة</div>', unsafe_allow_html=True)
        
        # خريطة بسيطة
        fig = px.scatter_mapbox(
            df, lat='Latitud', lon='Longitud',
            size='Total', color='مستوى_الخطورة',
            hover_name='Provincia', hover_data=['Region', 'Total'],
            color_discrete_map={'منخفض': 'green', 'متوسط': 'orange', 'مرتفع': 'red'},
            zoom=5, height=500
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 الاتجاهات الزمنية</div>', unsafe_allow_html=True)
        
        # تحويل البيانات للتحليل الزمني
        time_df = prepare_time_series_data(df)
        
        # رسم بياني زمني
        fig = px.line(
            time_df.groupby(['السنة', 'الشهر'])['عدد_الجرائم'].mean().reset_index(),
            x='الشهر', y='عدد_الجرائم', color='السنة',
            markers=True, title='متوسط الجرائم الشهرية'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # مقارنة بين المناطق
        top_regions = df.nlargest(5, 'Total')[['Region', 'Total']]
        st.markdown("#### أعلى 5 مناطق في معدل الجرائم")
        st.dataframe(top_regions, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        if not st.session_state.get('model_trained', False):
            st.warning("⚠️ يرجى تدريب النموذج أولاً")
        else:
            model_pack = st.session_state.model_pack
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 أداء النموذج</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("الدقة", f"{model_pack['metrics']['accuracy']*100:.1f}%")
            with col2:
                st.metric("Precision", f"{model_pack['metrics']['precision']*100:.1f}%")
            with col3:
                st.metric("Recall", f"{model_pack['metrics']['recall']*100:.1f}%")
            with col4:
                st.metric("F1 Score", f"{model_pack['metrics']['f1']*100:.1f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 أهمية الميزات</div>', unsafe_allow_html=True)
            
            # عرض أهمية الميزات
            importance_df = pd.DataFrame(
                list(model_pack['feature_importance'].items()),
                columns=['الميزة', 'الأهمية']
            ).sort_values('الأهمية', ascending=False)
            
            fig = px.bar(importance_df.head(10), x='الأهمية', y='الميزة',
                        orientation='h', title='أهم 10 ميزات')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 كشف الحالات الشاذة</div>', unsafe_allow_html=True)
        
        # حساب الانحراف المعياري
        mean_crimes = df['Total'].mean()
        std_crimes = df['Total'].std()
        
        df['انحراف'] = (df['Total'] - mean_crimes) / std_crimes
        anomalies = df[abs(df['انحراف']) > 1.5]
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f0f7ff, #ffffff); padding: 1rem; border-radius: 10px;">
            <h4>نتائج التحليل:</h4>
            <p>📊 متوسط الجرائم: {mean_crimes:.0f}</p>
            <p>📈 انحراف معياري: {std_crimes:.0f}</p>
            <p>🚨 عدد الحالات الشاذة: {len(anomalies)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if len(anomalies) > 0:
            st.markdown("#### المناطق الشاذة:")
            st.dataframe(
                anomalies[['Region', 'Provincia', 'Total', 'انحراف', 'مستوى_الخطورة']],
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🗺️ تحليل جرائم السرقة في تشيلي | بيانات CEAD-SPD (2018-2020) | IMFD</p>
        <p style="opacity:0.7;">تم التطوير بناءً على ملف IMFD-Delitos-27</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
