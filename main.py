import streamlit as st
import requests

st.set_page_config(page_title="PathGenie", layout="wide", initial_sidebar_state="expanded")

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
load_css("styles.css")

# Load Lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Hero Section
st.markdown("""
<div class="hero-section">
            <div style= "font-size:4rem">🧞‍♂️</div>
    <div class="main-title floating"> PathGenie</div>
    <div class="subtitle">Your AI-Powered Career Navigation Companion</div>
    <p style="font-size: 1.2rem; color: #E8E8E8; max-width: 600px; margin: 0 auto;">
        Unlock your career potential with intelligent resume analysis, job matching, and personalized career insights.
    </p>
</div>
""", unsafe_allow_html=True)

# Stats Section
st.markdown("""
<div class="stats-container">
    <div class="stat-item">
        <span class="stat-number">10K+</span>
        <span class="stat-label">Resumes Analyzed</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">95%</span>
        <span class="stat-label">Accuracy Rate</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">50+</span>
        <span class="stat-label">Job Categories</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">24/7</span>
        <span class="stat-label">AI Assistant</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Features Section
st.markdown('<h2 style="text-align: center; color: white; margin: 3rem 0 2rem 0; font-size: 2.5rem;">🚀 Powerful Features</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card pulse">
        <div class="feature-icon">📄</div>
        <div class="feature-title">Resume Classifier</div>
        <div class="feature-description">
            Get instant job role predictions and comprehensive skill gap analysis from your resume. 
            Our AI analyzes your experience and suggests career paths tailored to your expertise.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card pulse">
        <div class="feature-icon">🎯</div>
        <div class="feature-title">Job Fit Analyzer</div>
        <div class="feature-description">
            Analyze your compatibility with specific job descriptions. Get detailed insights on skill matches, 
            experience alignment, and improvement recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card pulse">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Analytics Dashboard</div>
        <div class="feature-description">
            View comprehensive model performance metrics, success rates, and detailed analytics. 
            Track your career progress and optimization opportunities.
        </div>
    </div>
    """, unsafe_allow_html=True)


# How it works section
st.markdown('<h2 style="text-align: center; color: white; margin: 3rem 0 2rem 0; font-size: 2.5rem;">⚡ How It Works</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="text-align: center; color: white; padding: 1rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📤</div>
        <h3 style="color: #FFD700;">Upload</h3>
        <p>Upload your resume or paste job description</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; color: white; padding: 1rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
        <h3 style="color: #FFD700;">Analyze</h3>
        <p>AI processes and analyzes your content</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; color: white; padding: 1rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">💡</div>
        <h3 style="color: #FFD700;">Insights</h3>
        <p>Get personalized recommendations</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="text-align: center; color: white; padding: 1rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🚀</div>
        <h3 style="color: #FFD700;">Succeed</h3>
        <p>Land your dream job with confidence</p>
    </div>
    """, unsafe_allow_html=True)


# Testimonials Section
st.markdown('<h2 style="text-align: center; color: white; margin: 3rem 0 2rem 0; font-size: 2.5rem;">💬 What Users Say</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="testimonial-card">
        <div class="testimonial-text">
            "PathGenie helped me identify skill gaps I never knew I had. The recommendations were spot-on!"
        </div>
        <div class="testimonial-author">- Sarah M., Software Engineer</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="testimonial-card">
        <div class="testimonial-text">
            "The job fit analyzer saved me hours of applying to unsuitable positions. Highly recommended!"
        </div>
        <div class="testimonial-author">- Mike R., Data Scientist</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="testimonial-card">
        <div class="testimonial-text">
            "Incredible accuracy in resume classification. It guided my career transition perfectly."
        </div>
        <div class="testimonial-author">- Lisa K., Marketing Manager</div>
    </div>
    """, unsafe_allow_html=True)


# Call to Action
st.markdown("""
<div class="cta-section">
    <h2 style="color: white; margin-bottom: 1rem; font-size: 2.5rem;">Ready to Transform Your Career?</h2>
    <p style="color: #E8E8E8; font-size: 1.2rem; margin-bottom: 2rem;">
        Join thousands of professionals who've discovered their perfect career path with PathGenie
    </p>
</div>
""", unsafe_allow_html=True)


# Enhanced Info Box
st.markdown("""
<div class="info-box">
    <h3 style="margin-bottom: 1rem; display: flex; align-items: center;">
        <span style="margin-right: 10px;">🎯</span>
        Get Started Now!
    </h3>
    <p style="margin-bottom: 1rem; font-size: 1.1rem;">
        Choose from our powerful tools in the sidebar to begin your career transformation journey:
    </p>
    <ul style="margin-left: 1rem; font-size: 1rem;">
        <li><strong>Resume Classifier</strong> - Discover your ideal job roles and skill gaps</li>
        <li><strong>Job Fit Analyzer</strong> - Match your profile with specific opportunities</li>
        <li><strong>Dashboard</strong> - Track your progress and view analytics</li>
    </ul>
</div>
""", unsafe_allow_html=True)


# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; color: white; background: rgba(255, 255, 255, 0.1); border-radius: 15px;">
    <p style="font-size: 1.1rem; margin-bottom: 1rem;">
        🌟 <span class="gradient-text">PathGenie</span> - Where AI Meets Career Success 🌟
    </p>
    <p style="color: #E8E8E8; font-size: 0.9rem;">
        Powered by advanced machine learning algorithms and designed for career professionals
    </p>
</div>
""", unsafe_allow_html=True)


# Sidebar enhancement
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: #FFD700; margin-bottom: 0.5rem;">🧞‍♂️ PathGenie</h2>
        <p style="color: white; font-size: 0.9rem;">Your Career Companion</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")
    
st.sidebar.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h4 style="color: #FFD700; margin-bottom: 0.5rem;">📈 Quick Stats</h4>
        <p style="color: white; font-size: 0.8rem; margin-bottom: 0.5rem;">• 95% Accuracy Rate</p>
        <p style="color: white; font-size: 0.8rem; margin-bottom: 0.5rem;">• 10K+ Resumes Processed</p>
        <p style="color: white; font-size: 0.8rem; margin-bottom: 0.5rem;">• 50+ Job Categories</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")

st.sidebar.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
        <h4 style="color: #FFD700; margin-bottom: 0.5rem;">💡 Tips</h4>
        <p style="color: white; font-size: 0.8rem;">Upload a well-formatted resume for best results!</p>
    </div>
    """, unsafe_allow_html=True)
