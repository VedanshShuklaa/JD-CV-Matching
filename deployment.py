import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from pathlib import Path
import hashlib
import sqlite3
import uuid
from functools import wraps

from main import CVMatcher, MatchConfig

def init_db():
    conn = sqlite3.connect('user_auth.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password TEXT,
        created_at TIMESTAMP
    )
    ''')
    conn.commit()
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def generate_id():
    return str(uuid.uuid4())

def register_user(username, email, password):
    conn = st.session_state.db_connection
    c = conn.cursor()
    
    c.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
    if c.fetchone():
        return False, "Username or email already exists"
    
    user_id = generate_id()
    hashed_pw = hash_password(password)
    try:
        c.execute(
            "INSERT INTO users (id, username, email, password, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, email, hashed_pw, datetime.now())
        )
        conn.commit()
        return True, "Registration successful"
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def login_user(username, password):
    conn = st.session_state.db_connection
    c = conn.cursor()
    
    hashed_pw = hash_password(password)
    c.execute("SELECT id, username, email FROM users WHERE username = ? AND password = ?", (username, hashed_pw))
    user = c.fetchone()
    
    if user:
        return True, {
            "id": user[0],
            "username": user[1],
            "email": user[2]
        }
    return False, "Invalid username or password"

def require_login(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if st.session_state.get("authenticated", False):
            return func(*args, **kwargs)
        else:
            st.warning("Please log in to access this feature")
            show_login_page()
    return wrapper

def show_login_page():
    st.markdown('<div class="section-header">Login</div>', unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if username and password:
                success, result = login_user(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user = result
                    st.success("Login successful!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(result)
            else:
                st.error("Please fill in all fields")
    
    st.markdown("Don't have an account? [Register](#register)")
    if st.session_state.get("show_register", False):
        show_register_page()

def show_logout_ui():
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user = None
        st.success("Logged out successfully!")
        time.sleep(1)
        st.rerun()

def save_uploaded_file(uploaded_file, directory):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix, dir=directory) as tmp:
            tmp.write(uploaded_file.getbuffer())
            return tmp.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def display_results(results, job_description_text):
    if not results:
        st.warning("No results to display.")
        return

    sorted_results = sorted(results, key=lambda x: x.total_score, reverse=True)
    
    tab1, tab2, tab3 = st.tabs(["Overview", "Detailed Analysis", "Skills Comparison"])
    
    with tab1:
        st.markdown('<div class="sub-header">Overall Match Scores</div>', unsafe_allow_html=True)
        
        overview_data = []
        for candidate in sorted_results:
            candidate_dict = candidate.to_dict()
            overview_data.append({
                "Candidate ID": candidate_dict["id"],
                "Total Score": round(candidate_dict["total_score"] * 100, 1),
                "Skills Score": round(candidate_dict["details"].get("Skills", 0) * 100, 1),
                "Experience Score": round(candidate_dict["details"].get("Experience", 0) * 100, 1),
                "Education Score": round(candidate_dict["details"].get("Education", 0) * 100, 1),
                "Seniority Match": round(candidate_dict["seniority_match"] * 100, 1),
                "Years of Experience": candidate_dict["years_experience"]
            })
        
        df_overview = pd.DataFrame(overview_data)
        
        def highlight_scores(val):
            if isinstance(val, (int, float)):
                if val >= 80:
                    return 'background-color: #d4edda; color: #155724'
                elif val >= 60:
                    return 'background-color: #fff3cd; color: #856404'
                elif val < 60:
                    return 'background-color: #f8d7da; color: #721c24'
            return ''
        
        st.dataframe(df_overview.style.applymap(highlight_scores, subset=[
            'Total Score', 'Skills Score', 'Experience Score', 'Education Score', 'Seniority Match'
        ]), height=400)
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### Key Insights")
        
        if len(sorted_results) > 0:
            top_candidate = sorted_results[0].to_dict()
            st.markdown(f"🥇 **Top candidate**: {top_candidate['id']} with a match score of {round(top_candidate['total_score'] * 100, 1)}%")
            
            scores = [r.total_score * 100 for r in sorted_results]
            avg_score = sum(scores) / len(scores)
            st.markdown(f"📊 **Average match score**: {avg_score:.1f}% across {len(sorted_results)} candidates")
            
            jd_skills = matcher.extract_skills(job_description_text)
            candidates_skills = []
            for result in sorted_results:
                candidate_skills = set()
                for skill_list in result.keywords_matched.values():
                    candidate_skills.update(skill_list)
                candidates_skills.append(candidate_skills)
            
            common_skills = set.intersection(*([jd_skills] + candidates_skills)) if candidates_skills else set()
            missing_skills = jd_skills - common_skills
            
            if missing_skills:
                missing_skills_str = ", ".join(list(missing_skills)[:5])
                st.markdown(f"⚠️ **Common skill gaps**: {missing_skills_str}{' and more' if len(missing_skills) > 5 else ''}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sub-header">Score Distribution</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(sorted_results) >= 1:
                top_n = min(3, len(sorted_results))
                categories = ['Skills', 'Experience', 'Education', 'Projects', 'Achievements', 'Summary']
                
                fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
                
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]
                
                for i in range(top_n):
                    candidate = sorted_results[i].to_dict()
                    values = [candidate["details"].get(cat, 0) * 100 for cat in categories]
                    values += values[:1]
                    
                    ax.plot(angles, values, linewidth=2, label=f"Candidate {candidate['id']}")
                    ax.fill(angles, values, alpha=0.1)
                
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_thetagrids(np.degrees(angles[:-1]), categories)
                ax.set_ylim(0, 100)
                ax.set_title("Top Candidates Category Comparison", pad=20)
                ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                
                st.pyplot(fig)
        
        with col2:
            if len(sorted_results) > 0:
                score_data = {
                    "Candidate": [r.to_dict()["id"] for r in sorted_results[:10]],
                    "Score": [r.total_score * 100 for r in sorted_results[:10]]
                }
                
                df_scores = pd.DataFrame(score_data)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                bars = ax.barh(df_scores["Candidate"], df_scores["Score"], color="skyblue")
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                            ha='left', va='center', fontweight='bold')
                
                ax.set_xlabel("Match Score (%)")
                ax.set_title("Top 10 Candidates by Match Score")
                ax.set_xlim(0, 100)
                
                st.pyplot(fig)
    
    with tab2:
        if len(sorted_results) > 0:
            selected_candidate_idx = st.selectbox(
                "Select a candidate for detailed analysis:",
                range(len(sorted_results)),
                format_func=lambda i: f"{sorted_results[i].to_dict()['id']} - Score: {sorted_results[i].total_score * 100:.1f}%"
            )
            
            selected_candidate = sorted_results[selected_candidate_idx].to_dict()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### Candidate: {selected_candidate['id']}")
                st.markdown(f"**Total Match Score:** {selected_candidate['total_score'] * 100:.1f}%")
                
                st.markdown("#### Category Scores")
                category_data = {
                    "Category": list(selected_candidate["details"].keys()),
                    "Score": [selected_candidate["details"][cat] * 100 for cat in selected_candidate["details"]]
                }
                df_categories = pd.DataFrame(category_data)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.barh(df_categories["Category"], df_categories["Score"], 
                               color=plt.cm.viridis(df_categories["Score"]/100))
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                            ha='left', va='center')
                
                ax.set_xlabel("Score (%)")
                ax.set_xlim(0, 100)
                
                st.pyplot(fig)
                
                st.markdown("#### Matched Keywords")
                for category, keywords in selected_candidate["keywords_matched"].items():
                    if keywords:
                        st.markdown(f"**{category}:** {', '.join(keywords)}")
            
            with col2:
                st.markdown("#### Candidate Profile")
                st.markdown(f"**Years of Experience:** {selected_candidate['years_experience']:.1f} years")
                st.markdown(f"**Seniority Match:** {selected_candidate['seniority_match'] * 100:.1f}%")
                
                if "education_details" in selected_candidate and selected_candidate["education_details"]:
                    st.markdown("#### Education")
                    edu = selected_candidate["education_details"]
                    
                    if "highest_degree" in edu:
                        st.markdown(f"**Highest Degree:** {edu['highest_degree'].title()}")
    
    with tab3:
        st.markdown("### Skills Analysis")
        
        if len(sorted_results) > 0:
            all_skills = set()
            skill_depth_by_candidate = {}
            
            for candidate in sorted_results:
                candidate_dict = candidate.to_dict()
                candidate_id = candidate_dict["id"]
                skill_depth_by_candidate[candidate_id] = candidate_dict.get("skill_details", {})
                
                for category, keywords in candidate_dict.get("keywords_matched", {}).items():
                    all_skills.update(keywords)
            
            all_skills_list = sorted(list(all_skills))
            
            skill_data = []
            
            for candidate in sorted_results:
                candidate_dict = candidate.to_dict()
                candidate_id = candidate_dict["id"]
                
                matched_skills = set()
                for keywords in candidate_dict.get("keywords_matched", {}).values():
                    matched_skills.update(keywords)
                
                for skill in all_skills_list:
                    skill_data.append({
                        "Candidate": candidate_id,
                        "Skill": skill,
                        "Matched": skill in matched_skills,
                        "Depth": skill_depth_by_candidate.get(candidate_id, {}).get(skill, 0)
                    })
            
            if skill_data:
                df_skills = pd.DataFrame(skill_data)
                
                top_n = min(10, len(sorted_results))
                top_candidates = [candidate.to_dict()["id"] for candidate in sorted_results[:top_n]]
                
                try:
                    if "Candidate" in df_skills.columns:
                        df_skills_filtered = df_skills[df_skills["Candidate"].isin(top_candidates)]
                        
                        if not df_skills_filtered.empty:
                            pivot_table = df_skills_filtered.pivot_table(
                                index="Skill", 
                                columns="Candidate", 
                                values="Depth",
                                fill_value=0
                            )
                            
                            fig, ax = plt.subplots(figsize=(12, max(8, len(all_skills_list) * 0.3)))
                            sns.heatmap(pivot_table, cmap="YlGnBu", linewidths=0.5, ax=ax, vmin=0, vmax=1, 
                                        cbar_kws={'label': 'Skill Depth'})
                            
                            plt.title("Skill Depth Comparison Across Top Candidates")
                            plt.ylabel("Skills")
                            plt.xlabel("Candidates")
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                        else:
                            st.info("No skill data available for the top candidates.")
                    else:
                        st.error("DataFrame structure is incorrect: 'Candidate' column is missing.")
                        
                except Exception as e:
                    st.error(f"Error creating skills heatmap: {str(e)}")
                
                try:
                    if "Matched" in df_skills.columns and "Skill" in df_skills.columns:
                        skill_freq = df_skills[df_skills["Matched"]].groupby("Skill").size().reset_index(name="Count")
                        
                        if not skill_freq.empty:
                            skill_freq = skill_freq.sort_values("Count", ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.bar(skill_freq["Skill"][:15], skill_freq["Count"][:15], color="skyblue")
                            
                            plt.title("Most Common Skills Among Candidates")
                            plt.xlabel("Skills")
                            plt.ylabel("Number of Candidates")
                            plt.xticks(rotation=45, ha="right")
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                        else:
                            st.info("No matched skills data available for analysis.")
                except Exception as e:
                    st.error(f"Error creating skill frequency chart: {str(e)}")
            else:
                st.info("No skills data available for the candidates.")

@require_login
def app_main():
    st.markdown('<div class="main-header">CV to Job Description Matcher</div>', unsafe_allow_html=True)
    st.markdown(
        """
        This application helps you match candidate CVs against job descriptions using natural language processing 
        and machine learning techniques. Upload your documents below to get started.
        """
    )
    st.sidebar.markdown("""
                        <div style="display:flex; justify-content:center; align-items:center;">
                            <img src="https://i.ibb.co/S70Gs0cb/Black-White-Minimalist-Initials-Monogram-Jewelry-Logo-2-removebg-preview.png" style="width: 150px; height: 100px;">
                        </div>
                        """, unsafe_allow_html=True)
    st.sidebar.markdown(f"## Welcome, {st.session_state.user['username']}")
    show_logout_ui()
    
    st.sidebar.markdown("### Configuration")
    
    st.sidebar.markdown("### Matching Weights")
    
    skills_weight = st.sidebar.slider("Skills Weight", 0.0, 1.0, 0.35, 0.05)
    experience_weight = st.sidebar.slider("Experience Weight", 0.0, 1.0, 0.25, 0.05)
    education_weight = st.sidebar.slider("Education Weight", 0.0, 1.0, 0.15, 0.05)
    projects_weight = st.sidebar.slider("Projects Weight", 0.0, 1.0, 0.10, 0.05)
    achievements_weight = st.sidebar.slider("Achievements Weight", 0.0, 1.0, 0.10, 0.05)
    
    total_weights = skills_weight + experience_weight + education_weight + projects_weight + achievements_weight
    misc_weight = max(0.0, 1.0 - total_weights)
    
    st.sidebar.markdown("### Algorithm Parameters")
    embedding_weight = st.sidebar.slider("Embedding Similarity Weight", 0.0, 1.0, 0.6, 0.1)
    semantic_weight = st.sidebar.slider("Semantic Matching Weight", 0.0, 1.0, 0.3, 0.1)
    exact_weight = st.sidebar.slider("Exact Matching Weight", 0.0, 1.0, 0.1, 0.1)
    
    match_threshold = st.sidebar.slider("Match Threshold", 0.0, 1.0, 0.6, 0.05)
    recency_weight = st.sidebar.slider("Recency Weight", 0.0, 1.0, 0.7, 0.1)
    seniority_match_weight = st.sidebar.slider("Seniority Match Weight", 0.0, 1.0, 0.8, 0.1)
    
    config = MatchConfig(
        skills_weight=skills_weight,
        experience_weight=experience_weight,
        education_weight=education_weight,
        projects_weight=projects_weight,
        achievements_weight=achievements_weight,
        misc_weight=misc_weight,
        embedding_weight=embedding_weight,
        semantic_weight=semantic_weight,
        exact_weight=exact_weight,
        match_threshold=match_threshold,
        recency_weight=recency_weight,
        seniority_match_weight=seniority_match_weight
    )
    
    tab1, tab2 = st.tabs(["Single Job Analysis", "Batch Processing"])
    
    with tab1:
        st.markdown('<div class="section-header">Single Job Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="sub-header">Job Description</div>', unsafe_allow_html=True)
            jd_option = st.radio("Input method:", ["Text Input", "Upload File"])
            
            jd_text = ""
            if jd_option == "Text Input":
                jd_text = st.text_area("Enter job description:", height=300)
            else:
                jd_file = st.file_uploader("Upload job description:", type=["pdf", "docx", "txt"])
                if jd_file:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        jd_path = save_uploaded_file(jd_file, temp_dir)
                        if jd_path:
                            temp_matcher = CVMatcher()
                            jd_text = temp_matcher.preprocess_document(jd_path)
                            st.success(f"Successfully processed job description: {jd_file.name}")
        
        with col2:
            st.markdown('<div class="sub-header">Candidate CVs</div>', unsafe_allow_html=True)
            cv_files = st.file_uploader("Upload candidate CVs:", type=["pdf", "docx", "txt"], accept_multiple_files=True)
            
        if st.button("Match CVs to Job Description"):
            if not jd_text:
                st.error("Please provide a job description.")
            elif not cv_files:
                st.error("Please upload at least one CV.")
            else:
                with st.spinner("Processing..."):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        global matcher
                        matcher = CVMatcher(config)
                        
                        cv_paths = []
                        for cv_file in cv_files:
                            cv_path = save_uploaded_file(cv_file, temp_dir)
                            if cv_path:
                                cv_paths.append((cv_file.name, cv_path))
                        
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, (cv_name, cv_path) in enumerate(cv_paths):
                            try:
                                cv_text = matcher.preprocess_document(cv_path)
                                if cv_text:
                                    result = matcher.match_cv_to_jd(cv_text, jd_text, cv_name)
                                    results.append(result)
                                
                                progress_bar.progress((i + 1) / len(cv_paths))
                                
                            except Exception as e:
                                st.error(f"Error processing {cv_name}: {str(e)}")
                        
                        if results:
                            st.success(f"Successfully processed {len(results)} CVs!")
                            
                            display_results(results, jd_text)
                        else:
                            st.warning("No results were generated. Please check your files and try again.")
    
    with tab2:
        st.markdown('<div class="section-header">Batch Processing</div>', unsafe_allow_html=True)
        st.markdown("""
        This feature allows you to process multiple CVs against multiple job descriptions. 
        Upload your documents in separate folders and run the analysis.
        """)
        st.info("This feature is coming soon. Please use the Single Job Analysis tab for now.")

def main():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: red;
            margin-bottom: 1rem;
        }
        .section-header {
            font-size: 1.8rem;
            font-weight: bold;
            color: red;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.4rem;
            font-weight: bold;
            color: #555;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .insight-box {
            background-color: #f0f8ff;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #1E88E5;
            margin-bottom: 1rem;
        }
        .highlight {
            background-color: #ffff99;
            padding: 0.2rem;
            border-radius: 0.2rem;
        }
        .score-badge {
            display: inline-block;
            padding: 0.3rem 0.6rem;
            border-radius: 1rem;
            font-weight: bold;
            color: white;
        }
        .auth-form {
            background-color: #f9f9f9;
            padding: 2rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            max-width: 500px;
            margin: 0 auto;
        }
        .stButton>button {
            width: 100%;
        }
        
    </style>
    """, unsafe_allow_html=True)
    
    if 'db_connection' not in st.session_state:
        st.session_state.db_connection = init_db()
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    if st.session_state.authenticated:
        app_main()
    else:
        st.markdown('<div class="main-header">CV to Job Description Matcher</div>', unsafe_allow_html=True)
        st.markdown("Please log in or register to continue")
        
        if 'show_register' not in st.session_state:
            st.session_state.show_register = False
            
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            show_login_page()
        
        with tab2:
            with st.form("register_form_tab"):
                username = st.text_input("Username")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                password_confirm = st.text_input("Confirm Password", type="password")
                submit = st.form_submit_button("Register")
                
                if submit:
                    if username and email and password and password_confirm:
                        if password != password_confirm:
                            st.error("Passwords do not match")
                        elif len(password) < 6:
                            st.error("Password must be at least 6 characters long")
                        else:
                            success, message = register_user(username, email, password)
                            if success:
                                st.success(message)
                                st.session_state.show_register = False
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(message)
                    else:
                        st.error("Please fill in all fields")

matcher = None

if __name__ == "__main__":
    main()