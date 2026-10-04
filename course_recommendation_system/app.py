import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from course_recommendation_system.model import CourseRecommender

@st.cache_resource
def get_recommender():
    return CourseRecommender()

def main():
    st.set_page_config(page_title='Course Recommendation System', layout='wide')
    st.title('🎓 Course Recommendation System')
    
    rec = get_recommender()
    df = rec.df
    
    st.write(f"📚 Dataset: {len(df)} courses loaded")
    
    # Sidebar inputs
    st.sidebar.header('👤 User Profile')
    skills = st.sidebar.text_area('Skills (comma separated)', value='python, data analysis', height=80)
    interests = st.sidebar.text_input('Interests (comma separated)', value='machine learning, web development')
    career_goals = st.sidebar.text_area('Career Goals', value='Become a data scientist', height=80)
    
    available_domains = [''] + sorted([d for d in df['domain'].dropna().unique().tolist() if d])
    domain = st.sidebar.selectbox('Preferred Domain (optional)', options=available_domains)

    st.sidebar.header('🔍 Filters')
    difficulty = st.sidebar.selectbox('Difficulty Level', options=['', 'Beginner', 'Intermediate', 'Advanced'])
    min_rating = st.sidebar.slider('Minimum Rating', 0.0, 5.0, 0.0, 0.5)
    topk = st.sidebar.slider('Number of Recommendations', 5, 50, 15)
    recommend_btn = st.sidebar.button('✨ Recommend Courses', key='recommend')

    st.sidebar.header('🔎 Search')
    search_query = st.sidebar.text_input('Search by title or description')

    # Main area
    if recommend_btn:
        with st.spinner('🔄 Generating recommendations...'):
            interests_list = [s.strip() for s in interests.split(',') if s.strip()]
            
            # Only apply filters if they're explicitly selected
            filters = {}
            if domain:
                filters['domain'] = domain
            if difficulty:
                filters['difficulty'] = difficulty
            if min_rating > 0:
                filters['min_rating'] = min_rating
            
            st.write(f"**Search Query:** Skills: {skills} | Interests: {interests_list} | Goals: {career_goals}")
            if filters:
                st.write(f"**Filters Applied:** {filters}")
            
            results = rec.recommend(
                skills=skills, 
                interests=interests_list, 
                goals=career_goals, 
                topk=topk, 
                filters=filters if filters else None
            )
            
            # Apply search query
            if search_query and not results.empty:
                mask = (
                    results['title'].fillna('').str.contains(search_query, case=False, na=False) | 
                    results['description'].fillna('').str.contains(search_query, case=False, na=False)
                )
                results = results[mask].reset_index(drop=True)
            
            st.subheader(f'📖 Recommendations ({len(results)} courses found)')
            
            if results.empty:
                st.warning('❌ No recommendations found. Try with different filters or keywords.')
                st.write("**Debug Info:**")
                st.write(f"- Total courses in dataset: {len(df)}")
                st.write(f"- Unique domains: {df['domain'].nunique()}")
                st.write(f"- Courses with ratings >= {min_rating}: {len(df[pd.to_numeric(df['rating'], errors='coerce').fillna(0) >= min_rating])}")
            else:
                # Display results table
                cols_display = ['title', 'difficulty', 'duration', 'rating', 'score']
                cols_display = [c for c in cols_display if c in results.columns]
                
                display_df = results[cols_display].copy()
                display_df['score'] = display_df['score'].apply(lambda x: f"{x:.4f}")
                display_df['rating'] = display_df['rating'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                
                st.dataframe(display_df.reset_index(drop=True), use_container_width=True, hide_index=True)
                
                # Download CSV
                csv = results.to_csv(index=False)
                st.download_button(
                    label='📥 Download Recommendations (CSV)',
                    data=csv,
                    file_name='course_recommendations.csv',
                    mime='text/csv'
                )
                
                # Visualizations
                st.subheader('📊 Analysis')
                
                col1, col2 = st.columns(2)
                with col1:
                    if 'domain' in results.columns and results['domain'].notna().sum() > 0:
                        fig1 = px.histogram(
                            results[results['domain'].notna()],
                            x='domain',
                            title='📂 Course Domain Distribution',
                            labels={'domain': 'Domain', 'count': 'Count'}
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    else:
                        st.info("No domain data available for visualization")
                
                with col2:
                    top_results = results.head(15)
                    fig2 = px.bar(
                        top_results,
                        x='title',
                        y='score',
                        title='🏆 Top Recommendation Scores',
                        labels={'score': 'Similarity Score', 'title': 'Course Title'}
                    )
                    fig2.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Score distribution
                st.subheader('📈 Score Distribution')
                score_dist = pd.DataFrame({'Score': results['score']})
                st.bar_chart(score_dist)
                
                # Course details
                st.subheader('📋 Course Details')
                selected_idx = st.selectbox(
                    'Select a course to view details:',
                    options=range(len(results)),
                    format_func=lambda i: results.iloc[i]['title'][:60]
                )
                
                if selected_idx is not None:
                    course = results.iloc[selected_idx]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Title:** {course['title']}")
                        st.write(f"**Domain:** {course.get('domain', 'N/A')}")
                        st.write(f"**Difficulty:** {course.get('difficulty', 'N/A')}")
                        st.write(f"**Duration:** {course.get('duration', 'N/A')}")
                    with col2:
                        st.write(f"**Rating:** {course.get('rating', 'N/A')}")
                        st.write(f"**Similarity Score:** {course['score']:.4f}")
                        if 'url' in course and pd.notna(course['url']):
                            st.markdown(f"[🔗 View Course]({course['url']})")
                    
                    if 'explanation' in course and course['explanation']:
                        exp = course['explanation']
                        if isinstance(exp, dict) and 'reason' in exp:
                            st.info(f"**Why Recommended:** {exp['reason']}")
    else:
        st.info('👈 Configure your profile on the left sidebar and click "Recommend Courses"')
        st.write("---")
        st.write("**Dataset Overview:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Courses", len(df))
        with col2:
            st.metric("Unique Domains", df['domain'].nunique())
        with col3:
            avg_rating = pd.to_numeric(df['rating'], errors='coerce').mean()
            st.metric("Avg Rating", f"{avg_rating:.2f}")

if __name__ == '__main__':
    main()
