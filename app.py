
import streamlit as st
import pickle
import re
import nltk
import requests


nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

# Text cleaning function
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# Streamlit UI
def main():
    st.title("Resume to Job Category Prediction")
    st.sidebar.title("Options")

    # File upload widget
    uploaded_file = st.sidebar.file_uploader("Upload a resume (in TXT or PDF format)", type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]

        # Map category ID to category name
        category_mapping = {
            0: "Advocate",
            1: "Arts",
            2: "Automation Testing",
            3: "Blockchain",
            4: "Business Analyst",
            5: "Civil Engineer",
            6: "Data Science",
            7: "Database",
            8: "DevOps Engineer",
            9: "DotNet Developer",
            10: "ETL Developer",
            11: "Electrical Engineering",
            12: "HR",
            13: "Hadoop",
            14: "Health and fitness",
            15: "Java Developer",
            16: "Mechanical Engineer",
            17: "Network Security Engineer",
            18: "Operations Manager",
            19: "PMO",
            20: "Python Developer",
            21: "SAP Developer",
            22: "Sales",
            23: "Testing",
            24: "Web Designing",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        # Display results
        st.header("Resume Analysis Results")
        st.subheader("Predicted Job Category:")
        st.success(category_name)

        # LinkedIn Job Search API Integration
        st.header("Job Openings for Predicted Category")
        url = "https://jsearch.p.rapidapi.com/search"

        querystring = {"query": f"{category_name} in India", "page": "3", "num_pages": "1"}

        headers = {
            "X-RapidAPI-Key": "d84fca7d4emshd10e32a65c27c6ep19764ejsn800030a17dc2",
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers, params=querystring)

        data = response.json().get("data", [])

        if data:
            for job in data:
                if job['employer_logo']:
                    st.image(job['employer_logo'])
                st.markdown(f'<p style="font-size: 16px;">{job["job_title"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size: 16px;">Company: {job["employer_name"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size: 14px;">Location: {job["job_city"]}, {job["job_state"]}, {job["job_country"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size: 14px;">Posted Date: {job["job_posted_at_datetime_utc"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<a style="color: #0366d6; font-size: 18px;" href="{job["job_apply_link"]}">click here to apply</a>',unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.warning(f"No jobs found for the predicted category '{category_name}' in India.")


if __name__ == "__main__":
    main()
