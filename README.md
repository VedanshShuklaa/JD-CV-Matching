# JD-CV-Matching
A model that matches a given Job Description with multiple CVs to pick out the best candidates

CV Matcher is an AI-powered tool designed to match resumes (CVs) with job descriptions using advanced Natural Language Processing (NLP) techniques. It evaluates candidates based on skills, experience, education, projects, and achievements, providing a comprehensive score to help recruiters and hiring managers identify the best-fit candidates.

---

## **Features**
- **Skill Extraction**: Identifies and matches technical and soft skills from resumes and job descriptions.
- **Experience Matching**: Evaluates years of experience and seniority levels.
- **Education Matching**: Recognizes academic qualifications and degrees.
- **Semantic Matching**: Uses DistilBERT embeddings to compare the semantic similarity between resumes and job descriptions.
- **Keyword Matching**: Matches exact keywords and their synonyms.
- **Seniority Detection**: Detects seniority levels (entry, mid, senior, management) from text.
- **Skill Depth Analysis**: Measures the depth of expertise for each skill mentioned.
- **Batch Processing**: Processes multiple resumes in bulk and generates a ranked list of candidates.

---

## **How It Works**
1. **Preprocessing**: Extracts text from PDF, DOCX, or plain text files.
2. **Section Parsing**: Identifies and extracts key sections like Skills, Experience, Education, etc.
3. **Skill Extraction**: Uses regex patterns and spaCy's NLP pipeline to extract skills.
4. **Embedding Generation**: Generates sentence embeddings using DistilBERT.
5. **Matching Algorithm**: Combines keyword matching, semantic similarity, and embedding-based matching to compute scores.
6. **Ranking**: Ranks candidates based on their total match score.

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cv-matcher.git
   cd cv-matcher
   ```
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download Spacy's English Model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
---

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
    ```
2. Open the app in your browser (usually at http://localhost:8501).

3. Log in or register to access the tool.

4. Upload a job description and candidate CVs.

5. Adjust matching weights and parameters in the sidebar.

6. View the results, including:

  -Overall match scores.

  -Detailed analysis of each candidate.

  -Skills comparison across candidates.

---

## Dependencies

The project relies on the following Python libraries:

- **`streamlit`**: For building the interactive web interface.
- **`transformers`**: For using DistilBERT embeddings.
- **`spacy`**: For natural language processing tasks like entity recognition.
- **`numpy` and `torch`**: For handling embeddings and numerical computations.
- **`scikit-learn`**: For calculating cosine similarity between embeddings.
- **`PyPDF2` and `python-docx`**: For parsing PDF and DOCX files.
- **`pandas`**: For data manipulation and analysis.
- **`matplotlib` and `seaborn`**: For creating visualizations like bar charts and heatmaps.
- **`sqlite3`**: For user authentication and data storage.

To install all dependencies, run:
```bash
pip install -r requirements.txt
```
