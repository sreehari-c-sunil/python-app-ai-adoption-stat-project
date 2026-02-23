AI ADOPTION RESEARCH ANALYSIS PLATFORM


Project Title:
Analyzing AI Adoption Among College Students:
Trends, Impact, and Academic Outcomes

--------------------------------------------------------------------

OVERVIEW
This is a beginner-friendly web-based statistical analysis platform
built using Python and Flask. Users can upload survey data in CSV
format and automatically receive descriptive statistics, frequency
tables, visual charts, and statistical test results with plain-
English interpretations and a final research conclusion.

This project combines statistics, research methodology, and web
development into a single academic research tool.

--------------------------------------------------------------------

OBJECTIVE
- Upload and validate structured CSV survey data
- Store responses in a SQLite database
- Perform statistical tests automatically
- Generate visualizations
- Provide research-ready interpretations

--------------------------------------------------------------------

RESEARCH PURPOSE
The platform analyzes AI adoption among college students by studying:

- AI usage and GPA
- Gender and AI adoption
- Major and AI adoption
- AI frequency and academic performance
- Comfort level and GPA
- Career interest and optimism

--------------------------------------------------------------------

PERSONA FOCUS
Student Researcher / Student Developer

Designed for statistics students who want to:
- Practice hypothesis testing
- Understand statistical relationships
- Automate inference writing
- Learn Flask backend development
- Document AI-assisted development

--------------------------------------------------------------------

TECHNOLOGIES USED

Python 3.10+      - Core programming language
Flask             - Web framework
SQLite            - Database storage
Pandas            - Data manipulation
NumPy             - Numerical computation
SciPy             - Statistical tests
Matplotlib        - Chart generation

--------------------------------------------------------------------

STATISTICAL METHODS IMPLEMENTED

1. Descriptive Statistics
2. Frequency Tables
3. Chi-Square Test of Independence
4. Independent Sample T-Test
5. One-Way ANOVA
6. Pearson Correlation
7. Summary p-value comparison chart
8. Automated Final Research Conclusion

--------------------------------------------------------------------

KEY FEATURES

- Strict CSV column validation
- Numeric type enforcement
- Automatic database ID reset
- Auto-generated visual charts
- One-line inference + conclusion per test
- Summary statistical results dashboard
- Sample data generator (100 records)

--------------------------------------------------------------------

REQUIRED CSV FORMAT

The CSV file must contain exactly these columns:

age,gender,major,academic_year,ai_usage,ai_frequency,
comfort_level,gpa,career_interest,ethical_concern,
job_replacement_fear,optimism_score

Example row:
21,Female,Computer Science,Junior,Yes,Daily,4,3.7,5,3,2,4

--------------------------------------------------------------------

HOW TO RUN LOCALLY

1. Install Python 3.10+
   python --version

2. Create virtual environment
   python -m venv venv
   venv\Scripts\activate     (Windows)
   source venv/bin/activate  (Mac/Linux)

3. Install dependencies
   pip install -r requirements.txt

4. Run the application
   python app.py

5. Open in browser
   http://localhost:5000

--------------------------------------------------------------------

PROJECT STRUCTURE

```
ai_research_platform/
│
├── app.py               # Main Flask application (routes, logic, statistics)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
│
├── static/              # Static files (CSS, generated charts)
│   └── charts/          # Auto-generated statistical charts
│
└── templates/           # HTML templates
    ├── layout.html      # Base template (navbar + footer)
    ├── index.html       # Home dashboard
    ├── upload.html      # CSV upload page
    ├── analysis.html    # Statistical results page
    ├── view_data.html   # Data preview page
    ├── learn.html       # Educational explanations
    └── about.html       # Project description page
```

LIMITATIONS

- Cross-sectional survey data (no causation claims)
- Results depend on data quality
- Small sample sizes reduce reliability
- Self-reported data may contain bias
- Statistical assumptions must be understood

--------------------------------------------------------------------

EDUCATIONAL PURPOSE

This project is intended for:
- Academic research practice
- Internship documentation
- Statistics learning
- Flask + Data Science portfolio demonstration

--------------------------------------------------------------------

LICENSE
Educational / Academic Use Only
====================================================================#
