
<h1 align="center">AI-Based Placement Readiness & Resume Analysis System</h1>

🚀 A Machine Learning-powered web application that predicts student placement readiness and provides personalized skill development recommendations. The system also includes a Resume Analyzer with ATS scoring and role-based evaluation for Software Developer (SDE) and Data Scientist roles.

# ✨ Features
🎯 Placement Readiness Prediction
- Predicts placement readiness probability using Machine Learning
- Uses academic and skill-based inputs:
- CGPA
- Coding Skills
- DSA Practice
- Projects
- Internship Experience
- Communication Skills
- Certifications
- Mock Interview Score

📊 Performance Visualization
```text
- ROC Curve visualization
- Probability-based prediction output
- Feature impact analysis
```
#🛠 Skill Development Roadmap

Provides personalized recommendations based on weak areas:

- Coding improvement resources
- DSA roadmap
- Communication skill guidance
- Internship suggestions
- Project recommendations

Includes:

- 📺 YouTube learning resources
- 📄 Downloadable PDF notes

#📄 Resume Analyzer & ATS Score
- Upload resume in PDF format
- Extracts resume text using PyPDF2
- Calculates ATS score
- Identifies missing skills
- Provides role-based recommendations

Supported Roles:

- Software Developer (SDE)
- Data Scientist

#🧠 Machine Learning Models Used
| Model	| Purpose |
|---|---|
| Logistic Regression |	Main prediction model |
| Decision Tree |	Model comparison |
| StandardScaler	| Feature scaling |


🛠 Tech Stack
```text
Python
Streamlit
Scikit-learn
Pandas
NumPy
Matplotlib
PyPDF2
```

#Folder Structure
```text
Placement_Readiness_Predictor/
│
├── app.py
├── train.py
├── placement_data.csv
├── placement_lr_model.pkl
├── placement_dt_model.pkl
├── scaler.pkl
├── requirements.txt
│
├── resources/
│   ├── communication_notes.pdf
│   ├── python basic programes.pdf
│   └── DSA complete Cheatsheet.pdf
│
└── README.md
```
🚀 Installation & Setup
#1️⃣ Clone Repository
```text
git clone https://github.com/your-username/your-repo.git
cd your-repo
```
#2️⃣ Install Dependencies
```text
pip install -r requirements.txt
```
#3️⃣ Run Application
```text
streamlit run app.py
```

🌐 Deployment

The project is deployed using Render.

#Live Demo
```text
👉 https://placement-readiness-predictor.onrender.com/
```

#📈 Model Performance
```text
Accuracy: ~80–85%
ROC Curve based evaluation
Probability-based classification system
```
#🎯 Future Improvements
- NLP-based advanced resume analysis
- AI Interview Simulator
- Company-specific placement prediction
- Chatbot-based career guidance
- Real-time student analytics dashboard

👨‍💻 Author

#Akshith
```text
GitHub: https://github.com/AvunuriAkshith
LinkedIn: https://www.linkedin.com/in/avunuriakshith
```

⭐ If you like this project

Give this repository a ⭐ on GitHub!
