from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load ML model
model = pickle.load(open('model/career_model.pkl', 'rb'))

# Career mapping
careers = {
    0: "Software Developer",
    1: "Data Scientist",
    2: "Cybersecurity Analyst",
    3: "AI/ML Engineer",
    4: "Cloud Architect",
    5: "UI/UX Designer"
}

# Upskilling suggestion
suggestions = {
    "Software Developer": ["Learn Python", "Contribute to GitHub", "Build full-stack apps"],
    "Data Scientist": ["Master Pandas & NumPy", "Practice Kaggle", "Learn ML/Stats"],
    "Cybersecurity Analyst": ["Study network security", "Try CTF challenges", "Certifications like CEH"],
    "AI/ML Engineer": ["Deep Learning courses", "TensorFlow & PyTorch", "Research papers"],
    "Cloud Architect": ["AWS/GCP Certification", "Learn Docker & Kubernetes", "Build scalable systems"],
    "UI/UX Designer": ["Figma", "Design Systems", "A/B Testing & Prototyping"]
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    skill = int(request.form['skill'])
    logic = int(request.form['logic'])
    creativity = int(request.form['creativity'])
    teamwork = int(request.form['teamwork'])

    data = np.array([[skill, logic, creativity, teamwork]])
    prediction = model.predict(data)[0]
    career = careers[prediction]
    courses = suggestions[career]

    return render_template('index.html', result=career, courses=courses)

if __name__ == "__main__":
    app.run(debug=True)
