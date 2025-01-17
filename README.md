# Disease-Diagnosis-Chat-Assistant
An AI-powered chatbot that predicts diseases based on user symptoms and provides detailed explanations using machine learning and generative AI.

## Features
- Interactive chat interface for symptom input
- Machine learning-based disease prediction
- Detailed disease explanations using Google's Generative AI
- Real-time response system
- User-friendly interface

## Tech Stack
- Backend: FastAPI, Python, Scikit-learn
- Frontend: HTML, CSS, JavaScript
- ML: Joblib, NumPy, Pandas
- AI: Google Generative AI

## API Endpoints
- `GET /`: Welcome message and API information
- `GET /health`: Health check endpoint
- `POST /predict`: Predicts disease based on symptoms
- `POST /explain`: Provides detailed disease explanation
- `GET /docs`: API documentation (Swagger UI)

## Installation & Setup

1. Clone the repository:
`
git clone [your-repository-url]
cd Disease_Diagnosis_Chat_Assistant `
2. Create and activate virtual environment:
`
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
`
3. Install dependencies:
`
pip install -r requirements.txt `

4. Set up environment variables:
Create a .env file and add:
`
GOOGLE_GENERATIVEAI_API_KEY=your_api_key_here
`
5. Run the application:
`
python -m uvicorn api:app --reload
`
6. Open frontend:
Navigate to the frontend directory
Open index.html using a live server


Project Structure
CopyDisease_Diagnosis_Chat_Assistant/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── api.py
├── ml_model.py
├── disease_prediction_model.joblib
├── requirements.txt
└── README.md


