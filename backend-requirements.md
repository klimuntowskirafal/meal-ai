# Project overview
We are building a meal calorie estimation app. For now we are using a single image of a meal as input. This is for simplicity and to get a working prototype.

# Feature Requirements
- On the start of the uvicorn server, we will run a calorie estimation model on the image.
- We will store the result in a database (for now in-memory).
- We will create a simple frontend to view the result (for now just a JSON response).
- Later on we will implement: 
  - Cross-reference with a nutritional database
  - Add user verification/correction capabilities
  - Include error margins in the estimates
  - Add caching for repeated analyses of the same meal

# Relevant documentation
xxxx

# Current File Structure
MEAL.AI/
├── __pycache__/
├── venv/
├── .env
├── .gitignore
├── backend-requirements.md
├── main.py
└── meal1.webp