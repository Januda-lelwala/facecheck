from flask import Flask, request, jsonify
import numpy as np
from scipy.spatial.distance import cosine
import facecheck
import uvicorn

app = Flask(__name__)
extract_embedding = facecheck.extract_embedding
# Mock database (replace with a real database)
database = {}


# API to enroll a person
@app.route('/enroll', methods=['POST'])
def enroll():
    data = request.json
    name = data['name']
    image_path = data['image_path']  # Replace with actual image upload logic
    embedding = extract_embedding(image_path)
    database[name] = embedding.tolist()  # Store embedding as a list
    return jsonify({"status": "success", "message": f"{name} enrolled."})

# API to recognize a person
@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json
    image_path = data['image_path']  # Replace with actual image upload logic
    embedding = extract_embedding(image_path)
    best_match = None
    best_score = float('inf')

    for name, stored_embedding in database.items():
        score = cosine(embedding, np.array(stored_embedding))
        if score < best_score:
            best_score = score
            best_match = name

    threshold = 0.6
    if best_score < threshold:
        return jsonify({"status": "success", "name": best_match})
    else:
        return jsonify({"status": "success", "name": "Unknown"})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.1", port=5000, log_level="info")
