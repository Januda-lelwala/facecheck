import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

# Load face detection and recognition models
mtcnn = MTCNN(image_size=160, margin=0)  # Face detection and alignment
resnet = InceptionResnetV1(pretrained='vggface2').eval()  # Face embedding model

# Function to extract face embeddings
def extract_embedding(image_path):
    img = Image.open(image_path).convert('RGB')  # Load image
    face = mtcnn(img)  # Detect and align face
    if face is not None:
        face = face.unsqueeze(0)  # Add batch dimension
        embedding = resnet(face)  # Extract embedding
        return embedding.detach().numpy().flatten()  # Convert to numpy array
    else:
        return None
    
database = {}

# Enroll a person
def enroll_person(image_path, name):
    embedding = extract_embedding(image_path)
    if embedding is not None:
        database[name] = embedding
        print(f"{name} enrolled successfully.")
    else:
        print("No face detected in the image.")

from scipy.spatial.distance import cosine

# Recognize a person
def recognize_person(image_path, threshold=0.6):
    embedding = extract_embedding(image_path)
    if embedding is None:
        return "No face detected."

    best_match = None
    best_score = float('inf')  # Lower cosine distance is better

    for name, stored_embedding in database.items():
        score = cosine(embedding, stored_embedding)
        if score < best_score:
            best_score = score
            best_match = name

    if best_score < threshold:
        return best_match
    else:
        return "Unknown"