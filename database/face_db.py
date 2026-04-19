# database/face_db.py
import pickle

class FaceDB:
    def __init__(self, path="faces.pkl"):
        self.path = path

    def load(self):
        try:
            return pickle.load(open(self.path, "rb"))
        except:
            return []

    def save(self, data):
        pickle.dump(data, open(self.path, "wb"))