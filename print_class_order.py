import pickle

with open('models/target_encoder.pkl', 'rb') as f:
    pt_encoder = pickle.load(f)

print('Class order:', list(pt_encoder.classes_)) 