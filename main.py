from Capture.keylogger import KeyLogger
from Capture.feature_extractor import extract_features
from Model.encoder import KeystrokeEncoder
from Auth.enroll import enroll
from Auth.authenticate import authenticate

encoder = KeystrokeEncoder()

print("Type password 3 times (ESC to stop each time)")

samples = []
for _ in range(3):
    kl = KeyLogger()
    records = kl.record()
    feats = extract_features(records)
    samples.append(feats)

template = enroll(encoder, samples)

print("Now authenticate")
kl = KeyLogger()
records = kl.record()
feats = extract_features(records)

dist, ok = authenticate(encoder, template, feats, threshold=0.8)

print("Distance:", dist)
print("Access:", ok)
