import numpy as np

def extract_features(records):
    features = []

    for i in range(len(records)):
        dwell = records[i]["release"] - records[i]["press"]

        if i == 0:
            flight = 0.0
        else:
            flight = records[i]["press"] - records[i-1]["release"]

        features.append([dwell, flight])

    return np.array(features, dtype=np.float32)
