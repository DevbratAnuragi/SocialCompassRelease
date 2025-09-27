import numpy as np, json

epoch = "1758660238"  # from your _meta.json
imu  = np.fromfile(f"{epoch}_imu.bin",  dtype="<f4").reshape(800, 6)
mfcc = np.fromfile(f"{epoch}_mfcc.bin", dtype="<f4").reshape(430, 13)

with open(f"{epoch}_meta.json") as f:
    meta = json.load(f)
print(imu, mfcc, meta["epoch_start_s"])
