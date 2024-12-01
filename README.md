# SMSNet
A Transformer-based method to simulate multi-scale soil moisture
Here, we present a Transformer-based SM simulation network (SMSNet) to improve the quality of the widely used Soil Moisture Active Passive (SMAP) SM product by reconstructing and downscaling the pixels, and obtain daily SM products with resolutions of 9-km and 1-km.
The main.py is used to carry out SMSNet model trainning using sample dataset (the decompressed samples_20241011.rar). It could save the best model parameters through iterative training. The saved best model parameters are used for reconstruct and downscale.
The transformer_raw.py is the core althgorithm section of SMSNet.
The config.py is used to initialize parameter configuration of SMSNet.
The datasets.py is used to read sample datasets.
The earlystopping.py is used to check whether the SMSNet is eligible for early termination.
The reconstruct.py takes the best model parameters and five consecutive days of 9-km scale independent variables to realize 9-km scale reconstructing.
The downscale.py takes the best model parameters and five consecutive days 1-km scale independent variables to realize 1-km scale downscaling.
The samples_20241011.rar provides 200 samples as examples. The tiles.rar, DEM_USA_II.rar, HWSD_USA_II.rar, IV_USA_III.rar, MCD11A1_USA_II_ST.rar, MCD12C1_USA_II.rar, and MCD43A4_USA_II_ST.rar are 9-km scale independent variable sample datasets. The five consecutive days of 1-km scale independent variables are too large to be uploaded. Please contact me if you need them.
