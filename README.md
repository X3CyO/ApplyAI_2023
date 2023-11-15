# ApplyAI_2023
Creating a computer vision program that analyzes measures of fatigue/sleepiness to optimize studying patterns for students using indicators.


Blinking Metrics are a quantifiable variable that has been extracted using a high-speed camera to identify sleepiness trends, but we currently lack a more commonly available tool for the average person to adequately capture enough frames to identify their own level of sleepiness from.

Because of this, I combined two base models from git-hub together, and heavily modified them to create a program that tracks blinking metrics for live classification of sleepiness using a standardized, low-quality web camera (640x480).

The program works and has an accuracy of ___. The 3 trends identified, used had p-values of ~.08 from a dataset of 57 subjects. The classification models within the ensemble had Mann-Whitney agreement scores of .8, .6, and .4 between the actual data and the trained models.

The application will be used in an undergraduate research study in the near future regarding memory encoding.


Techniques used in Python: 
Computer Vision: Deep learning for distance tracking (MIDAS), Optical Flow to limit false blinks registering from rapid movement, and Face-mesh for blink recognition (media pipe), combined to determine facial orientation (to also limit false blinks).
Analytics: K-means
Statistics: U-test, T-test, and Mann-whitney.
Machine-Learning: Random forest & Deterministic models using interpolation, as an Ensemble.
