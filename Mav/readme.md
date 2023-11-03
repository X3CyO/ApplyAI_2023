Hey everybody!

Real quick; just need you to install the requirements.txt, then we're off to the races!

Just need to run this in your Terminal: pip install -r requirements.txt


Once everthing is downloaded, if you have multiple cameras, you can run python camera_select.py to see what camera you want to use. The value is added into the analyze_link.py file, otherwise it will automatically use the "0" camera.


then you run python analyze_blink.py from the terminal to run the blink capture


blink captures will be stored in the ESS_Data folder as either Sleepy, or Alert based on your ESS Score.
    our Past research has shown that the particular ranges we've set are the most generally used Sleepy >10, Alert <11

This folder can then be analyzed using pupillometry.py (which I have to migrate over from datacamp)



This will then allow you to learn the significance of the difference between your alert and sleepy ESS scores near the .3 of a second and above cutoff.
    This cutoff was selected from a prior study we'd done to find where there was the highest level of significance compared to all other versions of ESS Alert vs Sleepy, and time cutoffs. The reason for this, is because the criteria for a blink is scrutinized to be from .3 - 1.2 seconds or longer. (which we'd confirmed)



From this, we can then compare the values obtained to the general p-values obtained originally to see if this camera is comperable to the original high speed cameras used in a controlled environment. (Objective 1)

Afterwards, we can then move on to the livestream classification module which will use an ensemble of deterministic models, and a random forests to confirm if you are indeed tired or sleepy. (Objective 2)

The final goal is to use a convolutional neural net to ensure that no hyperparameters are lost, and any user could use this and tune it to their own particular physiology (In theory) (Objective 3)