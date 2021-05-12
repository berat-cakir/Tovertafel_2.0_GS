Tovertafel 2.0 projects game visualizations onto any standard table. A camera tracks motion of hands and other objects that try to interact with the game projections. The result is an easy to use game device where projections react to player input. Additionally, the system can extract player emotions from facial expressions and observe their body language through additional cameras.

The system is composed of two sub-systems. The rst - Game System - is responsible for the
processing of the game, this includes rendering its environment, processing the inputs of the players,
making the game respond accordingly and adapting the environment of the game based on the
information received from the second sub-system regarding the players' mood and engagement.
The second sub-system - Player State Estimation system or P.S.E. - assesses the players' state
of mind based on mood and engagement measurements obtained through the processing of live
video stream of the players and their interactions.

<h1><b>Requirements:</b></h1><br>
<b><a href="https://www.python.org/downloads/release/python-377/">Python 3.7.7 (64-bit)</a></b><br>
<h1><b>Setup (Windows/Linux):</b></h1><br>
<ol>
    <li>Install dependencies by executing in terminal:
	<b><i>pip install -r requirements.txt</i></b></li>
    <li>Install mosquitto. For Windows detailed instructions follow this link: https://subscription.packtpub.com/book/application_development/9781787287815/1/ch01lvl1sec13/installing-a-mosquitto-broker-on-windows</li>
</ol>
<h1><b>Instructions:</b></h1><br>
<ul>
    <li>Start the MQTT Broker Service</li>
    <li>Run <b><i>main.py</i></b> to start the program</li>
</ul>
