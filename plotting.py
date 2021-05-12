import matplotlib.pyplot as plt
import numpy as np


low_engagement_log_file = "low_eng.txt"
high_engagement_log_file = "high_eng.txt"

low_eng_data = np.loadtxt(low_engagement_log_file, delimiter=',')
high_eng_data = np.loadtxt(high_engagement_log_file, delimiter=',')


def _f1(elbow_angle_diff):
    '''
    Non-linear squashing function for elbow angle diff
        > Modified sigmoid
    '''
    center_point = 30  # Found these values through a small experiment
    factor = 1/2       #
    return 1/(1+np.exp(-(elbow_angle_diff-center_point)*factor))


def _f2(upper_body_movement):
    '''
    Non-linear squashing function for upper body movement
        > Modified sigmoid
    '''
    center_point = 2.5  # Found these values through a small experiment
    factor = 3          #
    return 1/(1+np.exp(-(upper_body_movement-center_point)*factor))



# Plotting elbow angle diff for both high and low engagement
x1 = np.arange(-20, 110, 1)
y1 = _f1(x1)

plt.scatter(low_eng_data[:120,0], np.zeros(low_eng_data[:120,0].shape[0]), color='r', alpha=0.25,  label='During Low Engagement')
plt.scatter(high_eng_data[:,0], np.ones(high_eng_data.shape[0]), color='b', alpha=0.25,  label='During High Engagement')
plt.plot(x1,y1,'g', label='Squashing function f1(a)')

plt.xticks(np.arange(-20, 110, 20))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid()
plt.legend()
plt.xlabel("Max. elbow angle differences, a")
plt.ylabel("Engagement level")
plt.title("Relationship between Elbow Angle Variation and Engagement level")
plt.savefig("plot_f1.png")
plt.show()

# Plotting upper body movement data for both high and low engagement
x2 = np.arange(0, 5, 0.01)
y2 = _f2(x2)

corrected_high_eng_data = high_eng_data[high_eng_data[:,1] < 7.0]

plt.scatter(low_eng_data[:120,1], np.zeros(low_eng_data[:120,1].shape[0]), color='r', alpha=0.25,  label='During Low Engagement')
plt.scatter(corrected_high_eng_data[:,1], np.ones(corrected_high_eng_data.shape[0]), color='b', alpha=0.25,  label='During High Engagement')
plt.plot(x2,y2,'g', label='Squashing function f2(u)')

plt.xticks(np.arange(0, 9, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid()
plt.legend()
plt.xlabel("Avg. Upper Body Movement, u")
plt.ylabel("Engagement level")
plt.title("Relationship between Avg.Upper Body Movement and Engagement level")
plt.savefig("plot_f2.png")
plt.show()
