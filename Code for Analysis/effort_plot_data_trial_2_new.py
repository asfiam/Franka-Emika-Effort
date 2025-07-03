import rosbag
import csv
import numpy as np

#bag_files = ['/home/diego/test_1_2024-11-25-11-48-36.bag', '/home/diego/test_2_2024-11-25-11-58-32.bag', '/home/diego/test_3_2024-11-25-12-02-54.bag', '/home/diego/test_4_2024-11-25-12-08-27.bag', '/home/diego/test_5_2024-11-25-12-10-04.bag']
bag_files = ['/home/diego/test_2_2024-11-25-11-58-32.bag'] 
topic_1 = '/panda_teleop/joint_states_rectified' #real
topic_2 = '/joint_states_new' #simulated

#---Joint 3---
joint3_outputfile1 = 'joint3_outoutfile1.csv'
joint3_outputfile2 = 'joint3_outoutfile2.csv'

with open(joint3_outputfile1, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Effort'])

    for bag_file in bag_files:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_1]):
                #print(f"topic value {topic}")
                effort = msg.effort[2]
                time = t.to_sec() 
            
                writer.writerow([time, effort])

print(f"Data saved to {joint3_outputfile1}")

with open(joint3_outputfile2, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Effort'])

    for bag_file in bag_files:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_2]):
                #print(f"topic value {topic}")
                effort = msg.effort[2]
                time = t.to_sec() 
            
                writer.writerow([time, effort])

print(f"Data saved to {joint3_outputfile2}")


#---Joint 4---
joint4_outputfile1 = 'joint4_outoutfile1.csv'
joint4_outputfile2 = 'joint4_outoutfile2.csv'

with open(joint4_outputfile1, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Effort'])

    for bag_file in bag_files:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_1]):
                #print(f"topic value {topic}")
                effort = msg.effort[3]
                time = t.to_sec() 
            
                writer.writerow([time, effort])

print(f"Data saved to {joint4_outputfile1}")

with open(joint4_outputfile2, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Effort'])

    for bag_file in bag_files:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_2]):
                #print(f"topic value {topic}")
                effort = msg.effort[3]
                time = t.to_sec() 
            
                writer.writerow([time, effort])

print(f"Data saved to {joint4_outputfile2}")

#---Joint 5---
joint5_outputfile1 = 'joint5_outoutfile1.csv'
joint5_outputfile2 = 'joint5_outoutfile2.csv'

with open(joint5_outputfile1, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Effort'])

    for bag_file in bag_files:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_1]):
                #print(f"topic value {topic}")
                effort = msg.effort[4]
                time = t.to_sec() 
            
                writer.writerow([time, effort])

print(f"Data saved to {joint5_outputfile1}")

with open(joint5_outputfile2, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Effort'])

    for bag_file in bag_files:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_2]):
                #print(f"topic value {topic}")
                effort = msg.effort[4]
                time = t.to_sec() 
            
                writer.writerow([time, effort])

print(f"Data saved to {joint5_outputfile2}")

#---Joint 6---
joint6_outputfile1 = 'joint6_outoutfile1.csv'
joint6_outputfile2 = 'joint6_outoutfile2.csv'

with open(joint6_outputfile1, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Effort'])

    for bag_file in bag_files:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_1]):
                #print(f"topic value {topic}")
                effort = msg.effort[5]
                time = t.to_sec() 
            
                writer.writerow([time, effort])

print(f"Data saved to {joint6_outputfile1}")

with open(joint6_outputfile2, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Effort'])

    for bag_file in bag_files:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_2]):
                #print(f"topic value {topic}")
                effort = msg.effort[5]
                time = t.to_sec() 
            
                writer.writerow([time, effort])

print(f"Data saved to {joint6_outputfile2}")

#---plots---
import pandas as pd
import matplotlib.pyplot as plt

#---Joint 3---
window_size = 10

data = pd.read_csv(joint4_outputfile1)

time_1 = data['Time']
effort_1 = data['Effort']
print(f"Joint 3 effort 1 value {effort_1}")

data = pd.read_csv(joint4_outputfile2)

time_2 = data['Time']
#effort_2 = data['Effort'].rolling(window = window_size).mean()
effort_2 = data['Effort']
#effort_2 = data['Effort']
print(f"Joint 3 effort 2 value {effort_2}")

#transforming simulated effort values
A = np.stack([effort_2[:2646], np.ones(len(effort_2[:2646]))]).T
print("i am here")
a, b = np.linalg.lstsq(A, effort_1[:2646], rcond=None)[0]
print("i am here 1")

effort_adjusted = a*effort_2 + b

print(effort_1[:2646].shape)
print(effort_2[:2646].shape)
print(effort_adjusted.shape)

plt.figure(figsize=(10, 6))
plt.plot(time_1, effort_1, label=topic_1, color='green', linewidth=1)
plt.plot(time_2, effort_2, label=topic_2, color='blue', linewidth=1)
plt.plot(time_2[:2646], effort_adjusted[:2646], label="Transformed Effort Readings", color='pink', linewidth=1)

plt.title('Effort vs Time (Joint 3)', fontsize=16)
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Effort', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)

plt.show()

#---Joint 4---
data = pd.read_csv(joint3_outputfile1)

time_1 = data['Time']
effort_1 = data['Effort']
print(f"oint 4 effort 1 value {effort_1}")

data = pd.read_csv(joint3_outputfile2)

time_2 = data['Time']
effort_2 = data['Effort']
#effort_2 = data['Effort']
print(f"oint 4 effort 2 value {effort_2}")

plt.figure(figsize=(10, 6))
plt.plot(time_1, effort_1, label=topic_1, color='green', linewidth=1)
plt.plot(time_2, effort_2, label=topic_2, color='blue', linewidth=1)


plt.title('Effort vs Time (Joint 4)', fontsize=16)
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Effort', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)

#plt.show()

#---Joint 5---
window_size = 10

data = pd.read_csv(joint5_outputfile1)

time_1 = data['Time']
effort_1 = data['Effort']
print(f"Joint 5 effort 1 value {effort_1}")

data = pd.read_csv(joint5_outputfile2)

time_2 = data['Time']
effort_2 = data['Effort'].rolling(window = window_size).mean()
#effort_2 = data['Effort']
print(f"Joint 5 effort 2 value {effort_2}")

plt.figure(figsize=(10, 6))
plt.plot(time_1, effort_1, label=topic_1, color='green', linewidth=1)
plt.plot(time_2, effort_2, label=topic_2, color='blue', linewidth=1)


plt.title('Effort vs Time (Joint 5)', fontsize=16)
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Effort', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)

#plt.show()

#---Joint 6---
window_size = 10

data = pd.read_csv(joint6_outputfile1)

time_1 = data['Time']
effort_1 = data['Effort']
print(f"Joint 6 effort 1 value {effort_1}")

data = pd.read_csv(joint6_outputfile2)

time_2 = data['Time']
effort_2 = data['Effort'].rolling(window = window_size).mean()
#effort_2 = data['Effort']
print(f"Joint 6 effort 2 value {effort_2}")

plt.figure(figsize=(10, 6))
plt.plot(time_1, effort_1, label=topic_1, color='green', linewidth=1)
plt.plot(time_2, effort_2, label=topic_2, color='blue', linewidth=1)


plt.title('Effort vs Time (Joint 6)', fontsize=16)
plt.xlabel('Time (seconds)', fontsize=14)
plt.ylabel('Effort', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)

#plt.show()

