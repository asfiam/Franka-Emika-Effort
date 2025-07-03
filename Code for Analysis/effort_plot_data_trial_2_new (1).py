import rosbag
import csv
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

#bag_files = ['/home/diego/test_1_2024-11-25-11-48-36.bag', '/home/diego/test_2_2024-11-25-11-58-32.bag', '/home/diego/test_3_2024-11-25-12-02-54.bag', '/home/diego/test_4_2024-11-25-12-08-27.bag', '/home/diego/test_5_2024-11-25-12-10-04.bag']
bag_files = ['/home/asfia/Internship PANDA ARM/Rosbags/test_2_2024-11-25-11-58-32.bag'] #case 1
#bag_files = ['/home/asfia/Internship PANDA ARM/Rosbags/test_15_2024-11-25-18-10-18.bag'] #case 2
topic_real = '/panda_teleop/joint_states_rectified' #real
topic_sim = '/joint_states_new' #simulated

#---Joint 3---
joint3_outputfile1 = 'joint3_outoutfile1.csv'
joint3_outputfile2 = 'joint3_outoutfile2.csv'

with open(joint3_outputfile1, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Effort'])

    for bag_file in bag_files:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_real]):
                #print(f"topic value {topic}")
                effort = msg.effort[2]
                time = t.to_sec() 
            
                writer.writerow([time, effort])

df = pd.read_csv('/home/asfia/Internship PANDA ARM/joint3_outoutfile1.csv')
num_rows = len(df)
print(f"Data saved to Real values {joint3_outputfile1} with length {num_rows} ")

with open(joint3_outputfile2, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Effort'])

    for bag_file in bag_files:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_sim]):
                #print(f"topic value {topic}")
                effort = msg.effort[2]
                time = t.to_sec() 
            
                writer.writerow([time, effort])

df = pd.read_csv('/home/asfia/Internship PANDA ARM/joint3_outoutfile2.csv')
num_rows = len(df)
print(f"Data saved to Simulated values {joint3_outputfile2} with length {num_rows} ")

#---Joint 4---
joint4_outputfile1 = 'joint4_outoutfile1.csv'
joint4_outputfile2 = 'joint4_outoutfile2.csv'

with open(joint4_outputfile1, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Effort'])

    for bag_file in bag_files:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_real]):
                #print(f"topic value {topic}")
                effort = msg.effort[3]
                time = t.to_sec() 
            
                writer.writerow([time, effort])

df = pd.read_csv('/home/asfia/Internship PANDA ARM/joint4_outoutfile1.csv')
num_rows = len(df)
print(f"Data saved to Real values {joint4_outputfile1} with length {num_rows} ")

with open(joint4_outputfile2, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Effort'])

    for bag_file in bag_files:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_sim]):
                #print(f"topic value {topic}")
                effort = msg.effort[3]
                time = t.to_sec() 
            
                writer.writerow([time, effort])

df = pd.read_csv('/home/asfia/Internship PANDA ARM/joint4_outoutfile2.csv')
num_rows = len(df)
print(f"Data saved to Simulated values {joint4_outputfile2} with length {num_rows} ")

#---Joint 5---
joint5_outputfile1 = 'joint5_outoutfile1.csv'
joint5_outputfile2 = 'joint5_outoutfile2.csv'

with open(joint5_outputfile1, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Effort'])

    for bag_file in bag_files:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_real]):
                #print(f"topic value {topic}")
                effort = msg.effort[4]
                time = t.to_sec() 
            
                writer.writerow([time, effort])

df = pd.read_csv('/home/asfia/Internship PANDA ARM/joint5_outoutfile1.csv')
num_rows = len(df)
print(f"Data saved to Real values {joint5_outputfile1} with length {num_rows} ")

with open(joint5_outputfile2, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Effort'])

    for bag_file in bag_files:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_sim]):
                #print(f"topic value {topic}")
                effort = msg.effort[4]
                time = t.to_sec() 
            
                writer.writerow([time, effort])

df = pd.read_csv('/home/asfia/Internship PANDA ARM/joint5_outoutfile2.csv')
num_rows = len(df)
print(f"Data saved to Simulated values {joint5_outputfile2} with length {num_rows} ")

#---Joint 6---
joint6_outputfile1 = 'joint6_outoutfile1.csv'
joint6_outputfile2 = 'joint6_outoutfile2.csv'

with open(joint6_outputfile1, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Effort'])

    for bag_file in bag_files:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_real]):
                #print(f"topic value {topic}")
                effort = msg.effort[5]
                time = t.to_sec() 
            
                writer.writerow([time, effort])

df = pd.read_csv('/home/asfia/Internship PANDA ARM/joint6_outoutfile1.csv')
num_rows = len(df)
print(f"Data saved to Real values {joint6_outputfile1} with length {num_rows} ")

with open(joint6_outputfile2, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Effort'])

    for bag_file in bag_files:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_sim]):
                #print(f"topic value {topic}")
                effort = msg.effort[5]
                time = t.to_sec() 
            
                writer.writerow([time, effort])

df = pd.read_csv('/home/asfia/Internship PANDA ARM/joint6_outoutfile2.csv')
num_rows = len(df)
print(f"Data saved to Simulated values {joint6_outputfile2} with length {num_rows} ")


#-------------------------All JOINTS----------------------
import pandas as pd
import matplotlib.pyplot as plt
output_files = ([joint3_outputfile1,joint3_outputfile2,3], [joint4_outputfile1,joint4_outputfile2,4], [joint5_outputfile1,joint5_outputfile2,5], [joint6_outputfile1,joint6_outputfile2,6])

rmse_real_sim = {}
rmse_real_adj = {}

for real_file, sim_file, joint in output_files:
    
    i = 0
    data = pd.read_csv(real_file)
    time_real = data['Time']
    effort_real = data['Effort']

    data = pd.read_csv(sim_file)
    time_sim = data['Time']
    effort_sim = data['Effort']

    time_real = np.array(time_real)
    effort_real = np.array(effort_real)
    time_sim = np.array(time_sim)
    effort_sim = np.array(effort_sim)

    effort_sim_interp = np.interp(time_real, time_sim, effort_sim)

    min_len = min(len(time_real), len(effort_sim_interp))

    degree = 2
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(effort_sim_interp.reshape(-1,1), effort_real)
    effort_sim_adjusted = model.predict(effort_sim_interp.reshape(-1,1))

    rmse_real_sim[joint] = np.sqrt(mean_squared_error(effort_real, effort_sim_interp))
    print(f"RMSE (Real vs. Simulatd) for joint {joint}: {rmse_real_sim[joint]}")

    rmse_real_adj[joint] = np.sqrt(mean_squared_error(effort_real, effort_sim_adjusted))
    print(f"RMSE (Real vs. Adjusted) for joint {joint}: {rmse_real_adj[joint]}")

    #plots without transformed values
    plt.figure(figsize=(10, 6))
    plt.plot(time_real, effort_real, label=f"Real Effort Values ({topic_real})", color='green', linewidth=1)
    plt.plot(time_real, effort_sim_interp, label=f"Simulated Effort Values ({topic_sim})", color='blue', linewidth=1)
    #plt.plot(time_real, effort_sim_adjusted, label = "Tranformed Effort Values", color = "red", linewidth =1, linestyle = 'dotted')

    #plt.plot(time_2_trimmed, effort_adjusted, label="Transformed Effort Readings", color='red', linewidth=1)

    plt.title(f"Case 1: Effort vs Time for Joint {joint}", fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Effort', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)

    plt.savefig(f"/home/asfia/Internship PANDA ARM/Plots/Effort Vs. Time Joint {joint}.png", dpi=300)
    #plt.show()

    #plots with transformed values
    plt.figure(figsize=(10, 6))
    plt.plot(time_real, effort_real, label=f"Real Effort Values ({topic_real})", color='green', linewidth=1)
    plt.plot(time_real, effort_sim_interp, label=f"Simulated Effort Values ({topic_sim})", color='blue', linewidth=1)
    plt.plot(time_real, effort_sim_adjusted, label = "Tranformed Effort Values", color = "red", linewidth =1, linestyle = 'dotted')

    #plt.plot(time_2_trimmed, effort_adjusted, label="Transformed Effort Readings", color='red', linewidth=1)

    plt.title(f"Case 1: Effort vs Time for Joint {joint}", fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Effort', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)

    plt.savefig(f"/home/asfia/Internship PANDA ARM/Plots/Effort Vs. Time (Transformed) Joint {joint}.png", dpi=300)
    #plt.show()
    i = i+1

#plot for RMSE Vs. Joints
joint = list(rmse_real_sim.keys())
rmse_real_sim = list(rmse_real_sim.values())
rmse_real_adj = list(rmse_real_adj.values())

avg_rmse_real_sim = sum(rmse_real_sim)/len(rmse_real_sim)
avg_rmse_real_adj = sum(rmse_real_adj)/len(rmse_real_adj)
print(f"Avg RMSE (Real vs. Sim): {avg_rmse_real_sim}, Avg RMSE (Real vs. Adj): {avg_rmse_real_adj}")

x = np.arange(len(joint))  # X-axis positions for the joints
bar_width = 0.35  # Width of each bar

plt.figure(figsize=(8, 5))

bars1 = plt.bar(x - bar_width/2, rmse_real_sim, width=bar_width, color='skyblue', label="Real vs. Simulated")
bars2 = plt.bar(x + bar_width/2, rmse_real_adj, width=bar_width, color='orange', label="Real vs. Transformed")

plt.xlabel("Joints")
plt.ylabel("RMSE")
plt.title("Case 1: RMSE vs. Joints (Real vs. Simulated Effort vs. Transformed Effort)")
plt.xticks(x, joint)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar, value in zip(bars1, rmse_real_sim):
    plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, f"{value:.3f}", 
             ha='center', va='bottom', fontsize=10, color='black')
    
for bar, value in zip(bars2, rmse_real_adj):
    plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, f"{value:.3f}", 
             ha='center', va='bottom', fontsize=10, color='black')

plt.legend()
plt.savefig(f"/home/asfia/Internship PANDA ARM/Plots/RMSE_vs_Joints.png", dpi=300)
plt.show()