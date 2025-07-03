#import rospy
import rosbag
import matplotlib.pyplot as plt

def plot_effort_from_rosbags(rosbag_files, topics, save_plots=False):
    """
    Plots effort vs. time data from multiple rosbag files.

    Args:
    - rosbag_files: List of rosbag file paths.
    - topics: List of topic names for each arm.
    - save_plots: Whether to save plots as PNG files.
    """
    for bag_file in rosbag_files:
        print(f"Processing bag file: {bag_file}")
        bag = rosbag.Bag(bag_file)
        
        # Initialize storage for time and effort data for each topic
        data = {topic: {"time": [], "effort": []} for topic in topics}
        
        # Extract data for specified topics
        for topic in topics:
            for topic_name, msg, t in bag.read_messages(topics=topic):
                data[topic]["time"].append(t.to_sec())
                data[topic]["effort"].append(msg.effort)  # Adjust if your effort data is stored differently
        
        bag.close()
        
        # Plot data
        fig, axs = plt.subplots(len(topics), 1, figsize=(10, 6))
        fig.suptitle(f"Effort vs Time: {bag_file}")

        for i, topic in enumerate(topics):
            axs[i].plot(data[topic]["time"], data[topic]["effort"][0], label="Simulated")
            axs[i].plot(data[topic]["time"], data[topic]["effort"][1], label="Real")
            axs[i].set_title(f"Effort for {topic}")
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel("Effort")
            axs[i].legend()
        
        plt.tight_layout()
        
        if save_plots:
            output_file = f"{bag_file}_effort_plot.png"
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
        
        plt.show()

# Example usage:
rosbag_files = ["/home/diego/test_4_2024-11-25-12-08-27.bag", "/home/diego/test_5_2024-11-25-12-10-04.bag"]  # Replace with your bag file paths
topics = ["joint_states_rectified/effort", "joint_states_new/effort"]  # Replace with your topics
plot_effort_from_rosbags(rosbag_files, topics, save_plots=True)
