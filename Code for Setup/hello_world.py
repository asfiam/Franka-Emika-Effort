from omni.isaac.examples.base_sample import BaseSample
# This extension has franka related tasks and controllers as well
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
import numpy as np
from omni.isaac.sensor.scripts.effort_sensor import EffortSensor

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import time
import omni.graph.core as og

rospy.init_node('franka_effort_publisher')
effort_pub = rospy.Publisher('/joint_states_new', JointState, queue_size=10)
joint_state_msg = JointState()
joint_state_msg.name = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]

class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        # Robot specific class that provides extra functionalities
        # such as having gripper and end_effector instances.
        franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka", position=np.array([0.0, 0.0, 0.026])))
        cube = world.scene.add(DynamicCuboid(prim_path="/World/random_cube",
                                            name="fancy_cube",
                                            position=np.array([0.3, 0.3, 0.3]),
                                            scale=np.array([0.115, 0.115, 0.115]),
                                            color=np.array([0, 0, 1.0])))

        self._sensor1 = world.scene.add(EffortSensor(
        prim_path="/World/Fancy_Franka/panda_link0/panda_joint1",
        sensor_period=0.1,
        use_latest_data=False,
        enabled=True))      

        self._sensor2 = world.scene.add(EffortSensor(
        prim_path="/World/Fancy_Franka/panda_link1/panda_joint2",
        sensor_period=0.1,
        use_latest_data=False,
        enabled=True)) 

        self._sensor3 = world.scene.add(EffortSensor(
        prim_path="/World/Fancy_Franka/panda_link2/panda_joint3",
        sensor_period=0.1,
        use_latest_data=False,
        enabled=True)) 

        self._sensor4 = world.scene.add(EffortSensor(
        prim_path="/World/Fancy_Franka/panda_link3/panda_joint4",
        sensor_period=0.1,
        use_latest_data=False,
        enabled=True)) 

        self._sensor5 = world.scene.add(EffortSensor(
        prim_path="/World/Fancy_Franka/panda_link4/panda_joint5",
        sensor_period=0.1,
        use_latest_data=False,
        enabled=True)) 

        self._sensor6 = world.scene.add(EffortSensor(
        prim_path="/World/Fancy_Franka/panda_link5/panda_joint6",
        sensor_period=0.1,
        use_latest_data=False,
        enabled=True)) 

        self._sensor7 = world.scene.add(EffortSensor(
        prim_path="/World/Fancy_Franka/panda_link6/panda_joint7",
        sensor_period=0.1,
        use_latest_data=False,
        enabled=True))                             

        self._create_action_graph()

        return
    
    def get_observations(self):
        #print("inside get observations 1")
        reading1 = self._sensor1.get_sensor_reading(use_latest_data=True)
        reading2 = self._sensor2.get_sensor_reading(use_latest_data=True)
        reading3 = self._sensor3.get_sensor_reading(use_latest_data=True)
        reading4 = self._sensor4.get_sensor_reading(use_latest_data=True)
        reading5 = self._sensor5.get_sensor_reading(use_latest_data=True)
        reading6 = self._sensor6.get_sensor_reading(use_latest_data=True)
        reading7 = self._sensor7.get_sensor_reading(use_latest_data=True)
        #print(f" Effort Sensort Reading at Joint 1: {reading1.value} at time: {reading1.time}")   
        #print(f" Effort Sensort Reading at Joint 2: {reading2.value} at time: {reading2.time}")  
        #print(f" Effort Sensort Reading at Joint 3: {reading3.value} at time: {reading3.time}")  
        #print(f" Effort Sensort Reading at Joint 4: {reading4.value} at time: {reading4.time}")  
        #print(f" Effort Sensort Reading at Joint 5: {reading5.value} at time: {reading5.time}")  
        #print(f" Effort Sensort Reading at Joint 6: {reading6.value} at time: {reading6.time}")  
        #print(f" Effort Sensort Reading at Joint 7: {reading7.value} at time: {reading7.time}")  
        #print("---------------------------------------------------------------------------------------")

        joint_state_msg.effort = [reading1.value, reading2.value, reading3.value, reading4.value, reading5.value, reading6.value, reading7.value ]
        effort_pub.publish(joint_state_msg)

        return

    def _create_action_graph(self):

        og.Controller.edit(
            {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("PublishJointState", "omni.isaac.ros_bridge.ROS1PublishJointState"),
                    ("SubscribeJointState", "omni.isaac.ros_bridge.ROS1SubscribeJointState"),
                    ("ArticulationController", "omni.isaac.core_nodes.IsaacArticulationController"),
                    ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),

                    ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),

                    ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                    ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                    ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                    ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    # Providing path to /panda robot to Articulation Controller node
                    # Providing the robot path is equivalent to setting the targetPrim in Articulation Controller node
                    # ("ArticulationController.inputs:usePath", True),      # if you are using an older version of Isaac Sim, you may need to uncomment this line
                    ("ArticulationController.inputs:robotPath", "/World/Fancy_Franka"),
                    ("PublishJointState.inputs:targetPrim", "/World/Fancy_Franka"),
                ],
            },
        )

    async def setup_post_load(self):
        self._world = self.get_world()
        # The world already called the setup_scene from the task (with first reset of the world)
        # so we can retrieve the task objects
        self._franka = self._world.scene.get_object("fancy_franka")
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        return

    async def setup_post_reset(self):
        #self._controller.reset()
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        # Gets all the tasks observations
        current_observations = self.get_observations()
        #print("here")
        return