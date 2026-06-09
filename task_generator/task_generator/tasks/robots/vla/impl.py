import asyncio
import io
import math
import time
import numpy as np
import requests

from PIL import Image 
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import sensor_msgs.msg

from arena_rclpy_mixins.Time import Time
from task_generator.shared import Orientation, Pose, Position
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.request import GoToPhase, TaskRequest

from arena_robots.Sensor import SensorType

#server 
_VLA_BASE = "http://127.0.0.1:8000"
_VLA_SERVER = _VLA_BASE + "/act"
_VLA_HEALTH = _VLA_BASE + "/health"

_INSTRUCTION = "go to a corner and stay there" #extend: later add arg or read from file

_INFERENCE_INTERVAL = 2 #done is called 500ms by default in interval, overwrite with 2s

_WAYPOINT_PROXIMITY_THRESHOLD = 0.2 #meter, threshhold before giving control to nav2
_MAX_INVALID_STREAK = 3 #consecutive skipping invalid waypoint before forcing proximity handoff

class TM_VLA(TM_Robots):

    _latest_images:dict
    _image_subs:dict

    _timeouts:dict
    _started:dict
    _last_attempt:dict #inference interval

    _near_goal:dict
    _invalid_streak:dict

    _inference_pending:dict #asyncio

    #---------------MAIN LOOP------------------(tm explore inspired)#
    async def reset(self, **kwargs: object) -> None:
            await super().reset(**kwargs)

            self._latest_images = {}
            self._image_subs = []
            self._timeouts = {}
            self._started = {}
            self._last_attempt = {}
            self._near_goal = {}
            self._inference_pending = {}
            self._invalid_streak = {}

            # Random valid spawn positions
            biggest_robot = max((r.safe_distance for r in self._ctx.robots.values()), default=0.5)
            n = len(self._ctx.robots)
            positions = self._ctx.world_manager.get_positions_on_map(n=n, safe_dist=biggest_robot)
            orientations = 2 * math.pi * self.node.conf.General.RNG.value.random(n)
            for (name, robot), pos, ori in zip(self._ctx.robots.items(), positions, orientations, strict=False):
                self._start_poses[name] = Pose(pos, Orientation.from_yaw(ori))
            
            #try reach server
            try:
                requests.get(_VLA_HEALTH, timeout=2.0).raise_for_status()
            except Exception:
                self.node.get_logger().error(
                    f"[TM_VLA]: VLA Server not reachable at {_VLA_BASE}"
                )
            
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )

            for name, robot in self._ctx.robots.items():
                self._latest_images[name]=None
                
                self._timeouts[name]=self.node.sim_time
                self._started[name]=False

                topic=self.find_image_topic(robot)
                if topic is None:
                    self.node.get_logger().warn(
                        f"[TM_VLA:{name}] no image sensor in model_params. Skipping {name}"
                    )
                    continue
                sub=self.node.create_subscription(
                    sensor_msgs.msg.Image,
                    topic,
                    lambda msg, n=name: self.robot_image(n,msg),
                    qos
                )
                self._image_subs.append(sub)
                self.node.get_logger().info(
                    f"[TM_VLA]sub to {topic}"
                )
    @property 
    async def done(self)->bool:
        timeout=self.node.conf.Robot.TIMEOUT.value

        for name, robot in self._ctx.robots.items():
            
            if self._near_goal.get(name, False):
                self.node.get_logger().warn(
                    f"[TM_VLA:{name}] proximity mode — waiting for nav2"
                )
                if await robot.is_done: #wait for nav2 to finish instead of activating submit_vla_goal
                    return True
                continue
            
            if (self.node.sim_time.sec - self._timeouts.get(name, Time()).sec) >= timeout:
                self.node.get_logger().warn(f"[TM_VLA:{name}] episode timeout — ending")
                return True
            
            await self.submit_vla_goal(name, robot)
        
        return False
    
    #---------------------help func--------------
    def find_image_topic(self, robot)->str|None:
        image_sensors = [s for s in robot.robot_view.model_params.sensors if s.type==SensorType.IMAGE]
        if not image_sensors:
            return None
        
        preferred = next((s for s in image_sensors if "front" in s.name.lower()), image_sensors[0])
        return str(robot.namespace(preferred.topic.removeprefix("${namespace}/")))
    #-----------------------
    def robot_image(self, robot, msg:sensor_msgs.msg.Image)->None:
        self._latest_images[robot]=msg
    #--------------------------
    def vla_inference(self, image:sensor_msgs.msg.Image)->list[tuple[float,float]]|None:

        #preping the image for sending to server
        arr = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1
        )
        if image.encoding=="bgr8":
            arr=arr[:,:,::-1].copy() #reformat to rgb cuz pil want that
        pil = Image.fromarray(arr).resize((512,512))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG")

        try:
            response=requests.post(
                _VLA_SERVER,
                files={"image": ("frame.jpg", buf.getvalue(), "image/jpeg")},
                data={"instruction":_INSTRUCTION},
                timeout=10.0
            )
            response.raise_for_status()
        except requests.exceptions.RequestException:
            self.node.get_logger().warn(f"[TM_VLA] server unreachable at {_VLA_BASE}, retrying in {_INFERENCE_RETRY_INTERVAL:.0f}s")
            return None

        return [(float(wp[0]), float(wp[1])) for wp in response.json()["waypoints"]]
    
    #----------------------
    async def submit_vla_goal(self, name, robot)->None:
        # Do the following:
        
        # run vla_inference at set _INFERENCE_INTERVAL if nothing changed in done()
        # (ie: if robot.is_done: submit_vla_goal #similar to explore)
        
        # extract (at max 8 waypoints cuz vla output), convert to proper waypoints in arena
        # one after another using to_pose
        # put those in list of GoToPhase for later submit_task(TaskRequest)

        #Filter out waypoints that is out of bound with respect to simulation
        # Handeling nav2 handoff when total distant < Threshhold
        now = time.monotonic()
        if now - self._last_attempt.get(name,0.0)<_INFERENCE_INTERVAL:
            return 
        self._last_attempt[name]=now

        #inference
        image = self._latest_images.get(name)
        current_pose=robot.pose
        if image is None or current_pose is None:
            self.node.get_logger().warn(
                f"No image or pose(most likely img)"
            )
            return
        
        if self._inference_pending.get(name,False):
            return
        self._inference_pending[name]=True
        try:
            waypoints=await asyncio.to_thread(self.vla_inference,image)
        finally:
            self._inference_pending[name]=False
        if waypoints is None:
            self.node.get_logger().warn(
                f"waypoint empty"
            )
            return
        

        #convert waypoint to proper waypoint and append to list for submission
        phases=[]
        
        for wp in waypoints:
            goal_pose=self.to_pose(current_pose,wp) 
            phases.append(GoToPhase(pose=self._ctx.environment_manager.ezilear(goal_pose)))
            #phases.append(GoToPhase(pose=goal_pose))
           

        #drop wp that is invalid
        phases=[wp for wp in phases if self.is_valid_pose(wp.pose.position.x,wp.pose.position.y)]
        for wp in phases:
            self.node.get_logger().warn(
                #f"[TM_VLA:WP], goal_valid:({wp.pose.position.x:.2f},{wp.pose.position.y:.2f}) "
                f"[TM_VLA:WP], goal_valid:({self._ctx.environment_manager.realize(wp.pose).position.x:.2f},{self._ctx.environment_manager.realize(wp.pose).position.y:.2f}) "
            )
        if not phases: #incase all invalide
            streak = self._invalid_streak.get(name, 0) + 1
            self._invalid_streak[name] = streak
            self.node.get_logger().warn(
                f"[TM_VLA:{name}] all waypoints invalid ({streak}/{_MAX_INVALID_STREAK}) — skipping"
            )
            if streak >= _MAX_INVALID_STREAK:
                self.node.get_logger().warn(
                    f"[TM_VLA:{name}] invalid streak limit reached — forcing proximity handoff"
                )
                self._near_goal[name] = True
            return
        
        # calculate total distance between current pose and last goal pose
        # if les than threshold, handoff to nav2
        phases=[phases[-1]]
        #wps = [current_pose]+[wp.pose for wp in phases]
        wps = [current_pose]+[self._ctx.environment_manager.realize(wp.pose) for wp in phases]
        
        total_dist=sum(
            math.hypot(
                wps[i+1].position.x-wps[i].position.x,
                wps[i+1].position.y-wps[i].position.y
            ) for i in range(len(wps)-1)
        )
        self._near_goal[name]=total_dist<_WAYPOINT_PROXIMITY_THRESHOLD
        if self._near_goal[name]: 
            phases = [phases[-1]]
            self.node.get_logger().warn(
                f"[TM_VLA:{name}] proximity mode — total_dist={total_dist:.2f}m < "
                f"{_WAYPOINT_PROXIMITY_THRESHOLD}m, handing off final goal to nav2"
            )
        
        self.node.get_logger().warn(
            f"[TM_VLA:{name}] {len(phases)} waypoints from "
            f"robot=({current_pose.position.x:.2f},{current_pose.position.y:.2f}) "
            #f"to ({phases[-1].pose.position.x:.2f},{phases[-1].pose.position.y:.2f})"
            f"to ({self._ctx.environment_manager.realize(phases[-1].pose).position.x:.2f},{self._ctx.environment_manager.realize(phases[-1].pose).position.y:.2f})"
        )

        self._invalid_streak[name] = 0
        self._started[name] = True
        self._timeouts[name] = self.node.sim_time
        await robot.submit_task(TaskRequest(phases=phases))
    
    #----------------------------
    def is_valid_pose(self, x:float, y:float)->bool:
        from task_generator.manager.world_manager.utils import WorldOccupancy
        world_map = self._ctx.world_manager.map
        row, col = world_map.tf_pos2grid(Position(x=x, y=y))
        rows, cols = world_map.occupancy.grid.shape
        if not (0 <= row < rows and 0 <= col < cols): 
            return False
        return bool(WorldOccupancy.not_full(world_map.occupancy.grid)[int(row), int(col)])
    
    def to_pose(self, current: Pose, action: tuple[float, float]) -> Pose:
        dx, dy = action
        yaw = current.orientation.to_yaw()
        new_x = current.position.x + dx * math.cos(yaw) - dy * math.sin(yaw)
        new_y = current.position.y + dx * math.sin(yaw) + dy * math.cos(yaw)
        
        move_dx = dx * math.cos(yaw) - dy * math.sin(yaw)
        move_dy = dx * math.sin(yaw) + dy * math.cos(yaw)
        new_yaw = math.atan2(move_dy, move_dx)
        return Pose(Position(new_x, new_y), Orientation.from_yaw(new_yaw))