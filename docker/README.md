## Install Docker

### 1) For computers without a Nvidia GPU

Install Docker and grant user permission.
```
curl https://get.docker.com | sh && sudo systemctl --now enable docker
sudo usermod -aG docker ${USER}
```
Make sure to **restart the computer**, then install additional packages.
```
sudo apt update && sudo apt install mesa-utils libgl1-mesa-dri libgl1 libglx-mesa0
```

### 2) For computers with Nvidia GPUs

Install Docker and grant user permission.
```
curl https://get.docker.com | sh && sudo systemctl --now enable docker
sudo usermod -aG docker ${USER}
```
Make sure to **restart the computer**, then install Nvidia Container Toolkit (Nvidia GPU Driver
should be installed already).

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
```
sudo apt update && sudo apt install nvidia-container-toolkit
```
Configure Docker runtime and restart Docker daemon.
```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Run Docker Containers

Clone the workshop repository to your home folder.
```
cd /path/to/desired/directory
git clone --recurse-submodules git@github.com:Yuxin916/CMU-VLN-Challenge-2026.git && cd ./CMU-VLN-Challenge-2026
```
Allow remote X connection.
```
xhost +
```
Go inside the docker folder.
```
cd docker
```
For computers **without a Nvidia GPU**, build and start both containers.
```bash
docker compose -f compose.yml up --build -d
```
For computers **with Nvidia GPUs**, use the GPU compose file instead.
```bash
docker compose -f compose_gpu.yml up --build -d
```
This starts two containers:
- `iros2026_system` — the base autonomy system (simulator + autonomy stack)
- `iros2026_ai_module` — the AI module development environment with the updated `dummy_vlm` built in

## Launch base autonomy system

Access the system container.
```bash
docker exec -it iros2026_system bash
```
Inside the container, launch the base autonomy system.
```bash
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
```

## Launch the AI module

Access the AI module container.
```bash
docker exec -it iros2026_ai_module bash
```
**Set the RMW implementation explicitly.** The image exports it from `~/.bashrc`,
which `bash -lc` does not source; without it the default FastDDS transport
discovers topics across containers but delivers no data.
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch dummy_vlm dummy_vlm.launch
```
The module listens on `/challenge_question` (std_msgs/String), explores the
scene, and answers according to the question type:
- **Object reference** (`Find the ...`): publishes a bounding box on `/selected_object_marker` and drives to its centre via `/way_point_with_heading`.
- **Numerical** (`How many ...` / `Count the number of ...`): publishes the count on `/numerical_response`.
- **Instruction following**: publishes an ordered sequence of waypoints on `/way_point_with_heading`, one per leg of the command.

See [../ai_module/README.md](../ai_module/README.md) for the design.

To send example questions, open a new terminal, exec into either container, and use `ros2 topic pub`. Both containers share the same ROS2 network via `--network=host`.

Object reference question (triggers marker + object waypoint):
```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'Find teal pillow on the sofa farthest from the window'}"
```
Numerical question (triggers a counted integer response):
```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'How many books are on the sofa'}"
```
Navigation question (triggers sequential waypoint following):
```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'Go to the potted plant closest to the pyramid candle holder and stop at the vase between the TV and the door.'}"
```

You should see the vehicle following waypoints and the selected object being highlighted in RVIZ.

## Rebuild after changing the model

The module lives under `ai_module/src/dummy_vlm/dummy_vlm/` (Python, ament_python).
After editing it, rebuild the image:
```bash
cd docker
docker compose -f compose.yml up --build -d      # or compose_gpu.yml
```
Any replacement must subscribe to `/challenge_question` (std_msgs/msg/String) and publish on the response topic matching the question type.
