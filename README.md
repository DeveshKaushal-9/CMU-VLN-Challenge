# CMU Vision-Language Navigation Challenge 2026 — Team Yo Yo 🥈

> **2nd Place — CMU Vision-Language-Navigation Challenge 2026**
>
> Presented at the **IEEE/RSJ International Conference on Intelligent
> Robots and Systems (IROS 2026)** in Pittsburgh, USA.

This repository contains my submission for the **CMU Vision-Language-Navigation
(VLN) Challenge 2026**, an embodied-AI challenge focused on grounding natural
language instructions in 3D environments and executing them using an autonomous
mobile robot.

The system was evaluated first in simulation and subsequently on the CMU
real-robot platform.

## 🏆 Result

**Final Rank: 2nd Place**

The submission advanced from the simulation stage to the real-robot evaluation
and finished **second overall** in the CMU-VLN Challenge 2026.

The approach was presented at the **AI Meets Autonomy: Vision, Language, and
Autonomous Systems Workshop at IROS 2026** in Pittsburgh.

## Overview

The challenge requires an autonomous agent to understand natural-language
queries, perceive an initially unknown environment, reason about spatial
relationships between objects, and generate appropriate robot actions.

The system handles three task categories:

### 1. Numerical Reasoning
Example:

> How many blue chairs are between the table and the wall?

Output:
`std_msgs/msg/Int32` → `/numerical_response`

### 2. Object Referential Grounding
Example:

> Find the orange chair between the table and sink that is closest to the window.

The system identifies the referred object and publishes its 3D bounding-box
marker.

Output:
`visualization_msgs/msg/Marker` → `/selected_object_marker`

### 3. Instruction Following
Example:

> Avoid the path between the two tables and go near the blue trash can near
> the window.

The system converts the instruction into a sequence of navigation waypoints.

Output:
`geometry_msgs/msg/Pose2D` → `/way_point_with_heading`

## System

The challenge platform uses:

- Ubuntu 24.04
- ROS 2 Jazzy
- Unity simulation environments
- 360° RGB camera
- 3D LiDAR
- Robot state estimation and autonomous navigation stack

During real-robot evaluation, the AI module was deployed remotely on the
challenge robot's onboard system.

### Real-Robot Compute

- 16× Intel Core i9 CPU cores
- 32 GB RAM
- NVIDIA RTX 4090
- Docker-based deployment

## Repository Structure

```text
CMU-VLN-Challenge/
├── ai_module/       # Vision-language navigation system
├── docker/          # Docker and runtime configuration
├── figures/         # Challenge/system figures
├── questions/       # Development questions and examples
├── requirements.txt
└── README.md
