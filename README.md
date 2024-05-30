# RobotLab
by Zhiao Huang

This is just a tutorial draft. If you find any mistakes or have any suggestions on this tutorial, please do not hesitate to contact me.

The RobotLab library is used by me to support sim2real experiments for mobile robots. It is a temporary library and will be potentially replaced by or integrated with something in ManiSkill in the future.
The major simulation part of the code resides in [robotics/sim](robotics/sim) and you can ignore other parts that enable ROS integration for real-world robot control.


## Installation
You first need to install [Blazar](https://github.com/HillbotAI/Blazar) before installing this package. Follow [docker](https://github.com/HillbotAI/Blazar/blob/main/docker/Dockerfile) to install the dependencies. We use SAPIEN3. The latest SAPIEN can be installed [here](https://github.com/haosulab/SAPIEN/releases). 
You can use the following command to install the latest version of SAPIEN.
```bash
python3 docker/install_sapien.py
```
However, if you meet Vulkan error in certain servers and do not need GPU, you may install an older [version](https://github.com/haosulab/SAPIEN/actions/runs/7205976971/artifacts/1114116919).

If you have found any assets missing, please message me.


## Simulator
The main interface is the `Simulator` class in [robotics/sim/simulator.py](robotics/sim/simulator.py). It builds SAPIEN scenes and provides utilities to  
- create SAPIEN scenes and corresponding physical systems.
- handle CPU and GPU simulation.
- load multiple user-defined `entities`.
- provide interfaces for creating and running a SAPIEN viewer.

Note that I put most sapien-related code in [robotics/sim/simulator_base.py](robotics/sim/simulator_base.py) to make the code clearer and you can use the SimulatorBase class alone.

### Entity

A core concept in the simulator is `Entity`, which is different from the `Entity` in SAPIEN. It allows user to add a level of abstraction over the original SAPIEN's objects and customize their behaviors. Right now, one can simply consider an `Entity` object as a config to create and organize the SAPIEN objects.

 Here are some simple aspects of an `Entity`:
- In the simulator, all the entities form a tree structure, where the root entity is the scene, or a dictionary called `elements`. One can traverse and look up any entity using this dictionary.
- An entity can have sub-entities, which are stored in the `__children__` attribute. However, this feature is seldom used, and one can ignore it right now.
- An entity must provide two functions: `_load` and `_get_sapien_entities` so that
    - The simulator will call `_load` for all entities in the entity tree when calling `simulator.reset()`. The `_load` function creates the SAPIEN objects and adds them to the scene.
    - It then calls `_get_sapien_entities` to get all the sapien objects in the entity tree. This is used for mesh generation, GPU parallelization and other purposes.
- The simulator can search and `find` an element within an entity. Although this feature is not widely used.

### Engine 

We use the `Engine` interface to unify the code for GPU and CPU. It provides common functions like creating scenes, setting and getting object pose and robot state.
 
To use the GPU simulator, one needs to set the `GPUEngineConfig`, and set the number of scenes. One needs to replace all commands like `actor.set_pose` with `engine.set_pose(actor, pose)` and `actor.get_pose` with `engine.get_pose(actor)`. To enable parallelization, we require all dynamics objects, articulations and cameras to be included in an `Entity` object and its `_get_sapien_entities`. Please see [examples](robotics/sim/tester/test_mycobot_gpu.py) for details.

[gpu_engine.py](robotics/sim/engines/gpu_engine.py) is the code for GPU simulation. The engine does the following things
- it maintains a `Record` class for each articulation object or normal SAPIEN entity. Record objects have indexes of the reference objects in the GPU memory. 
- every write operator to SAPIEN objects will be recorded in the `Record` object first. 
- every read operator will fetch the data from the GPU if the data is not up-to-date. 
- when calling `step`, the engine will synchronize the updates between CPU and GPU.
- however, before `simulator.reset` and the GPU initialization, one can only write attributes and read the attributes written before. 


The multi-scene parallelization is implemented as follows:
- The simulator creates multiple scenes and `_load`s the same set of entities into every scene through a simple for-loop.
- Then, it calls `_get_sapien_entities` of the element tree to get all the entities in a scene. We assume all scenes have the same number of SAPIEN objects and the same order of the objects. For example, an object `obj0`, a box, will exist at every scene after `sim.reset`.
- After `sim.reset`, the simulator initializes GPU memory using the objects created in all scenes. The same objects (like `obj0`) will point to the same record and the record contains a batch of objects that point to it.
- Now, all engine operators to original SAPIEN objects will be applied to the record first, and then be broadcasted to all objects in the batch associated with the record. For example, `engine.set_pose(obj0, pose)` will be broadcasted to all `obj0` in all scenes.
- **NOTE** The broadcast will not happen within the `Entity._load` function, which is before `sim.reset`. 
- **NOTE** Because of the legacy issue (previously there is only one scene in CPU simulator), I use `simulator._scene` to tell an entity the scene that we hope to `_load` into. The whole list of scenes is not supposed to be exposed to the entities during loading and we do not provide functions to directly operate on a single object in a single scene. Currently is the authors' duty to maintain the original sapien objects and modify them in the GPU buffer.  **We expect that the `_load` and `_get_sapien_entities` interface will be unified in the future**.

### Robot and the controller

`Robot` is a special entity. Currently, only one robot can be added per scene supported, which is an input for constructing the simulator. The `Robot` class handles the URDF and a robot's action space, sensors and ROS topics. The controllers need to be registered when creating the robot entity.

The ROS support is broken now and most legacy code is put in [robotics/sim/module.py](robotics/sim/module.py), which contains a way of determining the poses of different frames in ROS. You can ignore it now.

The get_sensor returns a dictionary of cameras mounted either on the world or on the robot. The format of the camera config is at [sensor_cfg.py](robotics/sim/sensors/sensor_cfg.py). Note that if you want to mount the camera on the robot, you need to set the `base` to something like `robot/base_link`.

Please see [examples](robotics/sim/robot/mycobot280pi.py) for more details.


## Examples
See [examples](robotics/sim/tester/test_mycobot.py) for more examples. Note the ROS support is broken now.