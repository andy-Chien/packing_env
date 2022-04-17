import pybullet as p
from time import sleep
import pybullet_data


physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf", [0,0,0])
boxId = p.loadURDF("cube.urdf", [0,3,2],useMaximalCoordinates = True)
print('1111')
# bunnyId = p.loadSoftBody("expo_marker_red.obj")#.obj")#.vtk")
bunnyId = p.loadSoftBody("power_drill.objj", basePosition = [0,0,2], scale = 5, mass = 1., 
                          useNeoHookean = 0, useBendingSprings=1, useMassSpring=1, 
                          springElasticStiffness=40, springDampingStiffness=.1, springDampingAllDirections = 1, 
                          useSelfCollision = 0, frictionCoeff = 5, useFaceContact=1)

print('2222')


# meshData = p.getMeshData(bunnyId)
# print("meshData=",meshData)
#p.loadURDF("cube_small.urdf", [1, 0, 1])
useRealTimeSimulation = 0
print('3333')

if (useRealTimeSimulation):
  p.setRealTimeSimulation(1)
print('4444')

print(p.getDynamicsInfo(planeId, -1))
print(p.getDynamicsInfo(bunnyId, 0))
print(p.getDynamicsInfo(boxId, -1))
print('5555')

p.changeDynamics(boxId,-1,mass=10)
while p.isConnected():

  # p.setGravity(0, 0, -10)

  if (useRealTimeSimulation):

    sleep(0.01)  # Time in seconds.
    #p.getCameraImage(320,200,renderer=p.ER_BULLET_HARDWARE_OPENGL )
  else:
    p.stepSimulation()


(0.0, 1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 3, 0.0)
(0.0, 1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 3, 0.0)