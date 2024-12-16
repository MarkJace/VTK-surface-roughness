# This program aims to calculate the surface roughness Ra and Rq.

import vtk
import numpy as np
from vtk.numpy_interface import dataset_adapter as dsa


# File name
Filename = 'example.vtk'
# Dimension of system in x, y ,z directions
Nx = 512    # cells number in x direction
Ny = 256    # cells number in y direction
Nz = 256    # cells number in z direction
dL = 0.5    # length per cell (um)

# use 'vtkStructuredPointsReader' to read 'structured_points' type data
reader = vtk.vtkStructuredPointsReader()
reader.SetFileName(Filename)
reader.Update()

# get the data we read
structured_points = reader.GetOutput()

# get the number of points
num_points = structured_points.GetNumberOfPoints()
print(f"Number of points: {num_points}")

# get the scalar data of points
point_data = structured_points.GetPointData()
point_scalars = point_data.GetScalars()

# use 'dsa.WrapDataObject' to wrap VTK data
data = dsa.WrapDataObject(structured_points)

# get data and transfer it to numpy array
point_values = data.PointData[point_scalars.GetName()]

# reshape the NumPy array to a 3D tensor
point_tensor = np.reshape(point_values, (Nz, Ny, Nx))

# point_tensor dimension：(z=)Nz,(y=)Ny,(x=)Nx
# Y-direction in tensor is reverse to vtk file, but here we only consider surface roughness,
# only Z-direction matters, so we can ignore the reverse Y-direction.

surf_position = np.zeros((Ny,Nx),'float32')

for i_y in range(Ny):
    for i_x in range(Nx):
        for i_z in range(Nz):
            if point_tensor[i_z,i_y,i_x] == 0:
                surf_position[i_y,i_x] = i_z
                break

surf_height = surf_position*dL


# average height
ave_height = np.mean(surf_height)
print('The average height (um):',ave_height)

# calculate the roughness
N = Nx*Ny

delta_height = abs(surf_height-ave_height)
delta_height2 = (surf_height-ave_height)**2

height_total_1 = np.sum(delta_height)
height_total_2 = np.sum(delta_height2)

Ra = height_total_1/N
print('Surface roughness Ra:',Ra)
Rq = height_total_2/N
print('Surface roughness Rq:',Rq)



