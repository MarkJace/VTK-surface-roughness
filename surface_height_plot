# This program aims to analyze 3d .vtk file and draw its surface height plot.

import vtk
import numpy as np
from vtk.numpy_interface import dataset_adapter as dsa
import matplotlib.pyplot as plt


# File name
Filename = 'example.vtk'
# Dimension of system in x, y ,z directions
Nx = 512    # cells number in x direction
Ny = 256    # cells number in y direction
Nz = 256    # cells number in z direction
dL = 0.5    # length per cell

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
# Y-direction in tensor is reverse to vtk file,
# So we should impose an inversion of Y-axis for tensor(matrix).

surf_position = np.zeros((Ny,Nx),'float32')

for i_y in range(Ny):
    for i_x in range(Nx):
        for i_z in range(Nz):
            if point_tensor[i_z,i_y,i_x] == 0:
                surf_position[Ny-1-i_y,i_x] = i_z
                break

surf_height = surf_position*dL

# Draw the surface height plot
plt.imshow(surf_height, cmap='viridis')
plt.colorbar()  # show the colorbar
plt.title('The Surface Height Plot (um)')  # plot title
plt.axis('off')  # close the axis
plt.show()  # show the plot
