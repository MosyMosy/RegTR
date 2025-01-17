# -*- coding: utf-8 -*-
"""
VTK visualization object

@author: Øystein Skotheim, SINTEF ICT <oystein.skotheim@sintef.no>
"""

import vtk
from vtk.util import numpy_support
import matplotlib.colors
from matplotlib.pyplot import cm as colormap
import numpy as np


class VTKObject:
    """VTK visualization object
    Class that sets up the necessary VTK pipeline for displaying
    various objects (point clouds, meshes, geometric primitives)"""

    def __init__(self):
        self.verts = None
        self.cells = None
        self.scalars = None
        self.normals = None
        self.pd = None
        self.LUT = None
        self.mapper = None
        self.actor = None

    def CreateFromSTL(self, filename):
        "Create a visualization object from a given STL file"

        if not filename.lower().endswith('.stl'):
            raise Exception('Not an STL file')

        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename)
        reader.Update()

        self.pd = vtk.vtkPolyData()
        self.pd.DeepCopy(reader.GetOutput())

        self.SetupPipelineMesh()

    def CreateFromPLY(self, filename):
        "Create a visualization object from a given PLY file"

        if not filename.lower().endswith('.ply'):
            raise Exception('Not a PLY file')

        reader = vtk.vtkPLYReader()
        reader.SetFileName(filename)
        reader.Update()

        self.pd = vtk.vtkPolyData()
        self.pd.DeepCopy(reader.GetOutput())

        self.SetupPipelineMesh()

    def CreateFromActor(self, actor):
        "Create a visualization object from a given vtkActor instance"
        if not isinstance(actor, vtk.vtkObject):
            raise Exception('Argument is not a VTK Object')
        self.actor = actor

    def CreateFromPolyData(self, pd):
        "Create a visualization object from a given vtkPolyData instance"
        self.pd = pd
        self.SetupPipelineMesh()

    def CreateFromArray(self, pc):
        """Create a point cloud visualization object from a given NumPy array

        The NumPy array should have dimension Nxd where d >= 3

        If d>3, the points will be colored according to the last column
        in the supplied array (values should be between 0 and 1, where
        0 is black and 1 is white)
        """

        nCoords = pc.shape[0]
        nElem = pc.shape[1]

        if nElem < 3:
            raise Exception('Number of elements must be greater than or equal to 3')

        self.verts = vtk.vtkPoints()
        self.cells = vtk.vtkCellArray()
        self.scalars = None
        self.pd = vtk.vtkPolyData()

        # Optimized version of creating the vertices array
        # - We need to be sure to keep the supplied NumPy array,
        #   because the numpy_support function creates a pointer into it
        # - We need to take a copy to be sure the array is contigous
        self.points_npy = pc[:, :3].copy()
        self.verts.SetData(numpy_support.numpy_to_vtk(self.points_npy))

        # Optimized version of creating the cell ID array
        # - We need to be sure to keep the supplied NumPy array,
        #   because the numpy_support function creates a pointer into it
        # - Note that the cell array looks like this: [1 vtx0 1 vtx1 1 vtx2 1 vtx3 ... ]
        #   because it consists of vertices with one primitive per cell
        self.cells_npy = np.vstack([np.ones(nCoords, dtype=int64),
                                    np.arange(nCoords, dtype=int64)]).T.flatten()

        self.cells.SetCells(nCoords, numpy_support.numpy_to_vtkIdTypeArray(self.cells_npy))

        self.pd.SetPoints(self.verts)
        self.pd.SetVerts(self.cells)

        # Optimized version of creating the scalars array
        # - We need to be sure to keep the NumPy array,
        #   because the numpy_support function creates a pointer into it
        # - We need to take a copy to be sure the array is contigous
        if (nElem > 3):
            self.scalars_npy = pc[:, -1].copy()
            self.scalars = numpy_support.numpy_to_vtk(self.scalars_npy)

            # Color the scalar data in gray values
            self.LUT = vtk.vtkLookupTable()
            self.LUT.SetNumberOfColors(255)
            self.LUT.SetSaturationRange(0, 0)
            self.LUT.SetHueRange(0, 0)
            self.LUT.SetValueRange(0, 1)
            self.LUT.Build()
            self.scalars.SetLookupTable(self.LUT)
            self.pd.GetPointData().SetScalars(self.scalars)

        self.SetupPipelineCloud()

    def AddNormals(self, ndata):
        "Add surface normals (Nx3 NumPy array) to the current point cloud visualization object"

        nNormals = ndata.shape[0]
        nDim = ndata.shape[1]

        if (nDim != 3):
            raise Exception('Expected Nx3 array of surface normals')

        if (self.pd.GetNumberOfPoints() == 0):
            raise Exception('No points to add normals for')

        if (nNormals != self.pd.GetNumberOfPoints()):
            raise Exception('Supplied number of normals incompatible with number of points')

        # Optimized version of creating the scalars array
        # - We need to be sure to keep the NumPy array,
        #   because the numpy_support function creates a pointer into it
        # - We need to take a copy to be sure the array is contigous
        self.normals_npy = ndata.copy()
        self.normals = numpy_support.numpy_to_vtk(self.normals_npy)

        self.pd.GetPointData().SetNormals(self.normals)
        self.pd.Modified()

    def SetColors(self, cdata, cmap: str = None, color_norm=None):
        """ Add colors (NumPy array of size (N,3) or (3,)) to the current point cloud visualization object

        Args:
            cdata: Color data.
                   If cmap is not provided, should be a numpy array of size (N,3) or (3,).
                       Should be of type uint8 with R, G, B values between 0 and 255.
                   If cmap is provided, should be a numpy array of size (N)

            cmap: matplotlib colormap. See https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
            color_norm: Color normalization scheme

        Returns:

        """
        if self.pd.GetNumberOfPoints() == 0:
            raise Exception('No points to add color for')
        num_points = self.pd.GetNumberOfPoints()

        if isinstance(cdata, list) or isinstance(cdata, tuple):
            cdata = np.array(cdata)

        if cmap is not None:
            assert cdata.shape == (num_points,)
            color_mapper = colormap.ScalarMappable(norm=color_norm,
                                                   cmap=colormap.get_cmap(cmap))

            if cmap == 'YlOrRd':
                self.colors_npy = color_mapper.to_rgba(cdata, bytes=False)[..., :3]
                inv_gamma = 1.0 / 0.5
                self.colors_npy = ((self.colors_npy ** inv_gamma) * 255).astype(np.uint8)
            else:
                self.colors_npy = color_mapper.to_rgba(cdata, bytes=True)[..., :3]


        elif cdata.ndim == 1:
            self.colors_npy = np.broadcast_to(cdata, (self.pd.GetNumberOfPoints(), 3)).copy().astype(np.uint8)

        else:
            if cdata.shape[0] != num_points:
                raise Exception('Supplied number of colors incompatible with number of points')

            if cdata.shape[1] != 3:
                raise Exception('Expected Nx3 array of colors')

            # Optimized version of creating the scalars array
            # - We need to be sure to keep the NumPy array,
            #   because the numpy_support function creates a pointer into it
            # - We need to take a copy to be sure the array is contigous
            self.colors_npy = cdata.copy().astype(np.uint8)

        self.colors = numpy_support.numpy_to_vtk(self.colors_npy)
        self.pd.GetPointData().SetScalars(self.colors)
        self.pd.Modified()

    def SetupPipelineCloud(self):
        "Set up the VTK pipeline for visualizing a point cloud"

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.pd)

        if (self.scalars != None):
            self.ScalarsOn()
            self.SetScalarRange(0, 1)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetRepresentationToPoints()
        self.actor.GetProperty().SetColor(0.0, 1.0, 0.0)

    def SetupPipelineMesh(self):
        "Set up the VTK pipeline for visualizing a mesh"

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.pd)

        if self.scalars != None:
            if self.pd.GetPointData() != None:
                if (self.pd.GetPointData().GetScalars != None) and (self.LUT != None):
                    self.pd.GetPointData().GetScalars().SetLookupTable(self.LUT)

            self.mapper.ScalarVisibilityOn()
            self.mapper.SetColorModeToMapScalars()
            self.mapper.SetScalarModeToUsePointData()
            self.mapper.SetScalarRange(0, 1)

        else:
            self.mapper.ScalarVisibilityOff()

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetRepresentationToSurface()
        self.actor.GetProperty().SetInterpolationToGouraud()

    def SetupPipelineHedgeHog(self, scale=15.0):
        """Set up the VTK pipeline for visualizing points with surface normals"

        The surface normals are visualized as lines with the given scale"""

        hh = vtk.vtkHedgeHog()
        hh.SetInputData(self.pd)
        hh.SetVectorModeToUseNormal()
        hh.SetScaleFactor(scale)
        hh.Update()

        # Set up mappers and actors
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(hh.GetOutputPort())
        self.mapper.Update()
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetColor(0, 0, 0)  # Default color

    def ScalarsOn(self):
        "Enable coloring of the points based on scalar array"
        self.mapper.ScalarVisibilityOn()
        self.mapper.SetColorModeToMapScalars()
        self.mapper.SetScalarModeToUsePointData()

    def ScalarsOff(self):
        "Disable coloring of the points based on scalar array"
        self.mapper.ScalarVisibilityOff()

    def SetScalarRange(self, smin, smax):
        "Set the minimum and maximum values for the scalar array"
        self.mapper.SetScalarRange(smin, smax)

    def GetActor(self):
        "Returns the current actor"
        return self.actor

    # Primitives
    def CreateSphere(self, origin, r, color=None):
        "Create a sphere with given origin (x,y,z) and radius r"
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(origin)
        sphere.SetRadius(r)
        sphere.SetPhiResolution(25)
        sphere.SetThetaResolution(25)
        sphere.Update()

        self.pd = vtk.vtkPolyData()
        self.pd.DeepCopy(sphere.GetOutput())
        self.scalars = None
        self.SetupPipelineMesh()

        if color is not None:
            self.actor.GetProperty().SetColor(color[0], color[1], color[2])

    def CreateCylinder(self, origin, r, h):
        "Create a cylinder with given origin (x,y,z), radius r and height h"
        cyl = vtk.vtkCylinderSource()
        cyl.SetCenter(origin)
        cyl.SetRadius(r)
        cyl.SetHeight(h)
        cyl.SetResolution(25)
        cyl.Update()

        self.pd = vtk.vtkPolyData()
        self.pd.DeepCopy(cyl.GetOutput())
        self.scalars = None
        self.SetupPipelineMesh()

    def CreateAxes(self, length):
        "Create a coordinate axes system with a given length of the axes"
        axesActor = vtk.vtkAxesActor()
        axesActor.AxisLabelsOff()
        axesActor.SetTotalLength(length, length, length)
        self.actor = axesActor

    def CreatePlane(self, normal=None, origin=None):
        "Create a plane (optionally with a given normal vector and origin)"
        plane = vtk.vtkPlaneSource()
        plane.SetXResolution(10)
        plane.SetYResolution(10)
        if not normal is None: plane.SetNormal(normal)
        if not origin is None: plane.SetCenter(origin)
        plane.Update()

        self.pd = vtk.vtkPolyData()
        self.pd.DeepCopy(plane.GetOutput())
        self.scalars = None
        self.SetupPipelineMesh()

    def CreateBox(self, bounds):
        "Create a box with the given bounds [xmin,xmax,ymin,ymax,zmin,zmax]"
        box = vtk.vtkTessellatedBoxSource()
        box.SetBounds(bounds)
        box.Update()
        self.pd = box.GetOutput()
        self.scalars = None
        self.SetupPipelineMesh()

    def CreateLine(self, p1, p2):
        "Create a 3D line from p1=[x1,y1,z1] to p2=[x2,y2,z2]"
        line = vtk.vtkLineSource()
        line.SetPoint1(*p1)
        line.SetPoint2(*p2)
        line.Update()

        self.pd = vtk.vtkPolyData()
        self.pd.DeepCopy(line.GetOutput())
        self.scalars = None
        self.SetupPipelineMesh()

    def CreateLines(self, lines, line_color=(1.0, 1.0, 1.0), line_width=1):

        if not isinstance(lines, np.ndarray):
            lines = np.array(lines)
        if not isinstance(line_color, np.ndarray):
            line_color = np.array(line_color)
        if issubclass(line_color.dtype.type, integer):
            line_color = line_color.astype(float) / 255

        assert lines.ndim == 2 and lines.shape[1] == 6, \
            'Lines should be a list with each element containing 6 values'

        num_lines = lines.shape[0]

        self.verts = vtk.vtkPoints()
        self.points_npy = lines.reshape(-1, 3).copy()
        self.verts.SetData(numpy_support.numpy_to_vtk(self.points_npy))

        # Create line connections
        lines = vtk.vtkCellArray()
        for i in range(num_lines):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, 2*i)
            line.GetPointIds().SetId(1, 2 * i + 1)
            lines.InsertNextCell(line)

        # Create a polydata to store everything in
        self.pd = vtk.vtkPolyData()
        self.pd.SetPoints(self.verts)
        self.pd.SetLines(lines)

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.pd)
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetColor(*line_color)
        self.actor.GetProperty().SetLineWidth(line_width)

    def CreateText(self, string, pos=(0, 0)):

        textProperty = vtk.vtkTextProperty()
        textProperty.SetFontSize(16)
        textProperty.SetColor(1.0, 1.0, 1.0)

        self.mapper = vtk.vtkTextMapper()
        self.actor = vtk.vtkActor2D()
        self.mapper.SetInput(string)
        self.mapper.SetTextProperty(textProperty)
        self.actor.SetMapper(self.mapper)
        self.actor.SetPosition(pos[0], pos[1])
