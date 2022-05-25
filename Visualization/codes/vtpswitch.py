import vtk


def main():
    for i in range(490):
        WriteVTP('./results/output' + str(i) + '.vtk', ReadVTU('./results/output' + str(i) + '.vtk'))
        print(i)






def WriteVTP(filename, vtkpolydata):
    write = vtk.vtkXMLPolyDataWriter()
    write.SetFileName(filename)
    write.SetDataModeToBinary()
    write.SetInputData(vtkpolydata)
    write.Write()


def ReadVTU(filename):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(reader.GetOutput())
    surface_filter.Update()
    vtkpolydata = surface_filter.GetOutput()
    print(" 输入 UNSTRUCTURED_GRID point 数量 ")
    print(reader.GetOutput().GetNumberOfPoints())
    print(" \n ")
    print(" 输出 vtkPolyData point 数量 ")
    print(surface_filter.GetOutput().GetNumberOfPoints())
    print(" \n ")
    return vtkpolydata


if __name__ == '__main__':
    main()
