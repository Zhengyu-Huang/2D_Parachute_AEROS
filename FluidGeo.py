def writeMeshSize(file , type, field_id,  *args):

    if type == 'Attractor':
        node_list = args[-1]
        file.write('Field[%d] = Attractor;\n' %field_id)
        file.write('Field[%d].NodesList = {' %field_id)

        file.write(','.join(str(x) for x in node_list))
        file.write('};\n')


    elif(type == 'Threshold'):
        IField, LcMin, LcMax, DistMin, DistMax, Sigmoid = args
        file.write('Field[%d] = Threshold;\n' % field_id)
        file.write('Field[%d].IField = %d;\n' % (field_id, IField))
        file.write('Field[%d].LcMin = %s;\n' %(field_id, LcMin))
        file.write('Field[%d].LcMax = %s;\n' %(field_id, LcMax))
        file.write('Field[%d].DistMin = %.15f;\n' %(field_id, DistMin))
        file.write('Field[%d].DistMax = %.15f;\n' %(field_id, DistMax))
        file.write('Field[%d].Sigmoid = %d;\n' %(field_id, Sigmoid)) #False

    elif(type == 'Min'):
        field_list = args[-1]
        file.write('Field[%d] = Min;\n' % field_id)
        file.write('Field[%d].FieldsList = {' % field_id)
        file.write(','.join(str(x) for x in field_list))
        file.write('};\n')
        file.write('Background Field = %d;\n' %field_id)

def writePhysical(file, physicalEntity, phsicalName, list):
    file.write('Physical %s("%s") = {' %(physicalEntity, phsicalName))
    file.write(','.join(str(x) for x in list))
    file.write('};\n')

def writeElem(file, type, id, list):
    file.write('%s(%d) = {' % (type, id))
    file.write(','.join(str(x) for x in list))
    file.write('};\n')
    return id + 1


def backgroundMesh(file, type, x_l, x_r, y_l , y_r, layer_n, point_id=1, line_id=1, line_loop_id=1, plane_id=1):
    if(type == 'cube'):
        file.write('x_l = %.15f;\n' %x_l)
        file.write('x_r = %.15f;\n' %x_r)
        file.write('y_l = %.15f;\n' %y_l)
        file.write('y_r = %.15f;\n' %y_r)

        point_id = writeElem(file, 'Point', point_id, ['x_l', 'y_r', 0.0, 'cl_bg'])
        point_id = writeElem(file, 'Point', point_id, ['x_r',  'y_r', 0.0, 'cl_bg'])
        point_id = writeElem(file, 'Point', point_id, ['x_r',  'y_l', 0.0, 'cl_bg'])
        point_id = writeElem(file, 'Point', point_id, ['x_l', 'y_l', 0.0, 'cl_bg'])


        line_id = writeElem(file, 'Line', line_id, [point_id - 4, point_id - 3])
        line_id = writeElem(file, 'Line', line_id, [point_id - 3, point_id - 2])
        line_id = writeElem(file, 'Line', line_id, [point_id - 2, point_id - 1])
        line_id = writeElem(file, 'Line', line_id, [point_id - 1, point_id - 4])


        writeElem(file, 'Line Loop', line_loop_id, [line_id-4,line_id-3,line_id-2,line_id-1])
        writeElem(file, 'Plane Surface', plane_id, [line_loop_id])


        # extrude the background mesh
        file.write('cube_surface[] = Extrude {0, 0, %d*'%layer_n)
        file.write('cl_layer')
        file.write('} {Surface {%d}; Layers {%d};};\n' %(plane_id, layer_n))

        writePhysical(file, 'Surface', 'OutletFixedSurface', ['cube_surface[2]', 'cube_surface[3]', 'cube_surface[4]', 'cube_surface[5]'])
        writePhysical(file, 'Surface', 'SymmetryFixedSurface', ['cube_surface[0]', plane_id])
        writePhysical(file, 'Volume', 'FluidMesh', ['cube_surface[1]'])


if __name__ == '__main__':
    import sys
    id =0
    writeMeshSize(sys.stdout, 'Attractor', 1, [1,2,3,4])
    writeMeshSize(sys.stdout, 'Threshold', 3, 2, 0.1,0.1,0.01,5,0)
    writeMeshSize(sys.stdout, 'Min', 5, [3, 4])
    writePhysical(sys.stdout, 'Surface', 'OutletFixedSurface',[4009,4021,4017])
    writePhysical(sys.stdout, 'Volume', 'FluidMesh', ['cube_surface[1]', id])
    writeElem(sys.stdout, 'Point', id, [1,2,3,'cl'])
    writeElem(sys.stdout, 'Line', id, [1,2])
    backgroundMesh(sys.stdout, 'cube', -1, 1, -1, 1, 4, point_id=1, line_id=1, line_loop_id=1,
                   plane_id=1)



