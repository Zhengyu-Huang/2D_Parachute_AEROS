import numpy as np
import matplotlib.pyplot as plt
import math
import Folding ,Cable ,AeroShell
#This is a mesh generator for quasi-1D parachte
#this is the one with matcher dressing techniques for the cables(suspension lines)




class Parachute:


#############################################################################################################################
#                                       \-----------------------\
#                                        \                      \ (layer_n + 1)*canopy_n - layer_n/2
#                         layer_n/2+1    \\                    / \
#                                         |\__________________/___\
#                                          \                 /
#                                           \               /
#                                            \             /
#                                             \          /
#                                              \       /
#                                               *******
#                            layer_n*canopy_n+1        layer_n*canopy_n+1
#
#
#For the canopy, it has nPoints-1 segments and is extruded 4 layers in the z direction,  the points are labeled as
#5 10 15 ... 5*nPoints
#4 9  14 ... 5*nPoints-1
#3 8  13 ... 5*nPoints-2
#2 7  12 ... 5*nPoints-3
#1 6  11 ... 5*nPoints-4
#For the cables, it has mPoints segments, 4 nodes connecting these cables are 1 , 4 , 4*nPoins - 3 , 4*nPoints, on the cables these nodes are
# 1 ,            5 ,            5*nPoins - 4 ,   5*nPoints,
# 5*nPoints + 1, 5*nPoints + 2, 5*nPoints + 3,   5*nPoints + 4,
# 5*nPoints + 5, 5*nPoints + 6, 5*nPoints + 7,   5*nPoints + 8,
# ......
# 5*nPoints + 4*mPoints - 7, 5*nPoints + 4*mPoints - 6, 5*nPoints + 4*mPoints - 5,   5*nPoints + 4*mPoints - 4,
# 5*nPoints + 4*mPoints - 3, 5*nPoints + 4*mPoints - 3, 5*nPoints + 4*mPoints - 3,   5*nPoints + 4*mPoints - 3,
# For the payload, it is a square panel
# 5*nPoints + 4*mPoints - 2, 5*nPoints + 4*mPoints - 1, 5*nPoints + 4*mPoints,   5*nPoints + 4*mPoints + 1




    def __init__(self, canopy, cable, capsule):
        '''

        :param canopy_n: (1D) canopy node number in x direction
        :param canopy_x: x coordinate of canopy nodes
        :param canopy_y: y coordinate of canopy nodes
        :param cable_n: node number on suspension line
        :param cable_k: node number on a cross-section of dressing surface, node number on dressing surface is cable_n*cable_k + 2(bottom and top)
        :param cable_r: cable radius
        :param layer_n: number of layers extruded in z direction, node number in z direction is layer_n+1
        :param capsule_y: capsule is at (x = 0, y = capsule_y, z = layer_n /2 * d)

        Attributes:
        self.canopy_node: an array of node number id in self.coord

        self.cable1_node: node number id in self.coord of left suspension line, its dressing is around node 1 to node cable_n-2,  from bottom to top,
        layer by layer, for those layer with dressing, it is labeled from center to dressing surface

        self.cable2_node: node number id in self.coord of right suspension line, from bottom to top

        self.capsule_node: node number of capsule nodes

        self.structure_coord
        self.emebedded_coord
        '''
        canopy_type, canopy_cl, canopy_xScale, canopy_yScale, self.layer_n, layer_t, *canopy_args = canopy
        *cable_type, cable_cl, self.cable_k,self.cable_r, self.cable_joint, self.phantom_offset = cable
        capsule_type, capsule_xScale, capsule_yScale = capsule


        self.layer_n = layer_n
        #######################################################################################
        # Canopy node
        #######################################################################################

        canopy_n, canopy_x, canopy_y = Folding.Canopy(canopy_type, canopy_cl, canopy_xScale, canopy_yScale, *canopy_args)
        self.canopy_n = canopy_n
        self.canopy_node = list(range((layer_n + 1) * canopy_n))


        #######################################################################################
        # Capsule node
        #######################################################################################

        capsule_n, capsule_x, capsule_y = AeroShell.AeroShell(capsule_type, capsule_xScale, capsule_yScale)
        self.capsule_n = capsule_n
        self.capsule_node = list(range((layer_n + 1) * canopy_n, (layer_n + 1) * canopy_n + (layer_n + 1)*capsule_n))

        #######################################################################################
        # Cable node
        #######################################################################################

        #we have 6 cables
        self.As = np.array([[canopy_x[0],canopy_y[0],layer_n//2*layer_t],           [canopy_x[-1],canopy_y[-1],layer_n//2*layer_t],
                       [cable_joint[0,0],cable_joint[0,1],layer_n//2*layer_t], [cable_joint[1,0],cable_joint[1,1],layer_n//2*layer_t],
                       [cable_joint[1,0],cable_joint[1,1],layer_n//2*layer_t], [cable_joint[1,0],cable_joint[1,1],layer_n//2*layer_t]],dtype= float)
        self.Bs = np.array([[cable_joint[0,0],cable_joint[0,1],layer_n//2*layer_t], [cable_joint[0,0],cable_joint[0,1],layer_n//2*layer_t],
                       [cable_joint[1,0],cable_joint[1,1],layer_n//2*layer_t], [capsule_x[0],capsule_y[0],layer_n//2*layer_t],
                       [capsule_x[1],capsule_y[1],layer_n//2*layer_t], [capsule_x[2],capsule_y[2],layer_n//2*layer_t]], dtype = float)

        cables_n, cables_coord, phantoms_coord = Cable.Cables(cable_type, self.As, self.Bs, cable_cl, cable_k, cable_r)
        self.cables_n = cables_n
        #################################################################################################################
        self.cables_node = []
        cable_joint_node =[(layer_n + 1) * canopy_n + (layer_n + 1)*capsule_n, (layer_n + 1) * canopy_n + (layer_n + 1)*capsule_n + 1]
        cable_start = (layer_n + 1) * canopy_n + (layer_n + 1)*capsule_n + 1
        for i in range(6):
            self.cables_node.append(list(range(cable_start,cable_start + cables_n[i])))
            cable_start += cables_n[i] - 2

        self.structure_node_n =  cable_start+1

        self.cables_node[0][0], self.cables_node[0][-1] = self.canopy_node[layer_n // 2], cable_joint_node[0]
        self.cables_node[1][0], self.cables_node[1][-1] = self.canopy_node[(layer_n + 1) * (canopy_n - 1) + layer_n // 2], cable_joint_node[0]
        self.cables_node[2][0], self.cables_node[2][-1] = cable_joint_node[0], cable_joint_node[1]
        self.cables_node[3][0], self.cables_node[3][-1] = cable_joint_node[1], self.capsule_node[layer_n//2]
        self.cables_node[4][0], self.cables_node[4][-1] = cable_joint_node[1], self.capsule_node[layer_n//2 +    1 +layer_n]
        self.cables_node[5][0], self.cables_node[5][-1] = cable_joint_node[1], self.capsule_node[layer_n//2 + 2*(1+layer_n)]

        self.phantoms_node = []
        phantom_start = (layer_n + 1) * canopy_n + (layer_n + 1)*capsule_n
        for i in range(6):
            self.phantoms_node.append(list(range(phantom_start,phantom_start + cable_k*(cables_n[i] - 2*phantom_offset) + 2)))
            phantom_start += cable_k*(cables_n[i] - 2*phantom_offset) + 2
        #add 2 because we wants to close the beam surface
        self.embedded_node_n  =  phantom_start


        ##################################################################################
        # Start the coordinate part
        ##################################################################################
        self.structure_coord = np.empty(shape=[self.structure_node_n,3],dtype=float)
        self.embedded_coord = np.empty(shape=[self.embedded_node_n, 3], dtype=float)
        for i in range(canopy_n):
           for j in range(layer_n + 1):
               self.structure_coord[self.canopy_node[i*(layer_n+1) + j],:] = [canopy_x[i],canopy_y[i],layer_t*j]
               self.embedded_coord[self.canopy_node[i * (layer_n + 1) + j], :] = [canopy_x[i], canopy_y[i], layer_t * j]



        for i in range(capsule_n):
           for j in range(layer_n + 1):
               self.structure_coord[self.capsule_node[i*(layer_n+1) + j],:] = [capsule_x[i],capsule_y[i],layer_t*j]
               self.embedded_coord[self.capsule_node[i * (layer_n + 1) + j], :] = [capsule_x[i], capsule_y[i], layer_t * j]

        for i in range(2):
            self.structure_coord[cable_joint_node[i],0:2],self.structure_coord[cable_joint_node[i],2] = cable_joint[i,:], layer_n//2*layer_t

        for i in range(6):
            for j in range(phantom_offset, cables_n[i] - phantom_offset):
                self.structure_coord[self.cables_node[i][j], :] = cables_coord[i][j,:]
                #first node for the beam surface
                self.embedded_coord[self.phantoms_node[i][0], :] = cables_coord[i][phantom_offset, :]
                for k in range(cable_k):

                    self.embedded_coord[self.phantoms_node[i][(j-phantom_offset)*cable_k + k], :] = phantoms_coord[i][j*cable_k +k,:]
                #last node for the beam surface
                self.embedded_coord[self.phantoms_node[i][-1], :] = cables_coord[i][-1-phantom_offset, :]

        self.thickness = 7.62e-5






    def _write_coord(self,file,which):
        '''
        Write the node coords on the real structure/embedded surface, not include phantom element nodes/beam-inside nodes
        To write embedded surface, mask  =  self.embeddedsurface_mask
        To write structure, mask  =  self.structure_mask
        :param file: output file
        :return: None
        '''

        coord = self.embedded_coord if which == 'embedded' else self.structure_coord
        id = 1
        for xyz in  coord:
            file.write('%d   %.15f  %.15f  %.15f \n' % (id, xyz[0], xyz[1], xyz[2]))
            id += 1


    def _write_cable_beam(self,file,topo,start_id):
        '''
        Write the beam element
        :param file: output file
        :param topo: element topology
        :param start_id: start node id in the output file
        :return:
        '''
        id = start_id
        for i in range(6):
            cable_node = self.cables_node[i]
            for j in range(len(cable_node) - 1):
                file.write('%d   %d   %d   %d\n' %(id, topo, cable_node[j] + 1, cable_node[j + 1] + 1))
                id += 1
        return id


    def _write_cable_frames(self, file):

        file.write('EFRAMES\n')
        ele_id = 2*self.layer_n*(self.capsule_n + self.canopy_n - 1) + 1
        for cable_i in range(6):
            cable_node = self.cables_node[cable_i]
            for j in range(len(cable_node) - 1):
                A = self.structure_coord[cable_node[j],:]
                B = self.structure_coord[cable_node[j+1],:]

                Sx = (B - A) / np.linalg.norm(B - A)  # direction of AB
                e1 = np.array([0.0, 0.0, 1.0], dtype=float)
                Sy = np.cross(Sx, e1)
                Sz = np.cross(Sx, Sy)
                file.write('%d  %f %f %f    %f %f %f    %f %f %f\n' % (ele_id, Sx[0], Sx[1], Sx[2], Sy[0], Sy[1], Sy[2], Sz[0], Sz[1], Sz[2]))
                ele_id += 1

        file.write('*\n')



    def _write_cable_surface(self,file,topo,id):
        k = self.cable_k
        temp = range(1, k + 1)
        for cable_i in range(6):
            phantom_node = self.phantoms_node[cable_i]
            n = len(self.cables_node[cable_i])


            #top triangles
            for j in range(k):
                file.write('%d   %d  %d  %d  %d\n' % (
                id, topo, phantom_node[0] + 1, phantom_node[temp[j]] + 1, phantom_node[temp[j-1]] + 1))
                id += 1


            for i in range(n - 1 - 2*self.phantom_offset):
                for j in range(k):

                    file.write('%d   %d  %d  %d  %d\n' % (id, topo, phantom_node[i * k + temp[j - 1]] + 1,
                                                          phantom_node[i * k + temp[j]] + 1,
                                                          phantom_node[(i+1) * k + temp[j - 1]] + 1))
                    id += 1
                    file.write('%d   %d  %d  %d  %d\n' % (id, topo, phantom_node[i * k + temp[j]] + 1,
                                                          phantom_node[(i + 1) * k + temp[j]] + 1,
                                                          phantom_node[(i + 1) * k + temp[j - 1]] + 1))
                    id += 1

            # phantom triangle at bottom
            for j in range(k):
                file.write('%d   %d  %d  %d  %d\n' % (
                id, topo, phantom_node[-1] + 1, phantom_node[-1 - temp[j-1]] + 1,
                phantom_node[-1 - temp[j]] + 1))
                id += 1

        return id

    def _write_canopy_surface(self,file,topo,start_id, special = True):
        n = self.canopy_n
        layer_n = self.layer_n
        canopy_node = self.canopy_node
        id = start_id;
        if(not special):
            for i in range(n - 1):
                for j in range(layer_n):

                    # (layer_n+1)*i + j        (layer_n+1)*(i+1) + j
                    # (layer_n+1)*i + j + 1    (layer_n+1)*(i+1) + j + 1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo,  canopy_node[(layer_n+1)*i + j]+1, canopy_node[(layer_n+1)*(i+1) + j+1]+1, canopy_node[(layer_n+1)*(i+1) + j]+1))
                    id += 1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo,  canopy_node[(layer_n+1)*i + j]+1, canopy_node[(layer_n+1)*i + j + 1]+1, canopy_node[(layer_n+1)*(i+1) + j + 1]+1))
                    id += 1
        else:
            for i in range(n-1):
                if(i <= (n-2)/2):
                    file.write('%d   %d  %d  %d %d \n' %(id, topo, (layer_n+1)*i +1, (layer_n+1)*i + 6, (layer_n+1)*i +7))
                    id +=1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo, (layer_n+1)*i +1, (layer_n+1)*i + 7, (layer_n+1)*i +2))
                    id +=1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo, (layer_n+1)*i +2, (layer_n+1)*i + 7, (layer_n+1)*i +3))
                    id +=1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo, (layer_n+1)*i +3, (layer_n+1)*i + 7, (layer_n+1)*i +8))
                    id +=1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo, (layer_n+1)*i +3, (layer_n+1)*i + 8, (layer_n+1)*i +9))
                    id +=1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo, (layer_n+1)*i +3, (layer_n+1)*i + 9, (layer_n+1)*i +4))
                    id +=1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo, (layer_n+1)*i +4, (layer_n+1)*i + 9, (layer_n+1)*i +5))
                    id +=1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo, (layer_n+1)*i +5, (layer_n+1)*i + 9, (layer_n+1)*i +10))
                    id +=1

                else:
                    file.write('%d   %d  %d  %d %d \n' %(id, topo, (layer_n+1)*i +1, (layer_n+1)*i + 6, (layer_n+1)*i +2))
                    id +=1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo, (layer_n+1)*i +2, (layer_n+1)*i + 6, (layer_n+1)*i +7))
                    id +=1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo, (layer_n+1)*i +2, (layer_n+1)*i + 7, (layer_n+1)*i +8))
                    id +=1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo, (layer_n+1)*i +2, (layer_n+1)*i + 8, (layer_n+1)*i +3))
                    id +=1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo, (layer_n+1)*i +3, (layer_n+1)*i + 8, (layer_n+1)*i +4))
                    id +=1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo, (layer_n+1)*i +4, (layer_n+1)*i + 8, (layer_n+1)*i +9))
                    id +=1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo, (layer_n+1)*i +4, (layer_n+1)*i + 9, (layer_n+1)*i +10))
                    id +=1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo, (layer_n+1)*i +4, (layer_n+1)*i + 10, (layer_n+1)*i +5))
                    id +=1

        return id

    def _write_capsule_surface(self,file,topo,id):



        capsule_node =self.capsule_node

        layer_n = self.layer_n
        for i in range(self.capsule_n):
            for j in range(layer_n):

                # (layer_n+1)*i + j        (layer_n+1)*(i+1) + j
                # (layer_n+1)*i + j + 1    (layer_n+1)*(i+1) + j + 1
                file.write('%d   %d  %d  %d %d \n' % (id, topo, capsule_node[(layer_n + 1) * i + j] + 1,
                                                      capsule_node[(layer_n + 1) * ((i + 1)%self.capsule_n) + j + 1] + 1,
                                                      capsule_node[(layer_n + 1) * ((i + 1)%self.capsule_n) + j] + 1))
                id += 1
                file.write('%d   %d  %d  %d %d \n' % (id, topo, capsule_node[(layer_n + 1) * i + j] + 1,
                                                      capsule_node[(layer_n + 1) * i + j + 1] + 1,
                                                      capsule_node[(layer_n + 1) * ((i + 1)%self.capsule_n) + j + 1] + 1))
                id += 1
        return id

    def _file_write_structure_top(self):
        file = open('structure.top','w')
        id = 1
        file.write('Nodes nodeset\n')
        self._write_coord(file,'structure')


        file.write('Elements canopy using nodeset\n')
        topo = 4;
        id = self._write_canopy_surface(file,topo,id)


        file.write('Elements cable beam using nodeset\n')
        topo = 1
        id = self._write_cable_beam(file,topo,id)


        file.write('Elements capsule using nodeset\n')
        topo = 4
        id = self._write_capsule_surface(file,topo,id)

        file.close()

    def _file_write_surface_top(self):
        
        file = open('surface.top','w')
        id = 1
        file.write('SURFACETOPO 1 SURFACE_THICKNESS %f\n' %(self.thickness))
        topo = 3;
        id = self._write_canopy_surface(file,topo,id)
        file.write('*\n')

        file.write('SURFACETOPO 2 \n')
        topo = 2;
        id = self._write_cable_surface(file, topo, id)
        file.write('*\n')


        file.write('SURFACETOPO 3 \n')
        topo = 3;
        id = self._write_capsule_surface(file,topo,id)
        file.write('*\n')

        file.close()

    def _file_write_embedded_surface_top(self):
        file = open('embeddedSurface.top','w')

        file.write('Nodes nodeset\n')
        self._write_coord(file,'embedded')
        id = 1
        file.write('Elements StickMovingSurface_8 using nodeset\n')
        topo = 4;
        id = self._write_canopy_surface(file,topo,id)

        file.write('Elements StickMovingSurface_9 using nodeset\n')
        topo = 4
        id = self._write_cable_surface(file,topo,id)


        file.write('Elements StickMovingSurface_10 using nodeset\n')
        topo = 4
        id = self._write_capsule_surface(file,topo,id)

        file.close()

    def _file_write_common_data_include(self):
        file = open('common.data.include','w')

        file.write('NODES\n')
        self._write_coord(file,'structure')

        # TopologyId, finite element type, node1Id, node2Id, node3Id ..
        # (some element type has more nodes, but 129 is 3 nodes membrane element)
        # First part for the canopy
        id = 1
        topo = 129;
        file.write('TOPOLOGY\n')
        id = self._write_canopy_surface(file,topo,id)


        #beam element 6
        topo = 6;
        id = self._write_cable_beam(file,topo,id)

        #capsule triangle around
        topo = 129
        id = self._write_capsule_surface(file,topo,id)
        file.close()

    def _file_write_aeros_mesh_include(self, Canopy_Matlaw = 'HyperElasticPlaneStress', pressure = -1, gravity = -1):


        file = open('aeros.mesh.include','w')
        # elementId, element attribute(material) id

        file.write('ATTRIBUTS\n')
        canopy_attr = 1;
        start_ele = 1
        end_ele = 2*self.layer_n*(self.canopy_n - 1)
        file.write('%d   %d   %d\n' %(start_ele, end_ele, canopy_attr))

        capsule_attr = 2;
        start_ele = end_ele + 1
        end_ele = end_ele + 2 * self.capsule_n * self.layer_n
        file.write('%d   %d   %d\n' % (start_ele, end_ele, capsule_attr))

        cable_beam_attr = 3;
        start_ele = end_ele + 1
        for i in range(6):
            end_ele += self.cables_n[i] - 1
        file.write('%d   %d   %d\n' %(start_ele, end_ele, cable_beam_attr))



        file.write('*\n')



        ###############################################################################################################################################################

        # Material specifies material
        # material id, ....
        youngsModulus = 6.08e8
        poissonRatio = 0.4
        density = 1153.4
        thickness = self.thickness
        file.write('MATERIAL\n')

        file.write('%d 0 %f %f %f 0 0 %f 0 0 0 0 0 0 0\n' %(canopy_attr, youngsModulus, poissonRatio, density, thickness))


        file.write('%d 0 %f %f %f 0 0 %f 0 0 0 0 0 0 0\n' %(capsule_attr, youngsModulus, poissonRatio, density, thickness))


        E = 3.0e8
        poissonRatio = 0.4
        rho = 1000
        cable_r = self.cable_r
        Ix = np.pi*cable_r**4/4.0
        Iy = np.pi*cable_r**4/4.0
        Iz = np.pi*cable_r**4/4.0
        area= np.pi*cable_r*cable_r

        file.write('%d %.15f %.15f %.10E %.10E 0 0 0 0 0 0 0 %.15f %.15f %.15f\n' %(cable_beam_attr, area, E, poissonRatio, rho, Ix,Iy,Iz))
        file.write('*\n')



        self._write_cable_frames(file)



        # MATUSAGE specifies material
        # start element number, end element number, material id
        file.write('MATUSAGE\n')
        file.write('1 %d  1\n' %(2*self.layer_n*(self.canopy_n - 1)))
        #aerosMeshInclude.write('%d %d 2\n' %(8*nPoints - 7,8*nPoints + 4*mPoints - 8))
        file.write('*\n')

        # MATLAW can specify nonlinear property of the material
        # material id, material name, ...
        file.write('MATLAW\n')
        if(Canopy_Matlaw == 'HyperElasticPlaneStress'):
            file.write('1 HyperElasticPlaneStress %f %f %f %f\n' %(density, youngsModulus, poissonRatio, thickness))
        elif(Canopy_Matlaw == 'PlaneStressViscoNeoHookean'):
            file.write('1 PlaneStressViscoNeoHookean %f %f %f 0.4 10 0.3 50 0.2 100 %f\n' % (density, youngsModulus, poissonRatio, thickness))

        file.write('*\n')


        if(pressure):
            file.write('PRESSURE\n')
            file.write('1 %d %f\n' %(2*self.layer_n*(self.canopy_n - 1), pressure))
            file.write('*\n')

        if(gravity):
            file.write('GRAVITY\n')
            file.write('%f %f %f\n' %(0.0, gravity, 0.0))
            file.write('*\n')

        #Fix the payload
        file.write('DISP\n')
        for i in range(self.capsule_n*(self.layer_n+1)):
            node_id = self.capsule_node[i] + 1
            if(i != layer_n//2 and i != layer_n//2 + 2*(layer_n+1)): #these are the connected nodes
                for freedom in range(1,7):
                    #Fix payload
                    file.write('%d %d 0.0\n' %(node_id, freedom))
            else:
                for freedom in range(1,4):
                    #Fix payload
                    file.write('%d %d 0.0\n' %(node_id, freedom))


        file.write('* symmetry planes\n')
        for freedom in range(3,7):
            file.write('%d thru %d step %d %d 0.0\n' %(1, (self.layer_n + 1)*self.canopy_n - self.layer_n, self.layer_n + 1,  freedom))
            file.write('%d thru %d step %d %d 0.0\n' %(1 + self.layer_n, (self.layer_n + 1)*self.canopy_n, self.layer_n + 1,  freedom))

        file.close()





#nPoints, xArray, yArray = curveRefine(num,x,y, cl,False, True)

#nPoints, xArray, yArray = straightLine(100)


canopy_type = 'line'
canopy_cl = 10
canopy_xScale, canopy_yScale = 1.,1.
layer_n = 4
layer_t = 1
k = 1

cable = ['straight','straight','straight','straight','straight','straight']
cable_cl = 100;
cable_k = 4
cable_r = 0.1
cable_joint = np.array([[0.,-35.826 ],[0., -43.372]])
phantom_offset = 0

capsule_type = 'square'
capsule_xScale, capsule_yScale = 0.3115, -44.721
canopy = [canopy_type, canopy_cl, canopy_xScale, canopy_yScale, layer_n, layer_t, k]
cable.extend([cable_cl, cable_k,cable_r, cable_joint, phantom_offset])
capsule =  [capsule_type, capsule_xScale, capsule_yScale]


parachute_mesh = Parachute(canopy, cable, capsule)

parachute_mesh._file_write_structure_top()

#Canopy_Matlaw = 'HyperElasticPlaneStress'
Canopy_Matlaw = 'PlaneStressViscoNeoHookean'
parachute_mesh._file_write_aeros_mesh_include(Canopy_Matlaw)

parachute_mesh._file_write_common_data_include()

parachute_mesh._file_write_embedded_surface_top()

parachute_mesh._file_write_surface_top()