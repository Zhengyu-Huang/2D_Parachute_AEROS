import numpy as np
import matplotlib.pyplot as plt
import math
import Folding ,Cable ,AeroShell, FluidGeo
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
#                                            1             2
#                                             \          /
#                                              \       /
#                                               \     /
#                                                \ A /
#                                                  |
#                                                  3
#                                                  |B
#                                                  |
#                                                  4
#                                                  |
#                                              *********
# The parachute has canopy part, capsult part and 6 cables
##################################################################################################################
    def __init__(self, canopy, capsule, cable):
        '''

        :param canopy: canopy type, canopy mesh size, canopy position (-xScale, xScale)*(0, yScale), layers=4, layer_t layer thickness
        :param capsule: capsule type capsule top at (-xScale, yScale)*(xScale, yScale)
        :param cable, cable type a list of 6 strings, cable mesh size,
        cable_k: node number on a cross-section of dressing surface
        cable_r: cable radius
        cable_joint: array([[Ax,Ay][Bx, By]])
        self.phantom_offset: 0 means the phantom surface is as long as the cable, 1 phantom surface has one offset on both side

        Attributes:
        self.canopy_node: an array of canopy node in self.coord
        self.canopy_n :  canopy node number
        self.capsule_node: an array of capsule node in self.coord
        self.capsule_n: capsule node number
        self.cables_node: an list of 6 list each contains cable node in self.structure.coord (cable order: from top to down from left to right)
        self.cables_n: cable node number list
        self.phantoms_node: an list of 6 list each contains phantom node in self.embedded.coord, phantom surface is a closed surface, which
        including a node at top and a node at the bottom.

        self.structure_coord: coordinates of structure nodes
        self.emebedded_coord: coordinates of embedded surface nodes

        self.As, self.Bs: cable i start point As[i], end point Bs[i]
        '''
        canopy_type, canopy_cl, canopy_xScale, canopy_yScale, self.layer_n, self.layer_t, *canopy_args = canopy
        *cable_type, self.cable_cl, self.cable_k,self.cable_r, self.cable_joint, self.phantom_offset = cable
        self.capsule_type, self.capsule_xScale, self.capsule_yScale, self.capsule_cl = capsule


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

        capsule_n, capsule_x, capsule_y, cable_attach = AeroShell.AeroShell(capsule_type, capsule_xScale, capsule_yScale,self.capsule_cl)
        self.capsule_n = capsule_n
        self.capsule_node = list(range((layer_n + 1) * canopy_n, (layer_n + 1) * canopy_n + (layer_n + 1)*capsule_n))

        #######################################################################################
        # Cable node
        #######################################################################################

        #we have 4 cables
        self.As = np.array([[canopy_x[0],canopy_y[0],layer_n//2*layer_t], [canopy_x[-1],canopy_y[-1],layer_n//2*layer_t],
                       [cable_joint[0,0],cable_joint[0,1],layer_n//2*layer_t], [cable_joint[1,0],cable_joint[1,1],layer_n//2*layer_t]],dtype= float)
        self.Bs = np.array([[cable_joint[0,0],cable_joint[0,1],layer_n//2*layer_t], [cable_joint[0,0],cable_joint[0,1],layer_n//2*layer_t],
                       [cable_joint[1,0],cable_joint[1,1],layer_n//2*layer_t], [capsule_x[cable_attach[1]],capsule_y[cable_attach[1]],layer_n//2*layer_t]], dtype = float)

        cables_n, cables_coord, phantoms_coord = Cable.Cables(cable_type, self.As, self.Bs, cable_cl, cable_k, cable_r)
        self.cables_n = cables_n
        #################################################################################################################
        self.cables_node = []
        cable_joint_node =[(layer_n + 1) * canopy_n + (layer_n + 1)*capsule_n, (layer_n + 1) * canopy_n + (layer_n + 1)*capsule_n + 1]
        cable_start = (layer_n + 1) * canopy_n + (layer_n + 1)*capsule_n + 1
        for i in range(4):
            self.cables_node.append(list(range(cable_start,cable_start + cables_n[i])))
            cable_start += cables_n[i] - 2

        self.structure_node_n =  cable_start+1
        self.cables_node[0][0], self.cables_node[0][-1] = self.canopy_node[layer_n // 2], cable_joint_node[0]
        self.cables_node[1][0], self.cables_node[1][-1] = self.canopy_node[(layer_n + 1) * (canopy_n - 1) + layer_n // 2], cable_joint_node[0]
        self.cables_node[2][0], self.cables_node[2][-1] = cable_joint_node[0], cable_joint_node[1]
        self.cables_node[3][0], self.cables_node[3][-1] = cable_joint_node[1], self.capsule_node[layer_n//2 + cable_attach[1]*(1 + layer_n)]

        self.phantoms_node = []
        phantom_start = (layer_n + 1) * canopy_n + (layer_n + 1)*capsule_n
        for i in range(4):
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
               z = self._layer_node_z(j)
               self.structure_coord[self.canopy_node[i*(layer_n+1) + j],:] = [canopy_x[i],canopy_y[i],z]
               self.embedded_coord[self.canopy_node[i * (layer_n + 1) + j], :] = [canopy_x[i], canopy_y[i], z]



        for i in range(capsule_n):
           for j in range(layer_n + 1):
               z = self._layer_node_z(j)
               self.structure_coord[self.capsule_node[i*(layer_n+1) + j],:] = [capsule_x[i],capsule_y[i],z]
               self.embedded_coord[self.capsule_node[i * (layer_n + 1) + j], :] = [capsule_x[i], capsule_y[i], z]

        for i in range(2):
            self.structure_coord[cable_joint_node[i],0:2],self.structure_coord[cable_joint_node[i],2] = cable_joint[i,:], layer_n//2*layer_t

        for cable_i in range(4):
            for j in range(phantom_offset, cables_n[cable_i] - phantom_offset):
                self.structure_coord[self.cables_node[cable_i][j], :] = cables_coord[cable_i][j,:]
                #first node for the beam surface
                self.embedded_coord[self.phantoms_node[cable_i][0], :] = cables_coord[cable_i][phantom_offset, :]
                for k in range(cable_k):

                    self.embedded_coord[self.phantoms_node[cable_i][(j-phantom_offset)*cable_k + k+1], :] = phantoms_coord[cable_i][j*cable_k +k,:]
                #last node for the beam surface
                self.embedded_coord[self.phantoms_node[cable_i][-1], :] = cables_coord[cable_i][-1-phantom_offset, :]

        self.thickness = 7.62e-5



    def _layer_node_z(self,j, eps = 1e-12):
        '''
        special issue about emebedded simulation for 2d, extrude structure a little
        :param j:
        :param eps:
        :return:
        '''
        layer_n, layer_t = self.layer_n, self.layer_t

        if j == 0:
            z = layer_t * j - eps
        elif j == layer_n:
            z = layer_t * j + eps
        else:
            z = layer_t * j
        return z


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
        for i in range(4):
            cable_node = self.cables_node[i]
            for j in range(len(cable_node) - 1):
                file.write('%d   %d   %d   %d\n' %(id, topo, cable_node[j] + 1, cable_node[j + 1] + 1))
                id += 1
        return id


    def _write_cable_frames(self, file):

        file.write('EFRAMES\n')
        ele_id = 2*self.layer_n*(self.capsule_n + self.canopy_n - 1) + 1
        for cable_i in range(4):# we have 4 cables
            cable_node = self.cables_node[cable_i] #take one cable
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
        for cable_i in range(4):
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

    def _write_canopy_surface(self,file,topo,start_id, list = None):
        n = self.canopy_n
        layer_n = self.layer_n
        id = start_id;

        if list is None:
            list = range(n-1)


        for i in list:
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
        id = self._write_capsule_surface(file,topo,id)
        file.write('*\n')

        file.close()

    def _file_write_embedded_surface_top(self, diffPorosity = False):
        file = open('embeddedSurface.top','w')

        file.write('Nodes nodeset\n')
        self._write_coord(file,'embedded')

        n = self.canopy_n
        id = 1
        if(diffPorosity):
            '''
            assume the parauchte canopy has 7 parts, fabric       gap         fabric       vent          fabric               gap                fabric
                                                           beta1*L     beta2*L       beta3*L    (1-beta3)*L      (1-beta2)*L        (1-beta1)*L
            '''

            beta1, beta2, beta3 = 0.2, 0.25,0.45
            # porous = 0.02
            set_2 = set()
            for i in range(n-1):
                if (i+ 1)/float(n-1)  <= beta1 or (beta2 <= i/float(n-1) and (i+1)/float(n-1) <= beta3):
                    set_2.update([i,n-2-i])
            # porous = 1.0
            set_1 = set(range(n-1)).difference(set_2)

            # porous = 1.0

            file.write('Elements StickMovingSurface_7 using nodeset\n')
            topo = 4;
            id = self._write_canopy_surface(file,topo,id, set_1)
        else:
            set_2 = None

        # porous = 0.02

        file.write('Elements StickMovingSurface_8 using nodeset\n')
        topo = 4;
        id = self._write_canopy_surface(file, topo, id, set_2)

        file.write('Elements StickMovingSurface_9 using nodeset\n')
        topo = 4
        id = self._write_capsule_surface(file, topo, id)

        file.write('Elements StickMovingSurface_10 using nodeset\n')
        topo = 4
        id = self._write_cable_surface(file,topo,id)




        file.close()

    def _file_write_capsule_surface_top(self):
        coord = self.embedded_coord
        layer_n, canopy_n, capsule_n = self.layer_n, self.canopy_n, self.capsule_n

        file = open('capsuleSurface.top','w')

        file.write('Nodes nodeset\n')
        id = 1
        for xyz in coord[(layer_n + 1) * canopy_n : (layer_n + 1) * canopy_n + (layer_n + 1) * capsule_n,:]:
            file.write('%d   %.15f  %.15f  %.15f \n' % (id, xyz[0], xyz[1], xyz[2]))
            id += 1


        id = 1
        topo = 4
        file.write('Elements StickMovingSurface_9 using nodeset\n')
        for i in range(self.capsule_n):
            for j in range(layer_n):
                # (layer_n+1)*i + j        (layer_n+1)*(i+1) + j
                # (layer_n+1)*i + j + 1    (layer_n+1)*(i+1) + j + 1
                file.write('%d   %d  %d  %d %d \n' % (id, topo, (layer_n + 1) * i + j + 1,
                                                                (layer_n + 1) * ((i + 1) % self.capsule_n) + j + 1 + 1,
                                                                (layer_n + 1) * ((i + 1) % self.capsule_n) + j + 1))
                id += 1
                file.write('%d   %d  %d  %d %d \n' % (id, topo, (layer_n + 1) * i + j + 1,
                                                                (layer_n + 1) * i + j + 1 + 1,
                                                                (layer_n + 1) * ((i + 1) % self.capsule_n) + j + 1 + 1))
                id += 1


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




        #capsule triangle around
        topo = 129
        id = self._write_capsule_surface(file,topo,id)


        # beam element 6
        topo = 6;
        id = self._write_cable_beam(file, topo, id)
        file.close()

    def _file_write_aeros_mesh_include(self, Canopy_Matlaw = 'HyperElasticPlaneStress', pressure = None, gravity = None):


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
        for i in range(4):
            end_ele += self.cables_n[i] - 1
        file.write('%d   %d   %d\n' %(start_ele, end_ele, cable_beam_attr))



        file.write('*\n')



        ###############################################################################################################################################################

        # Material specifies material
        # material id, ....
        youngsModulus = 1.0e9
        poissonRatio = 0.4
        density = 1153.4
        thickness = self.thickness
        file.write('MATERIAL\n')

        file.write('%d 0 %f %f %f 0 0 %f 0 0 0 0 0 0 0\n' %(canopy_attr, youngsModulus, poissonRatio, density, thickness))


        file.write('%d 0 %f %f %f 0 0 %f 0 0 0 0 0 0 0\n' %(capsule_attr, youngsModulus, poissonRatio, density, thickness))


        E = 12.9e9
        poissonRatio = 0.4
        rho = 1000
        cable_r = self.cable_r
        Ix = np.pi*cable_r**4/2.0
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


        if pressure is not None:
            file.write('PRESSURE\n')
            file.write('1 %d %f\n' %(2*self.layer_n*(self.canopy_n - 1), pressure))
            file.write('*\n')

        if gravity is not None:
            file.write('GRAVITY\n')
            file.write('%f %f %f\n' %(0.0, gravity, 0.0))
            file.write('*\n')

        #Fix the payload
        file.write('DISP\n')
        for i in range(self.capsule_n*(self.layer_n+1)):
            node_id = self.capsule_node[i] + 1
            if(i != layer_n//2 and i != layer_n//2 + (layer_n+1) and i != layer_n//2 + 2*(layer_n+1)): #these are the connected nodes
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
        '''
        file.write('* fix z coordinate in canopy\n')#because we do 2D simulation
        file.write('%d thru %d step %d %d 0.0\n' % (
        1, (self.layer_n + 1) * self.canopy_n, 1, 3))

        file.write('* fix z coordinate in cables\n')  # because we do 2D simulation
        endnode_n = len(self.structure_coord)
        file.write('%d thru %d step %d %d 0.0\n' % (
            (self.layer_n + 1) * (self.canopy_n + self.capsule_n) + 1, endnode_n, 1, 3))
        '''


        file.close()

    def _write_domain_geo(self, file_name, background, canopy, capsule, cable, refineOrNot = False):
        '''
        we have cl for small scale, cl_bg for backgroud mesh scale
        layer_cl for layer thickness

        :param file_name:
        :param refineOrNot:
        :return:
        '''




        cl, d_min_ball, d_max_ball, y_ball = canopy;
        cl_capsule, d_min_capsule, d_max_capsule = capsule
        cl_cable, d_min_cable, d_max_cable = cable

        cl_bg, x_l, y_l, x_r, y_r = background

        layer_n, canopy_n, capsule_n , cl_layer = self.layer_n, self.canopy_n, self.capsule_n, self.layer_t
        file = open(file_name, 'w')
        point_id = 1
        file.write('cl = %.15f;\n' %cl)
        file.write('cl_cable = %.15f;\n' % cl_cable)
        file.write('cl_capsule = %.15f;\n' % cl_capsule)
        file.write('cl_bg = %.15f;\n' %cl_bg)
        file.write('cl_layer = %.15f;\n' %cl_layer)

        if(refineOrNot):
            #point for field 1, ball
            point_id = FluidGeo.writeElem(file, 'Point', point_id, [0.0,y_ball,0.0 , 'cl'])
            # point for field 2, cable
            disturb = (np.random.uniform(0.,1.,2) - 0.5) * min(cl_capsule, cl_cable)/2.0# randomize the points location

            point_id = FluidGeo.writeElem(file, 'Point', point_id, [self.structure_coord[layer_n//2,0] + disturb[0],
                                                                    self.structure_coord[layer_n//2,1] + disturb[1],
                                                                    0.0, 'cl'])
            disturb = (np.random.uniform(0., 1., 2) - 0.5) * min(cl_capsule, cl_cable) / 2.0  # randomize the points location
            point_id = FluidGeo.writeElem(file, 'Point', point_id, [self.structure_coord[(layer_n + 1) * (canopy_n - 1) + layer_n // 2,0] + disturb[0],
                                                                    self.structure_coord[(layer_n + 1) * (canopy_n - 1) + layer_n // 2,1] + disturb[1],
                                                                    0.0, 'cl'])
            #beam node coord in self.structure_coord, row (layer_n + 1) * canopy_n + (layer_n + 1)*capsule_n to end
            for xyz in self.structure_coord[(layer_n + 1) * canopy_n + (layer_n + 1)*capsule_n :,:]:
                disturb = (np.random.uniform(0., 1., 2) - 0.5) * min(cl_capsule, cl_cable)/ 2.0  # randomize the points location
                point_id = FluidGeo.writeElem(file, 'Point', point_id, [xyz[0]+disturb[0], xyz[1]+disturb[1], 0.0, 'cl'])
            cable_id = point_id # cable id 2,3...cable_id-1


            capsule_n, capsule_x, capsule_y,_= AeroShell.AeroShell(self.capsule_type, self.capsule_xScale, self.capsule_yScale, 2*cl_capsule)

            capsule_n, capsule_x, capsule_y = Folding.curveRefine(capsule_n, capsule_x, capsule_y, 2*cl_capsule, closeOrNot = True)

            for i in range(capsule_n):
                disturb = (np.random.uniform(0., 1., 2) - 0.5) * min(cl_capsule, cl_cable) / 2.0  # randomize the points location
                point_id = FluidGeo.writeElem(file, 'Point', point_id, [capsule_x[i]+disturb[0], capsule_y[i]+disturb[1], 0.0, 'cl'])
            capsule_id = point_id # capsule id cable_id, cable_id +1 .. capsule_id-1

            FluidGeo.writeMeshSize(file, 'Attractor',1, [1])
            FluidGeo.writeMeshSize(file, 'Attractor', 2, range(2,cable_id) )
            FluidGeo.writeMeshSize(file, 'Attractor', 3, range(cable_id, capsule_id))


            #file, type, field_id, IField, LcMin, LcMax, DistMin, DistMax, Sigmoid
            FluidGeo.writeMeshSize(file, 'Threshold', 4, 1, 'cl', 'cl_bg',           d_min_ball,         d_max_ball, 0) # ball
            FluidGeo.writeMeshSize(file, 'Threshold', 5, 2, 'cl_cable', 'cl_bg',     d_min_cable,        d_max_cable, 0) # cable
            FluidGeo.writeMeshSize(file, 'Threshold', 6, 3, 'cl_capsule', 'cl_bg',   d_min_capsule,      d_max_capsule, 0) # capsule
            FluidGeo.writeMeshSize(file, 'Min', 7, [4,5,6])

        #Background Mesh, the domain is a square (x_l, x_r)*(y_l, y_r)
        FluidGeo.backgroundMesh(file, 'cube', x_l, x_r, y_l, y_r, layer_n, point_id)
        file.close()




if __name__ == "__main__":
    type = "line"

    if(type == "line"):

        canopy_type = 'line'
        canopy_cl = 0.05
        capsule_cl = 0.002#0.001
        canopy_xScale, canopy_yScale = 10.674,1.
        layer_n = 4
        layer_t = 0.1#0.01
        k = 1
        canopy = [canopy_type, canopy_cl, canopy_xScale, canopy_yScale, layer_n, layer_t, k]

        capsule_type = 'AFL'
        capsule_xScale, capsule_yScale = 0.3115, -44.721
        capsule =  [capsule_type, capsule_xScale, capsule_yScale, capsule_cl]


        cable = ['straight','straight','straight','straight']
        cable_cl = 0.05;
        cable_k = 4
        cable_r = 0.00215
        cable_joint = np.array([[0.,-35.826 ],[0., -43.372]])
        phantom_offset = 1
        cable.extend([cable_cl, cable_k,cable_r, cable_joint, phantom_offset])


        parachute_mesh = Parachute(canopy, capsule, cable)


        parachute_mesh._file_write_structure_top()

        Canopy_Matlaw = 'HyperElasticPlaneStress'
        #Canopy_Matlaw = 'PlaneStressViscoNeoHookean'
        parachute_mesh._file_write_aeros_mesh_include(Canopy_Matlaw)

        parachute_mesh._file_write_common_data_include()

        parachute_mesh._file_write_embedded_surface_top(False)

        parachute_mesh._file_write_capsule_surface_top()

        parachute_mesh._file_write_surface_top()

        ##################Fluid Mesh

        x_l = -100.
        x_r = -x_l
        y_l = 150.
        y_r = -100.
        cl_bg = 5.0
        mesh_background = [cl_bg, x_l,y_l,x_r, y_r]

        Dist_Max = min(abs(x_r - x_l)/2.0, abs(y_l - y_r)/2.0)/2.0
        R = 12
        mesh_canopy = [canopy_cl, R, Dist_Max,1.0]
        mesh_capsule = [capsule_cl, 2 * capsule_cl, Dist_Max]
        mesh_cable = [2*cable_cl, R, Dist_Max]

        parachute_mesh._write_domain_geo('domain.geo',  mesh_background, mesh_canopy, mesh_capsule, mesh_cable, True)

    elif(type == "hat"):
        canopy_type = 'hat'
        canopy_cl = 0.05
        capsule_cl = 0.001
        canopy_xScale, canopy_yScale = 0.1, 10.
        layer_n = 4
        layer_t = 0.025
        k = 1
        canopy = [canopy_type, canopy_cl, canopy_xScale, canopy_yScale, layer_n, layer_t, k]

        capsule_type = 'AFL'
        capsule_xScale, capsule_yScale = 0.3115, -44.721
        capsule = [capsule_type, capsule_xScale, capsule_yScale, capsule_cl]

        cable = ['straight', 'straight', 'straight', 'straight']
        cable_cl = 0.05;
        cable_k = 4
        cable_r = 0.00215
        cable_joint = np.array([[0., -35.826], [0., -43.372]])
        phantom_offset = 1
        cable.extend([cable_cl, cable_k, cable_r, cable_joint, phantom_offset])

        parachute_mesh = Parachute(canopy, capsule, cable)

        parachute_mesh._file_write_structure_top()

        Canopy_Matlaw = 'HyperElasticPlaneStress'
        # Canopy_Matlaw = 'PlaneStressViscoNeoHookean'
        parachute_mesh._file_write_aeros_mesh_include(Canopy_Matlaw)

        parachute_mesh._file_write_common_data_include()

        parachute_mesh._file_write_embedded_surface_top(False)

        parachute_mesh._file_write_capsule_surface_top()

        parachute_mesh._file_write_surface_top()

        ##################Fluid Mesh

        x_l = -100.
        x_r = -x_l
        y_l = 150.
        y_r = -80.
        mesh_background = [cl_bg, x_l, y_l, x_r, y_r]

        Dist_Max = min(abs(x_r - x_l) / 2.0, abs(y_l - y_r) / 2.0) / 2.0
        R = 12
        mesh_canopy = [canopy_cl, R, Dist_Max, 1.0]
        mesh_capsule = [capsule_cl, 2 * capsule_cl, Dist_Max]
        mesh_cable = [cable_cl, R, Dist_Max]

        parachute_mesh._write_domain_geo('domain.geo', mesh_background, mesh_canopy, mesh_capsule, mesh_cable, True)

    else:
        canopy_type = 'line'
        canopy_cl = 0.1
        capsule_cl = 0.1
        cable_cl = 0.1;
        canopy_xScale, canopy_yScale = 10.674, 1.
        layer_n = 4
        layer_t = 0.1
        k = 1
        canopy = [canopy_type, canopy_cl, canopy_xScale, canopy_yScale, layer_n, layer_t, k]

        capsule_type = 'AFL'
        capsule_xScale, capsule_yScale = 0.3115, -44.721
        capsule = [capsule_type, capsule_xScale, capsule_yScale, capsule_cl]

        cable = ['straight', 'straight', 'straight', 'straight']

        cable_k = 4
        cable_r = 0.00215
        cable_joint = np.array([[0., -35.826], [0., -43.372]])
        phantom_offset = 1
        cable.extend([cable_cl, cable_k, cable_r, cable_joint, phantom_offset])

        parachute_mesh = Parachute(canopy, capsule, cable)

        parachute_mesh._file_write_structure_top()

        Canopy_Matlaw = 'HyperElasticPlaneStress'
        # Canopy_Matlaw = 'PlaneStressViscoNeoHookean'
        parachute_mesh._file_write_aeros_mesh_include(Canopy_Matlaw,-1200)

        parachute_mesh._file_write_common_data_include()

        parachute_mesh._file_write_embedded_surface_top(False)

        parachute_mesh._file_write_capsule_surface_top()

        parachute_mesh._file_write_surface_top()

        ##################Fluid Mesh

        x_l = -100.
        x_r = -x_l
        y_l = 150.
        y_r = -100.
        cl_bg = 5.0
        mesh_background = [cl_bg, x_l, y_l, x_r, y_r]

        Dist_Max = min(abs(x_r - x_l) / 2.0, abs(y_l - y_r) / 2.0) / 2.0
        R = 12
        mesh_canopy = [canopy_cl, R, Dist_Max, 1.0]
        mesh_capsule = [capsule_cl, 2 * capsule_cl, Dist_Max]
        mesh_cable = [2 * cable_cl, R, Dist_Max]

        parachute_mesh._write_domain_geo('domain.geo', mesh_background, mesh_canopy, mesh_capsule, mesh_cable, True)