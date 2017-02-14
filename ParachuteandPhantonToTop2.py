import numpy as np
import matplotlib.pyplot as plt
import math
#This is a mesh generator for quasi-1D parachte
#this is the one with matcher dressing techniques for the cables(suspension lines)

#This is a sin curve
#int  nPoints, number of nodes
#int* xArray, x coordinates of these nodes
#int* yArray, y coordinates of these nodes
def sinCurve(plotOrNot):
    nPoints = 800;

    A = 0.2;
    k = 9
    xArray = np.linspace(-0.5,0.5,num=nPoints)
    yArray = A*np.sin((2*k)*np.pi*(xArray + 0.5)-0.5*np.pi)
    if(plotOrNot):
        plt.plot(xArray, yArray,'-*')
        plt.ylim([-0.5,1.5])
        plt.xlim([-1,1])
        plt.show()
    return nPoints, xArray, yArray

#This is a straight line
#int  nPoints, number of nodes
#int* xArray, x coordinates of these nodes
#int* yArray, y coordinates of these nodes
def straightLine(nPoints):
    xArray = np.linspace(-0.5,0.5,num=nPoints)
    yArray = 0*xArray
    return nPoints, xArray, yArray



#This is a hat shape function
#int  nPoints, number of nodes
#int* xArray, x coordinates of these nodes
#int* yArray, y coordinates of these nodes
def hatShape():
    k = 25
    nPoints = 301;
    xArray = np.zeros(nPoints)
    yArray = np.zeros(nPoints)

    xArray[0:k+1] = np.linspace(0,0.25,num=k+1)
    yArray[0:k+1] = 0

    xArray[k:2*k+1] = 0.25
    yArray[k:2*k+1] = np.linspace(0,0.25,num=k+1)

    xArray[2*k:3*k+1] = np.linspace(0.25,0,num=k+1)
    yArray[2*k:3*k+1] = 0.25

    xArray[3*k:4*k+1] = 0
    yArray[3*k:4*k+1] = np.linspace(0.25,0.5,num=k+1)

    xArray[4*k:8*k+1] = np.linspace(0,1,num=4*k+1)
    yArray[4*k:8*k+1] = 0.5

    xArray[8*k:9*k+1] = 1
    yArray[8*k:9*k+1] = np.linspace(0.5,0.25,num=k+1)

    xArray[9*k:10*k+1] = np.linspace(1,0.75,num=k+1)
    yArray[9*k:10*k+1] = 0.25

    xArray[10*k:11*k+1] = 0.75
    yArray[10*k:11*k+1] = np.linspace(0.25,0,num=k+1)

    xArray[11*k:12*k+1] = np.linspace(0.75,1,num=k+1)
    yArray[11*k:12*k+1] = 0

    return nPoints, xArray, yArray

def candle():
    xx = np.array([-0.5, -0.07, -0.05, -0.45, -0.5, -0.05, -0.04, -0.5, -0.48, -0.38, -0.375, -0.25, -0.245, -0.13, -0.125, -0.05, -0.045, 0.045, 0.05, 0.125, 0.13, 0.245, 0.25,
          0.375, 0.38, 0.48, 0.5, 0.04, 0.05, 0.5, 0.45, 0.05, 0.07, 0.5]);
    yy = np.array([0, 0, 0.125, 0.125, 0.25, 0.25, 0.375, 0.375, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1, 0.375, 0.375, 0.25, 0.25, 0.125, 0.125, 0, 0]);

    num = len(xx)
    return num, xx, yy



def hilbertCurveRecursive(n):
    ''' Generate Hilbert curve . 'n' must be a power of two. '''
    # recursion base
    if n == 1:
        return np.zeros((1, 1), np.int32)
    # make (n/2, n/2) index
    t = hilbertCurveRecursive(n//2)
    # flip it four times and add index offsets
    a = np.flipud(np.rot90(t))
    b = t + t.size
    c = t + t.size*2
    d = np.flipud(np.rot90(t, -1)) + t.size*3
    # and stack four tiles into resulting array
    return np.vstack(map(np.hstack, [[a, b], [d, c]]))
def hilbertCurve(order,scaleX, scaleY):

    n = 2**order;
    idx = hilbertCurveRecursive(n)
    idx = np.rot90(idx,-1)
    idx = np.fliplr(idx)
    y, x = np.indices(idx.shape).reshape(2, -1) /(n-1.0)
    x = x - 0.5;
    x[idx.ravel()], y[idx.ravel()] = x.copy(), y.copy()



    return x.size, x*scaleX, y*scaleY

def refinedHilbertCurve(order, cl, scale, plotOrNot):

    num,x,y = hilbertCurve(order,scale)
    ds = np.sqrt((y[1]-y[0])**2 + (x[1] - x[0])**2)
    n = max(np.ceil(ds/cl), 1) #n small segments on each segment
    nPoints = int((num-1)*n + 1)
    xArray = np.zeros(nPoints)
    yArray = np.zeros(nPoints)
    k1 = 0;
    k2 = 0;
    for i in range(num-1):

        k2 = k1 + n

        xArray[k1:k2+1] = np.linspace(x[i],x[i+1],num=n+1)
        yArray[k1:k2+1] = np.linspace(y[i],y[i+1],num=n+1)

        k1 = k2;

    if(plotOrNot):
        plt.plot(xArray, yArray,'-*')
        plt.ylim([-0.5,1.5])
        plt.xlim([-1,1])
        plt.show()
    return nPoints, xArray, yArray







def sFolding(k,scaleX,scaleY):
    idx = np.zeros([4*k,4*k], np.int32)
    for i in range(k):
        idx[4*i  , 1:4*k] = range(i*(8*k), i*(8*k) + 4*k - 1)
        idx[4*i  , 0] = 4*k*(4*k-2*i)-1

        idx[4*i+1, 4*k-1] = i*(8*k) + 4*k-1
        idx[4*i+1, 0:4*k-1] = range(4*k*(4*k-2*i)-2, 4*k*(4*k-2*i) - 4*k-1, -1 )

        idx[4*i+2, 4*k-1] = i*(8*k) + 4*k
        idx[4*i+2, 0:4*k-1] = range(4*k*(4*k-2*i)-8*k+1, 4*k*(4*k-2*i) - 4*k)

        idx[4*i+3, 1:4*k] = range((i+1)*(8*k)-1,i*(8*k) + 4*k , -1)
        idx[4*i+3,0] = 4*k*(4*k-2*i)-8*k


    #idx = np.rot90(idx,-1)
    idx = np.fliplr(idx)
    y, x = np.indices(idx.shape).reshape(2, -1) /(4*k-1.0)
    x = x - 0.5;
    x[idx.ravel()], y[idx.ravel()] = x.copy(), y.copy()
    num= x.size
    x = x[4*k-2:]
    y =y[4*k-2:]

    return num-4*k+2, x*scaleX, y*scaleY




def AFL(xa, ya, xb, yb,k = 10, plotOrNot=True):
    x4 = 2.25
    x3 = x4 - 1.012*np.tan(36.9*np.pi/180)
    x2 = x3 - 0.506*np.tan(59*np.pi/180)
    x1 = x2 - 0.5*np.tan(33.95*np.pi/180)
    y1,y2,y3,y4 = 0, -0.5 , -0.5-0.506, -0.5-0.506-1.012

    o_x,o_y = 0.0, y4 - 2.25*np.tan(20*np.pi/180) + 0.5* 2.25/np.cos(20*np.pi/180)

    nose_x, nose_y = o_x + 1.125*np.cos(np.linspace(-70,-110,k)*np.pi/180), o_y + 1.125*np.sin(np.linspace(-70,-110,k)*np.pi/180)

    x_p = np.append(np.append([-x1,x1,x2,x3,x4,],nose_x),[-x4,-x3,-x2])
    y_p = np.append(np.append([y1,y1,y2,y3,y4], nose_y), [y4,y3,y2])

    num = k+8
    scale = (xb-xa)/(2*x1)
    x,y = x_p*scale, y_p *scale
    x , y = x + xa - x[0], y + ya - y[0]
    print('smallest y coord is %f',np.amin(y))
    if (plotOrNot):
        plt.fill(x, y,edgecolor='r', fill=False)
        plt.show()


    return num, x, y


def squareCurve(xa, ya, xb, yb):
    h = xb - xa

    x = np.array([xa, xb, xb, xa])
    y = np.array([ya, ya, ya-h, ya-h])
    num = 4
    return num, x, y

def curveRefine(num, xx, yy,cl, closeOrNot, plotOrNot):
    segNum = num if closeOrNot else num - 1

    numArray = np.zeros(segNum + 1,np.int32)

    for i in range(segNum):
        ds = np.sqrt((yy[(i+1)%num]-yy[i])**2 + (xx[(i+1)%num] - xx[i])**2)
        n = max(np.ceil(ds/cl), 1) #n small segments on each segment
        numArray[i+1] = numArray[i] + n;

    nPoints = numArray[-1]+1
    xArray = np.zeros(nPoints)
    yArray = np.zeros(nPoints)

    for i in range(segNum):
        xArray[numArray[i]:(numArray[i+1]+1)] = np.linspace(xx[i],xx[(i+1)%num],num=numArray[i+1] - numArray[i] + 1)
        yArray[numArray[i]:(numArray[i+1]+1)] = np.linspace(yy[i],yy[(i+1)%num],num=numArray[i+1] - numArray[i] + 1)

    if(plotOrNot):
        plt.plot(xArray, yArray,'-*')
        plt.ylim([-0.5,1.5])
        plt.xlim([-1,1])
        plt.show()
    return nPoints, xArray, yArray

###########################################################################################################################
###########################################################################################################################
#Start to generate AEROS and Mathcer input file






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

    @staticmethod
    def _compute_phantom_coordinates(A,B,n,k,cable_r):
        # cable AB
        #A = np.array([0.0,0.0,0.0],dtype=float)
        #B = np.array([0.0,0.0,1.0],dtype=float)

        #n = 20 # number of nodes on the cable AB
        #k = 6  # number of nodes on the phantom skeleton
        #cable_r = 0.1 # radius of cable
        #choose A cross e_1(1,0,0) as the first rigid phantom edge


        X = np.empty(shape=[n,3],dtype=float)
        for i in range(3):
            X[:,i]    = np.linspace(A[i],B[i],num=n)

        nx,ny,nz = dir = (B-A)/np.linalg.norm(B-A) # direction of AB
        e1 = np.array([1.0,0.0,0.0],dtype=float)

        theta = 2*np.pi/k

        #Rotation matrix on https://en.wikipedia.org/wiki/Rotation_matrix
        R = np.array([[np.cos(theta) + nx*nx*(1-np.cos(theta))   ,nx*ny*(1-np.cos(theta)) - nz*np.sin(theta)   ,nx*nz*(1-np.cos(theta)) + ny*np.sin(theta)],
                      [ny*nx*(1-np.cos(theta))+nz*np.sin(theta)  ,np.cos(theta) + ny*ny*(1-np.cos(theta))      ,ny*nz*(1-np.cos(theta)) - nx*np.sin(theta)],
                      [nz*nx*(1-np.cos(theta))-ny*np.sin(theta)  ,nz*ny*(1-np.cos(theta)) + nx*np.sin(theta)   ,np.cos(theta) + nz*nz*(1-np.cos(theta)) ]],dtype=float)

        phontom_dr = np.empty(shape=[k,3],dtype=float)
        phontom_dr[0,:] = cable_r*np.cross(dir,e1)

        for i in range(1,k):
            phontom_dr[i,:] = np.dot(R,phontom_dr[i-1,:])

        coord = np.empty(shape=[n*(k+1),3],dtype=float)
        for i in range(n):
            coord[i*(k+1),:] = X[i,:]
            for j in range(1,k+1):
                coord[i*(k+1)+j,:] = X[i,:] + phontom_dr[j-1,:]
        return coord



    def __init__(self,canopy_n, canopy_x, canopy_y, cable_n, cable_k, cable_r, layer_n,layer_t, capsule_x, capsule_y = -0.5):
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
        '''




        self.canopy_n = canopy_n
        self.cable_n = cable_n
        self.cable_k = cable_k
        self.layer_n = layer_n
        self.cable_r = cable_r
        self.capsule_y = capsule_y

        self.canopy_node = np.array(range((layer_n+1)* canopy_n),dtype=int)



        #Left suspension line, from bottom to top
        self.cable1_node = np.array(range((layer_n+1)*canopy_n, (layer_n+1)*canopy_n + (cable_k +1)*(cable_n-2) + 2),dtype=int)
        self.cable1_node[-1]  = layer_n/2




        #Right suspension line, from bottom to top
        self.cable2_node = np.array(range((layer_n+1)*canopy_n + (cable_k +1)*(cable_n-2) + 1,  (layer_n+1)*canopy_n + 2*(cable_k +1)*(cable_n-2) + 3),dtype=int)
        self.cable2_node[-1]  = (layer_n + 1)*canopy_n - layer_n/2 - 1

        #Capsule nodes

        capsule_xl, capsule_yl, capsule_xr, capsule_yr = capsule_x, capsule_y, -capsule_x, capsule_y
        #####We have two kinds of capsules, squeare and AFL
        capsule_n, capsule_x, capsule_y = AFL(capsule_xl, capsule_yl, capsule_xr, capsule_yr)
        #capsule_n, capsule_x, capsule_y = squareCurve(capsule_xl, capsule_yl, capsule_xr, capsule_yr)

        self.capsule_n = capsule_n
        capsule_con_l = self.capsule_con_l = 2 #todo check magic number
        capsule_con_r = self.capsule_con_r = 7 #todo check magic number

        self.capsule_node = np.array(range((layer_n+1)*canopy_n + 2*(cable_k +1)*(cable_n-2) + 2, (layer_n+1)*canopy_n + 2*(cable_k +1)*(cable_n-2) + 2 + (layer_n+1)*capsule_n) ,dtype=int)

        # node capsule_con_l and capsule_con_r are these connected node
        self.capsule_node[capsule_con_l] = (layer_n+1)*canopy_n
        self.capsule_node[capsule_con_l + 1 :] -= 1
        self.capsule_node[capsule_con_r] = (layer_n+1)*canopy_n + (cable_k +1)*(cable_n-2) + 1
        self.capsule_node[capsule_con_r + 1 :] -= 1

        node_n =  (layer_n+1)*canopy_n + 2*(cable_k +1)*(cable_n-2) + (layer_n+1)*capsule_n
        self.node_n = node_n

        ##################################################################################
        # Finish the node number part
        ##################################################################################

        ##################################################################################
        # Start the coordinate part
        ##################################################################################
        self.coord = np.empty(shape=[node_n,3],dtype=float)
        for i in range(canopy_n):
           for j in range(layer_n + 1):
               self.coord[self.canopy_node[i*(layer_n+1) + j],:] = [canopy_x[i],canopy_y[i],layer_t*j]

        self.beam1_start_coord = np.array([capsule_xl, capsule_yl, layer_n//2*layer_t], dtype=float) #This is the capsule_con_l node
        self.beam1_end_coord = self.coord[layer_n//2,:]
        coord = Parachute._compute_phantom_coordinates(self.beam1_start_coord,  self.beam1_end_coord,  cable_n,   cable_k,   cable_r)
        self.coord[self.cable1_node[0],:] = self.beam1_start_coord
        for i in range(1,cable_n-1):
            self.coord[self.cable1_node[(i-1)*(cable_k+1) + 1],:] = coord[i*(cable_k+1),:]
            for j in range(cable_k):
                self.coord[self.cable1_node[(i-1)*(cable_k+1)+j + 2],:] = coord[i*(cable_k+1) + j + 1,:]

        self.beam2_start_coord = np.array([capsule_xr, capsule_yr, layer_n/2*layer_t], dtype=float)   #This is the capsule_con_r node
        self.beam2_end_coord = self.coord[(layer_n + 1)*canopy_n - layer_n//2 - 1,:]

        coord = Parachute._compute_phantom_coordinates(self.beam2_start_coord,self.beam2_end_coord,cable_n,cable_k,cable_r)
        self.coord[self.cable2_node[0],:] = self.beam2_start_coord
        for i in range(1,cable_n-1):
            self.coord[self.cable2_node[(i - 1)*(cable_k+1) + 1],:] = coord[i*(cable_k+1),:]
            for j in range(cable_k):
                self.coord[self.cable2_node[(i - 1)*(cable_k+1)+j + 2],:] = coord[i*(cable_k+1) + j + 1,:]

        '''
        for i in range(3):
            for j in range(3):
                self.coord[self.capsule_node[i*3 + j],:] = [(i-1)*layer_n//2 *layer_t ,capsule_y , j* layer_n//2 *layer_t ]
        '''



        for i in range(capsule_n):
           for j in range(layer_n + 1):
               self.coord[self.capsule_node[i*(layer_n+1) + j],:] = [capsule_x[i],capsule_y[i],layer_t*j]


        self.thickness = 7.62e-5


        '''
        Attribute
        structure_mask/ embedded_mask
        0 1  2 3 4 ... node_n
        0 -1 1 2 3 ....
        if node is not in structure/embeddedsurface its id is 0, otherwise it is its new id
        '''


        self.structure_mask = np.ones(node_n,dtype=int)

        for i in range(1,self.cable_n - 1):
            for k in range(self.cable_k):
             self.structure_mask[self.cable1_node[1 + i*(cable_k+1) - cable_k + k]] = -1
             self.structure_mask[self.cable2_node[1 + i*(cable_k+1) - cable_k + k]] = -1
        id = 0
        for i in range(node_n):
            if(self.structure_mask[i] > 0):
                self.structure_mask[i] = id
                id +=1


        self.embeddedsurface_mask = np.ones(node_n,dtype=int)
        for i in range(2, self.cable_n - 2):
            self.embeddedsurface_mask[self.cable1_node[i*(cable_k + 1) - cable_k]] = -1
            self.embeddedsurface_mask[self.cable2_node[i*(cable_k + 1) - cable_k]] = -1

        id = 0
        for i in range(node_n):
            if (self.embeddedsurface_mask[i] > 0):
                self.embeddedsurface_mask[i] = id
                id += 1








    def _write_coord(self,file,mask=None):
        '''
        Write the node coords on the real structure/embedded surface, not include phantom element nodes/beam-inside nodes
        To write embedded surface, mask  =  self.embeddedsurface_mask
        To write structure, mask  =  self.structure_mask
        :param file: output file
        :return: None
        '''

        if mask is None:
            mask = np.arange(self.node_n)
        coord  = self.coord

        id = 1
        for i in  range(self.node_n):
            if mask[i] == -1:
                continue
            xyz = coord[i, :]
            assert(id == mask[i]+1)
            file.write('%d   %.15f  %.15f  %.15f \n' % (id, xyz[0], xyz[1], xyz[2]))
            id += 1






    def _write_cable_beam(self,file,topo,start_id,mask = None):
        '''
        Write the beam element
        :param file: output file
        :param topo: element topology
        :param start_id: start node id in the output file
        :return:
        '''
        if mask is None:
            mask = np.arange(self.node_n)



        cable_k = self.cable_k
        id = start_id
        file.write('%d   %d   %d   %d\n' %(id, topo,  mask[self.cable1_node[0]]+1, mask[self.cable1_node[1]]+1))
        id += 1
        for i in range(self.cable_n - 2):
            file.write('%d   %d   %d   %d\n' %(id, topo,  mask[self.cable1_node[i*(cable_k+1) + 1]]+1, mask[self.cable1_node[(i+1)*(cable_k+1) + 1]]+1))
            id +=1

        file.write('%d   %d   %d   %d\n' %(id, topo,  mask[self.cable2_node[0]]+1, mask[self.cable2_node[1]]+1))
        id += 1
        for i in range(self.cable_n - 2):
            file.write('%d   %d   %d   %d\n' %(id,  topo, mask[self.cable2_node[i*(cable_k+1) + 1]]+1, mask[self.cable2_node[(i+1)*(cable_k+1) + 1]]+1))
            id += 1
        return id

    def _write_cable_rigid_beam(self,file,topo,start_id):
        n = self.cable_n
        k = self.cable_k
        id = start_id
        for i in range(n - 2):
            for j in range(1,k+1):
                file.write('%d   %d  %d  %d\n' %(id, topo,  self.cable1_node[(k+1)*i + 1]+1, self.cable1_node[(k+1)*i+j + 1]+1))
                id +=1
        for i in range(n - 2):
            for j in range(1,k+1):
                file.write('%d   %d  %d  %d\n' %(id, topo,  self.cable2_node[(k+1)*i + 1]+1, self.cable2_node[(k+1)*i+j + 1]+1))
                id +=1
        return id

    @staticmethod
    def _write_cable_surface_helper(file, topo, k, n, cable_node, start_id,mask = None):
        '''
        :param file: output file name
        :param topo: material topology
        :param k: cable_k
        :param n: cable_n
        :param cable_node: cable node number array
        :param start_id: start node id in the output file
        :return:
        '''

        if mask is None:
            mask = np.arange(self.node_n)

        id = start_id
        temp = range(1,k+1)
        for i in range(n-3):
            for j in range(k):
                file.write('%d   %d  %d  %d  %d\n' %(id, topo,  mask[cable_node[i*(k+1)+temp[j-1] + 1]]+1, mask[cable_node[i*(k+1)+temp[j] + 1]]+1,
                                                     mask[cable_node[(i+1)*(k+1)+temp[j-1] + 1]]+1))
                id +=1
                file.write('%d   %d  %d  %d  %d\n' %(id, topo,  mask[cable_node[i*(k+1)+temp[j] + 1]]+1, mask[cable_node[(i+1)*(k+1)+temp[j] + 1]]+1,
                                                     mask[cable_node[(i+1)*(k+1)+temp[j-1] + 1]]+1))
                id +=1


        #phantom triangle at bottom
        temp = range(1,k+1)
        i = 1
        for j in range(k):
            file.write('%d   %d  %d  %d  %d\n' %(id, topo,  mask[cable_node[i]]+1, mask[cable_node[i + temp[j]]] + 1, mask[cable_node[i+temp[j-1]]]+1))
            id +=1


        #phantom triangle at top
        i = (n-3)*(k+1) + 1
        for j in range(k):
            file.write('%d   %d  %d  %d  %d\n' %(id, topo,  mask[cable_node[i]]+1, mask[cable_node[i + temp[j-1]]]+1, mask[cable_node[i+temp[j]]]+1))
            id +=1
        return id


    def _write_cable_surface(self,file,topo,start_id,mask=None):
        if mask is None:
            mask = np.arange(self.node_n)

        id = Parachute._write_cable_surface_helper(file, topo, self.cable_k, self.cable_n, self.cable1_node, start_id,mask)
        id = Parachute._write_cable_surface_helper(file, topo, self.cable_k, self.cable_n, self.cable2_node, id,mask)
        return id

    def _write_canopy_surface(self,file,topo,start_id, special = True):
        n = self.canopy_n
        layer_n = self.layer_n
        canopy_node = self.canopy_node
        id = start_id;
        if(not special):
            for i in range(n - 1):
                for j in range(layer_n):
                    # (layer_n+1)*i + j + 1    (layer_n+1)*(i+1) + j + 1
                    # (layer_n+1)*i + j        (layer_n+1)*(i+1) + j
                    file.write('%d   %d  %d  %d %d \n' %(id, topo,  canopy_node[(layer_n+1)*i + j]+1, canopy_node[(layer_n+1)*(i+1) + j]+1, canopy_node[(layer_n+1)*(i+1) + j+1]+1))
                    id += 1
                    file.write('%d   %d  %d  %d %d \n' %(id, topo,  canopy_node[(layer_n+1)*i + j]+1, canopy_node[(layer_n+1)*(i+1) + j + 1]+1, canopy_node[(layer_n+1)*i + j + 1]+1))
                    id += 1
        else:
            for i in range(n-1):
                if(i <= (n-1)/2):
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

    def _write_capsule_surface(self,file,topo,id,mask):


        if mask is None:
            mask = np.arange(self.node_n)

        capsule_node =self.capsule_node

        '''
        file.write('%d   %d  %d  %d %d \n' %(start_id,   topo,  mask[capsule_node[0]] + 1, mask[capsule_node[3]] + 1, mask[capsule_node[4]] + 1))
        file.write('%d   %d  %d  %d %d \n' %(start_id+1, topo,  mask[capsule_node[0]] + 1, mask[capsule_node[4]] + 1, mask[capsule_node[1]] + 1))
        file.write('%d   %d  %d  %d %d \n' %(start_id+2, topo,  mask[capsule_node[1]] + 1, mask[capsule_node[4]] + 1, mask[capsule_node[2]] + 1))
        file.write('%d   %d  %d  %d %d \n' %(start_id+3, topo,  mask[capsule_node[2]] + 1, mask[capsule_node[4]] + 1, mask[capsule_node[5]] + 1))
        file.write('%d   %d  %d  %d %d \n' %(start_id+4, topo,  mask[capsule_node[3]] + 1, mask[capsule_node[6]] + 1, mask[capsule_node[4]] + 1))
        file.write('%d   %d  %d  %d %d \n' %(start_id+5, topo,  mask[capsule_node[4]] + 1, mask[capsule_node[6]] + 1, mask[capsule_node[7]] + 1))
        file.write('%d   %d  %d  %d %d \n' %(start_id+6, topo,  mask[capsule_node[4]] + 1, mask[capsule_node[7]] + 1, mask[capsule_node[8]] + 1))
        file.write('%d   %d  %d  %d %d \n' %(start_id+7, topo,  mask[capsule_node[4]] + 1, mask[capsule_node[8]] + 1, mask[capsule_node[5]] + 1))
        return start_id + 9
        '''
        layer_n = self.layer_n
        for i in range(self.capsule_n):
            for j in range(layer_n):
                # (layer_n+1)*i + j + 1    (layer_n+1)*(i+1) + j + 1
                # (layer_n+1)*i + j        (layer_n+1)*(i+1) + j
                file.write('%d   %d  %d  %d %d \n' % (id, topo, mask[capsule_node[(layer_n + 1) * i + j]] + 1, mask[capsule_node[(layer_n + 1) * ((i + 1)%self.capsule_n) + j]] + 1,
                mask[capsule_node[(layer_n + 1) * ((i + 1)%self.capsule_n) + j + 1]] + 1))
                id += 1
                file.write('%d   %d  %d  %d %d \n' % (id, topo, mask[capsule_node[(layer_n + 1) * i + j]] + 1, mask[capsule_node[(layer_n + 1) * ((i + 1)%self.capsule_n) + j + 1]] + 1,
                mask[capsule_node[(layer_n + 1) * i + j + 1]] + 1))
                id += 1
        return id




    def _file_write_structure_top(self):
        file = open('structure.top','w')
        id = 1
        file.write('Nodes nodeset\n')
        self._write_coord(file,self.structure_mask)


        file.write('Elements canopy using nodeset\n')
        topo = 4;
        id = self._write_canopy_surface(file,topo,id)


        file.write('Elements cable beam using nodeset\n')
        topo = 1
        id = self._write_cable_beam(file,topo,id,self.structure_mask)


        file.write('Elements payload using nodeset\n')
        topo = 4
        id = self._write_capsule_surface(file,topo,id,self.structure_mask)

        file.close()

    def _file_write_surface_top(self):
        mask = self.structure_mask

        file = open('surface.top','w')
        id = 1
        file.write('SURFACETOPO 1 SURFACE_THICKNESS %f\n' %(self.thickness))
        topo = 3;
        id = self._write_canopy_surface(file,topo,id)
        file.write('*\n')


        file.write('SURFACETOPO 3 \n')
        topo = 3;
        id = self._write_capsule_surface(file,topo,id,mask)
        file.write('*\n')

        file.close()

    def _file_write_embedded_surface_top(self):
        file = open('embeddedSurface.top','w')

        file.write('Nodes nodeset\n')
        self._write_coord(file,self.embeddedsurface_mask)
        id = 1
        file.write('Elements StickMovingSurface_8 using nodeset\n')
        topo = 4;
        id = self._write_canopy_surface(file,topo,id)

        file.write('Elements StickMovingSurface_9 using nodeset\n')
        topo = 4
        id = self._write_cable_surface(file,topo,id,self.embeddedsurface_mask)


        file.write('Elements StickMovingSurface_10 using nodeset\n')
        topo = 4
        id = self._write_capsule_surface(file,topo,id,self.embeddedsurface_mask)

        file.close()











    def _file_write_common_data_include(self):
        file = open('common.data.include','w')

        file.write('NODES\n')
        self._write_coord(file,self.structure_mask)

        # TopologyId, finite element type, node1Id, node2Id, node3Id ..
        # (some element type has more nodes, but 129 is 3 nodes membrane element)
        # First part for the canopy
        id = 1
        topo = 129;
        file.write('TOPOLOGY\n')
        id = self._write_canopy_surface(file,topo,id)


        #beam element 6
        topo = 6;
        id = self._write_cable_beam(file,topo,id,self.structure_mask)

        #capsule triangle around
        topo = 129
        id = self._write_capsule_surface(file,topo,id,self.structure_mask)
        file.close()

    def _file_write_aeros_mesh_include(self):

        mask = self.structure_mask

        file = open('aeros.mesh.include','w')
        # elementId, element attribute(material) id

        file.write('ATTRIBUTS\n')
        canopy_attr = 1;
        start_ele = 1
        end_ele = 2*self.layer_n*(self.canopy_n - 1)
        file.write('%d   %d   %d\n' %(start_ele, end_ele, canopy_attr))

        cable_beam_attr = 2;
        start_ele = end_ele + 1
        end_ele = end_ele + 2*(self.cable_n-1)
        file.write('%d   %d   %d\n' %(start_ele, end_ele, cable_beam_attr))

        capsule_attr = 4;
        start_ele = end_ele + 1
        end_ele = end_ele + 2*self.capsule_n*self.layer_n
        file.write('%d   %d   %d\n' %(start_ele, end_ele, capsule_attr))

        file.write('*\n')



        ###############################################################################################################################################################

        # Material specifies material
        # material id, ....
        youngsModulus = 6.08e8
        poissonRatio = 0.4
        density = 1153.4
        thickness = self.thickness
        file.write('MATERIAL\n')
        canopy_attr = 1;
        file.write('%d 0 %f %f %f 0 0 %f 0 0 0 0 0 0 0\n' %(canopy_attr, youngsModulus, poissonRatio, density, thickness))

        capsule_attr = 4;
        file.write('%d 0 %f %f %f 0 0 %f 0 0 0 0 0 0 0\n' %(capsule_attr, youngsModulus, poissonRatio, density, thickness))


        cable_beam_attr = 2;
        E = 7.9e8
        poissonRatio = 0.4
        rho = 1000
        cable_r = self.cable_r
        Ix = np.pi*cable_r**4/4.0
        Iy = np.pi*cable_r**4/4.0
        Iz = np.pi*cable_r**4/4.0
        area= np.pi*cable_r*cable_r

        file.write('%d %.15f %.15f %.10E %.10E 0 0 0 0 0 0 0 %.15f %.15f %.15f\n' %(cable_beam_attr, area, E, poissonRatio, rho, Ix,Iy,Iz))
        file.write('*\n')




        file.write('EFRAMES\n')
        B = self.beam1_end_coord
        A = self.beam1_start_coord
        Sx = (B-A)/np.linalg.norm(B-A) # direction of AB
        e1 = np.array([1.0,0.0,0.0],dtype=float)
        Sy = np.cross(Sx,e1)
        Sz = np.cross(Sx,Sy)
        for id in range(2*self.layer_n*(self.canopy_n - 1) + 1, 2*self.layer_n*(self.canopy_n - 1) + self.cable_n):
            file.write('%d  %f %f %f    %f %f %f    %f %f %f\n' %(id,  Sx[0],Sx[1],Sx[2],  Sy[0],Sy[1],Sy[2],  Sz[0],Sz[1],Sz[2]))

        B = self.beam2_end_coord
        A = self.beam2_start_coord
        Sx = (B-A)/np.linalg.norm(B-A) # direction of AB
        e1 = np.array([1.0,0.0,0.0],dtype=float)
        Sy = np.cross(Sx,e1)
        Sz = np.cross(Sx,Sy)
        for id in range(2*self.layer_n*(self.canopy_n - 1) + self.cable_n, 2*self.layer_n*(self.canopy_n - 1) + 2*self.cable_n - 1):
            file.write('%d  %f %f %f    %f %f %f    %f %f %f\n' %(id,  Sx[0],Sx[1],Sx[2],  Sy[0],Sy[1],Sy[2],  Sz[0],Sz[1],Sz[2]))
        file.write('*\n')



        # MATUSAGE specifies material
        # start element number, end element number, material id
        file.write('MATUSAGE\n')
        file.write('1 %d  1\n' %(2*self.layer_n*(self.canopy_n - 1)))
        #aerosMeshInclude.write('%d %d 2\n' %(8*nPoints - 7,8*nPoints + 4*mPoints - 8))
        file.write('*\n')

        # MATLAW can specify nonlinear property of the material
        # material id, material name, ...
        file.write('MATLAW\n')
        file.write('1 HyperElasticPlaneStress %f %f %f %f\n' %(density, youngsModulus, poissonRatio, thickness))
        file.write('*\n')


        # Pressure
        #file.write('PRESSURE\n')
        #file.write('1 %d %f\n' %(2*self.layer_n*(self.canopy_n - 1), -4000.0))
        #file.write('*\n')

        #Fix the payload
        file.write('DISP\n')
        for i in range(self.capsule_n*(self.layer_n+1)):
            node_id = mask[self.capsule_node[i]] + 1
            if(i != self.capsule_con_l and i != self.capsule_con_r): #these are the connected nodes
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



cl = 0.01
num,x,y = hilbertCurve(2,2,0.5)
nPoints, xArray, yArray = curveRefine(num,x,y, cl,False, True)
#AFL(-1, 0, 1, 0)

#nPoints, xArray, yArray = straightLine(5)
parachute_mesh = Parachute(nPoints, xArray, yArray, cable_n=100, cable_k=4, cable_r=5.0e-3, layer_n=4, layer_t=0.01, capsule_x = -0.05, capsule_y=-2)

parachute_mesh._file_write_structure_top()

parachute_mesh._file_write_aeros_mesh_include()

parachute_mesh._file_write_common_data_include()

parachute_mesh._file_write_embedded_surface_top()

parachute_mesh._file_write_surface_top()
