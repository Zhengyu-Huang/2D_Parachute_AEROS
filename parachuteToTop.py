import numpy as np
import matplotlib.pyplot as plt
import math
#This is a mesh generator for quasi-1D parachte
#there are three simple configurations: sinCurve, straightLine and hatShape

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
def straightLine( ):
    nPoints = 100;
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

def curveRefine(num, xx, yy,cl, plotOrNot):
    numArray = np.zeros(num,np.int32)
    for i in range(num-1):
        ds = np.sqrt((yy[i+1]-yy[i])**2 + (xx[i+1] - xx[i])**2)
        n = max(np.ceil(ds/cl), 1) #n small segments on each segment
        numArray[i+1] = numArray[i] + n;

    nPoints = numArray[-1]+1
    xArray = np.zeros(nPoints)
    yArray = np.zeros(nPoints)

    for i in range(num-1):
        xArray[numArray[i]:(numArray[i+1]+1)] = np.linspace(xx[i],xx[i+1],num=numArray[i+1] - numArray[i] + 1)
        yArray[numArray[i]:(numArray[i+1]+1)] = np.linspace(yy[i],yy[i+1],num=numArray[i+1] - numArray[i] + 1)

    if(plotOrNot):
        plt.plot(xArray, yArray,'-*')
        plt.ylim([-0.5,1.5])
        plt.xlim([-1,1])
        plt.show()
    return nPoints, xArray, yArray

###########################################################################################################################
###########################################################################################################################
#Start to generate AEROS and Mathcer input file


cl = 0.01
#num,x,y = candle()
num,x,y = hilbertCurve(2,0.5,0.5)
nPoints, xArray, yArray = curveRefine(num,x,y, cl,True)
#nPoints, xArray, yArray = straightLine( )




#############################################################################################################################
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
commonDataInclude = open('common.data.include','w')

# NodeId, x coordinate, y coordinate, z coordinate
commonDataInclude.write('NODES\n')
# First part for the canopy
for i in range(nPoints):
    # i = 0, 1, 2 ..n-1
    commonDataInclude.write('%d   %f  %f  %f \n' %(5*i+1, xArray[i], yArray[i], 0.0))
    commonDataInclude.write('%d   %f  %f  %f \n' %(5*i+2, xArray[i], yArray[i], cl))
    commonDataInclude.write('%d   %f  %f  %f \n' %(5*i+3, xArray[i], yArray[i], 2*cl))
    commonDataInclude.write('%d   %f  %f  %f \n' %(5*i+4, xArray[i], yArray[i], 3*cl))
    commonDataInclude.write('%d   %f  %f  %f \n' %(5*i+5, xArray[i], yArray[i], 4*cl))
# Second part for 4 cables
cStart=np.array([[xArray[0],yArray[0],0.0],[xArray[0],yArray[0],4*cl],[xArray[nPoints-1],yArray[nPoints-1],0.0],[xArray[nPoints-1],yArray[nPoints-1],4*cl]])
payloadPos = -3.0;
cEnd=np.array([0.0,payloadPos,2*cl])
#mPoints = int(np.linalg.norm(cStart[0,:] - cEnd)/0.01);
mPoints = 1
for i in range(mPoints-1):
    # i = 0, 1, 2 ..m-2
    for j in range(4):
        c = cStart[j,:] + (cEnd - cStart[j,:])*(i+1.0)/mPoints;
        commonDataInclude.write('%d   %f  %f  %f \n' %(5*nPoints+4*i+j+1, c[0], c[1], c[2]))
commonDataInclude.write('%d   %f  %f  %f \n' %(5*nPoints + 4*mPoints - 3, cEnd[0], cEnd[1], cEnd[2]))
# Third part for payload
commonDataInclude.write('%d   %f  %f  %f \n' %(5*nPoints + 4*mPoints - 2, -2*cl, payloadPos, 0.0))
commonDataInclude.write('%d   %f  %f  %f \n' %(5*nPoints + 4*mPoints - 1, -2*cl, payloadPos, 4*cl))
commonDataInclude.write('%d   %f  %f  %f \n' %(5*nPoints + 4*mPoints    ,  2*cl, payloadPos, 0.0))
commonDataInclude.write('%d   %f  %f  %f \n' %(5*nPoints + 4*mPoints + 1,  2*cl, payloadPos, 4*cl))

commonDataInclude.write('*\n')


# TopologyId, finite element type, node1Id, node2Id, node3Id ..
# (some element type has more nodes, but 129 is 3 nodes membrane element)
# First part for the canopy
topology = 129;
commonDataInclude.write('TOPOLOGY\n')
for i in range(nPoints-1):
    if(i <= (nPoints-1)/2):
        commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*i+1, topology,  5*i+1, 5*i+6, 5*i+7))
        commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*i+2, topology,  5*i+1, 5*i+7, 5*i+2))
        commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*i+3, topology,  5*i+2, 5*i+7, 5*i+8))
        commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*i+4, topology,  5*i+2, 5*i+8, 5*i+3))
        commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*i+5, topology,  5*i+3, 5*i+8, 5*i+4))
        commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*i+6, topology,  5*i+4, 5*i+8, 5*i+9))
        commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*i+7, topology,  5*i+4, 5*i+9, 5*i+5))
        commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*i+8, topology,  5*i+5, 5*i+9, 5*i+10))
    else:
        commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*i+1, topology,  5*i+1, 5*i+6, 5*i+2))
        commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*i+2, topology,  5*i+2, 5*i+6, 5*i+7))
        commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*i+3, topology,  5*i+2, 5*i+7, 5*i+3))
        commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*i+4, topology,  5*i+3, 5*i+7, 5*i+8))
        commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*i+5, topology,  5*i+4, 5*i+3, 5*i+9))
        commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*i+6, topology,  5*i+3, 5*i+8, 5*i+9))
        commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*i+7, topology,  5*i+5, 5*i+4, 5*i+10))
        commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*i+8, topology,  5*i+4, 5*i+9, 5*i+10))
# Second part for 4 cables
# For the cables, segment ids are
# 8*nPoints-7,    8*nPoints-6,    8*nPoints-5,      8*nPoints-4
# 8*nPoints-3,    8*nPoints-2,    8*nPoints-1,      8*nPoints
# ......
# 8*nPoints + 4*mPoints-11,    8*nPoints + 4*mPoints - 10,    8*nPoints + 4*mPoints - 9,      8*nPoints + 4*mPoints - 8
cableTopology = 203
for i in range(mPoints):
    if(i == 0):
        a = [1,5,5*nPoints-4,5*nPoints]
    else:
        a = [5*nPoints + 4*i-3,5*nPoints + 4*i-2,5*nPoints + 4*i-1,5*nPoints + 4*i]
    if(i == mPoints-1):
        b = [5*nPoints + 4*mPoints-3,5*nPoints+4*mPoints-3,5*nPoints+4*mPoints-3,5*nPoints+4*mPoints-3]
    else:
        b = [5*nPoints + 4*i+1,5*nPoints + 4*i+2,5*nPoints + 4*i+3,5*nPoints + 4*i+4]
    for k in range(4):

        commonDataInclude.write('%d   %d   %d   %d\n' %(8*nPoints + 4*i - 7 + k, cableTopology,  a[k],b[k]))

# Third part for payload(4 elements)
# 8*nPoints + 4*mPoints-7,    8*nPoints + 4*mPoints-6,    8*nPoints + 4*mPoints-5,      8*nPoints + 4*mPoints-4
payloadTopology = 129
commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*nPoints + 4*mPoints -7, payloadTopology,  5*nPoints + 4*mPoints - 2, 5*nPoints + 4*mPoints - 1, 5*nPoints + 4*mPoints - 3))
commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*nPoints + 4*mPoints -6, payloadTopology,  5*nPoints + 4*mPoints - 1, 5*nPoints + 4*mPoints + 1, 5*nPoints + 4*mPoints - 3))
commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*nPoints + 4*mPoints -5, payloadTopology,  5*nPoints + 4*mPoints + 1, 5*nPoints + 4*mPoints    , 5*nPoints + 4*mPoints - 3))
commonDataInclude.write('%d   %d  %d  %d %d \n' %(8*nPoints + 4*mPoints -4, payloadTopology,  5*nPoints + 4*mPoints - 2, 5*nPoints + 4*mPoints    , 5*nPoints + 4*mPoints - 3))



# elementId, element attribute(material) id
attributes = 1;
commonDataInclude.write('ATTRIBUTS\n')
for i in range(nPoints-1):
    commonDataInclude.write('%d   %d \n' %(8*i+1, attributes))
    commonDataInclude.write('%d   %d \n' %(8*i+2, attributes))
    commonDataInclude.write('%d   %d \n' %(8*i+3, attributes))
    commonDataInclude.write('%d   %d \n' %(8*i+4, attributes))
    commonDataInclude.write('%d   %d \n' %(8*i+5, attributes))
    commonDataInclude.write('%d   %d \n' %(8*i+6, attributes))
    commonDataInclude.write('%d   %d \n' %(8*i+7, attributes))
    commonDataInclude.write('%d   %d \n' %(8*i+8, attributes))
cableAttributes = 2;
for i in range(mPoints):
    commonDataInclude.write('%d   %d \n' %(8*nPoints-8 + 4*i + 1, cableAttributes))
    commonDataInclude.write('%d   %d \n' %(8*nPoints-8 + 4*i + 2, cableAttributes))
    commonDataInclude.write('%d   %d \n' %(8*nPoints-8 + 4*i + 3, cableAttributes))
    commonDataInclude.write('%d   %d \n' %(8*nPoints-8 + 4*i + 4, cableAttributes))
payloadAttributes = 1;
for i in range(4):
    commonDataInclude.write('%d   %d \n' %(8*nPoints + 4*mPoints -7 + i, payloadAttributes))

commonDataInclude.write('*\n')


commonDataInclude.close()
###############################################################################################################################################################
aerosMeshInclude = open('aeros.mesh.include','w')
# Material specifies material
# material id, ....
youngsModulus = 6.08e8
poissonRatio = 0.4
density = 1153.4
thickness = 7.62e-5
aerosMeshInclude.write('MATERIAL\n')
aerosMeshInclude.write('1 0 %f %f %f 0 0 %f 0 0 0 0 0 0 0\n' %(youngsModulus, poissonRatio, density, thickness))
stiffness = 4.0e3
aerosMeshInclude.write('2  SPRINGMAT  %f\n' %(stiffness))
aerosMeshInclude.write('*\n')

# MATUSAGE specifies material
# start element number, end element number, material id
aerosMeshInclude.write('MATUSAGE\n')
aerosMeshInclude.write('1 %d 1\n' %(8*nPoints - 8))
#aerosMeshInclude.write('%d %d 2\n' %(8*nPoints - 7,8*nPoints + 4*mPoints - 8))
aerosMeshInclude.write('*\n')

# MATLAW can specify nonlinear property of the material
# material id, material name, ...
aerosMeshInclude.write('MATLAW\n')
aerosMeshInclude.write('1 HyperElasticPlaneStress %f %f %f %f\n' %(density, youngsModulus, poissonRatio, thickness))
aerosMeshInclude.write('*\n')


#Pressure
aerosMeshInclude.write('PRESSURE\n')
aerosMeshInclude.write('1 %d %f\n' %(8*nPoints-8, -4000.0))
aerosMeshInclude.write('*\n')


aerosMeshInclude.write('DISP\n')
for freedom in range(1,7):
    #Fix payload
    aerosMeshInclude.write('%d %d 0.0\n' %(5*nPoints+4*mPoints-3, freedom))
    aerosMeshInclude.write('%d %d 0.0\n' %(5*nPoints+4*mPoints-2, freedom))
    aerosMeshInclude.write('%d %d 0.0\n' %(5*nPoints+4*mPoints-1, freedom))
    aerosMeshInclude.write('%d %d 0.0\n' %(5*nPoints+4*mPoints  , freedom))
    aerosMeshInclude.write('%d %d 0.0\n' %(5*nPoints+4*mPoints+1, freedom))

aerosMeshInclude.write('* symmetry planes\n')
aerosMeshInclude.write('%d thru %d step %d 3 0.0\n' %(1, 5*nPoints-4, 5))
aerosMeshInclude.write('%d thru %d step %d 4 0.0\n' %(1, 5*nPoints-4, 5))
aerosMeshInclude.write('%d thru %d step %d 5 0.0\n' %(1, 5*nPoints-4, 5))
aerosMeshInclude.write('%d thru %d step %d 3 0.0\n' %(5, 5*nPoints, 5))
aerosMeshInclude.write('%d thru %d step %d 4 0.0\n' %(5, 5*nPoints, 5))
aerosMeshInclude.write('%d thru %d step %d 5 0.0\n' %(5, 5*nPoints, 5))
aerosMeshInclude.close()

######################################################################################################################################################################
structureTop = open('structure.top','w')

structureTop.write('Nodes nodeset\n')
for i in range(nPoints):
    # i = 0, 1, 2 ..n-1
    structureTop.write('%d   %f  %f  %f \n' %(5*i+1, xArray[i], yArray[i], 0.0))
    structureTop.write('%d   %f  %f  %f \n' %(5*i+2, xArray[i], yArray[i], cl))
    structureTop.write('%d   %f  %f  %f \n' %(5*i+3, xArray[i], yArray[i], 2*cl))
    structureTop.write('%d   %f  %f  %f \n' %(5*i+4, xArray[i], yArray[i], 3*cl))
    structureTop.write('%d   %f  %f  %f \n' %(5*i+5, xArray[i], yArray[i], 4*cl))
# Second part for 4 cables
for i in range(mPoints-1):
    # i = 0, 1, 2 ..m-2
    for j in range(4):
        c = cStart[j,:] + (cEnd - cStart[j,:])*(i+1.0)/mPoints;
        structureTop.write('%d   %f  %f  %f \n' %(5*nPoints+4*i+j+1, c[0], c[1], c[2]))
structureTop.write('%d   %f  %f  %f \n' %(5*nPoints+4*mPoints-3, cEnd[0], cEnd[1], cEnd[2]))
# Third part for payload
structureTop.write('%d   %f  %f  %f \n' %(5*nPoints + 4*mPoints - 2, -2*cl, payloadPos, 0.0))
structureTop.write('%d   %f  %f  %f \n' %(5*nPoints + 4*mPoints - 1, -2*cl, payloadPos, 4*cl))
structureTop.write('%d   %f  %f  %f \n' %(5*nPoints + 4*mPoints    ,  2*cl, payloadPos, 0.0))
structureTop.write('%d   %f  %f  %f \n' %(5*nPoints + 4*mPoints + 1,  2*cl, payloadPos, 4*cl))

topology = 4;
structureTop.write('Elements StickMovingSurface using nodeset\n')
for i in range(nPoints-1):
    if(i <= (nPoints-1)/2):
        structureTop.write('    %d   %d  %d  %d %d \n' %(8*i+1, topology,  5*i+1, 5*i+6, 5*i+7))
        structureTop.write('    %d   %d  %d  %d %d \n' %(8*i+2, topology,  5*i+1, 5*i+7, 5*i+2))
        structureTop.write('    %d   %d  %d  %d %d \n' %(8*i+3, topology,  5*i+2, 5*i+7, 5*i+8))
        structureTop.write('    %d   %d  %d  %d %d \n' %(8*i+4, topology,  5*i+2, 5*i+8, 5*i+3))
    #Flip the edge
        structureTop.write('    %d   %d  %d  %d %d \n' %(8*i+5, topology,  5*i+3, 5*i+8, 5*i+4))
        structureTop.write('    %d   %d  %d  %d %d \n' %(8*i+6, topology,  5*i+4, 5*i+8, 5*i+9))
        structureTop.write('    %d   %d  %d  %d %d \n' %(8*i+7, topology,  5*i+4, 5*i+9, 5*i+5))
        structureTop.write('    %d   %d  %d  %d %d \n' %(8*i+8, topology,  5*i+5, 5*i+9, 5*i+10))
    else:
        structureTop.write('%d   %d  %d  %d %d \n' %(8*i+1, topology,  5*i+1, 5*i+6, 5*i+2))
        structureTop.write('%d   %d  %d  %d %d \n' %(8*i+2, topology,  5*i+2, 5*i+6, 5*i+7))
        structureTop.write('%d   %d  %d  %d %d \n' %(8*i+3, topology,  5*i+2, 5*i+7, 5*i+3))
        structureTop.write('%d   %d  %d  %d %d \n' %(8*i+4, topology,  5*i+3, 5*i+7, 5*i+8))
        structureTop.write('%d   %d  %d  %d %d \n' %(8*i+5, topology,  5*i+4, 5*i+3, 5*i+9))
        structureTop.write('%d   %d  %d  %d %d \n' %(8*i+6, topology,  5*i+3, 5*i+8, 5*i+9))
        structureTop.write('%d   %d  %d  %d %d \n' %(8*i+7, topology,  5*i+5, 5*i+4, 5*i+10))
        structureTop.write('%d   %d  %d  %d %d \n' %(8*i+8, topology,  5*i+4, 5*i+9, 5*i+10))
structureTop.write('Elements cable using nodeset\n')
cableTopology = 1
for i in range(mPoints):
    if(i == 0):
        a = [1,5,5*nPoints-4,5*nPoints]
    else:
        a = [5*nPoints + 4*i-3,5*nPoints + 4*i-2,5*nPoints + 4*i-1,5*nPoints + 4*i]
    if(i == mPoints-1):
        b = [5*nPoints + 4*mPoints-3,5*nPoints+4*mPoints-3,5*nPoints+4*mPoints-3,5*nPoints+4*mPoints-3]
    else:
        b = [5*nPoints + 4*i+1,5*nPoints + 4*i+2,5*nPoints + 4*i+3,5*nPoints + 4*i+4]
    for k in range(4):
        structureTop.write('%d   %d   %d   %d\n' %(8*nPoints + 4*i - 7 + k, cableTopology,  a[k],b[k]))
topology = 4;
structureTop.write('Elements payload using nodeset\n')
structureTop.write('%d   %d  %d  %d %d \n' %(8*nPoints + 4*mPoints -7, topology,  5*nPoints + 4*mPoints - 2, 5*nPoints + 4*mPoints - 1, 5*nPoints + 4*mPoints - 3))
structureTop.write('%d   %d  %d  %d %d \n' %(8*nPoints + 4*mPoints -6, topology,  5*nPoints + 4*mPoints - 1, 5*nPoints + 4*mPoints + 1, 5*nPoints + 4*mPoints - 3))
structureTop.write('%d   %d  %d  %d %d \n' %(8*nPoints + 4*mPoints -5, topology,  5*nPoints + 4*mPoints + 1, 5*nPoints + 4*mPoints    , 5*nPoints + 4*mPoints - 3))
structureTop.write('%d   %d  %d  %d %d \n' %(8*nPoints + 4*mPoints -4, topology,  5*nPoints + 4*mPoints - 2, 5*nPoints + 4*mPoints    , 5*nPoints + 4*mPoints - 3))

structureTop.close()

##############################################################################################################################################
embeddedSurface = open('embeddedSurface.top','w')

embeddedSurface.write('Nodes nodeset\n')
for i in range(nPoints):
    # i = 0, 1, 2 ..n-1
    embeddedSurface.write('%d   %f  %f  %f \n' %(5*i+1, xArray[i], yArray[i], 0.0))
    embeddedSurface.write('%d   %f  %f  %f \n' %(5*i+2, xArray[i], yArray[i], cl))
    embeddedSurface.write('%d   %f  %f  %f \n' %(5*i+3, xArray[i], yArray[i], 2*cl))
    embeddedSurface.write('%d   %f  %f  %f \n' %(5*i+4, xArray[i], yArray[i], 3*cl))
    embeddedSurface.write('%d   %f  %f  %f \n' %(5*i+5, xArray[i], yArray[i], 4*cl))
# Second part for 4 cables
for i in range(mPoints-1):
    # i = 0, 1, 2 ..m-2
    for j in range(4):
        c = cStart[j,:] + (cEnd - cStart[j,:])*(i+1.0)/mPoints;
        embeddedSurface.write('%d   %f  %f  %f \n' %(5*nPoints+4*i+j+1, c[0], c[1], c[2]))
embeddedSurface.write('%d   %f  %f  %f \n' %(5*nPoints+4*mPoints-3, cEnd[0], cEnd[1], cEnd[2]))
# Third part for payload
embeddedSurface.write('%d   %f  %f  %f \n' %(5*nPoints + 4*mPoints - 2, -2*cl, payloadPos, 0.0))
embeddedSurface.write('%d   %f  %f  %f \n' %(5*nPoints + 4*mPoints - 1, -2*cl, payloadPos, 4*cl))
embeddedSurface.write('%d   %f  %f  %f \n' %(5*nPoints + 4*mPoints    ,  2*cl, payloadPos, 0.0))
embeddedSurface.write('%d   %f  %f  %f \n' %(5*nPoints + 4*mPoints + 1,  2*cl, payloadPos, 4*cl))

topology = 4;
embeddedSurface.write('Elements StickMovingSurface_8 using nodeset\n')
for i in range(nPoints-1):
    embeddedSurface.write('%d   %d  %d  %d %d \n' %(8*i+1, topology,  5*i+1, 5*i+6, 5*i+7))
    embeddedSurface.write('%d   %d  %d  %d %d \n' %(8*i+2, topology,  5*i+1, 5*i+7, 5*i+2))
    embeddedSurface.write('%d   %d  %d  %d %d \n' %(8*i+3, topology,  5*i+2, 5*i+7, 5*i+8))
    embeddedSurface.write('%d   %d  %d  %d %d \n' %(8*i+4, topology,  5*i+2, 5*i+8, 5*i+3))
    #Flip the edge
    embeddedSurface.write('%d   %d  %d  %d %d \n' %(8*i+5, topology,  5*i+3, 5*i+8, 5*i+4))
    embeddedSurface.write('%d   %d  %d  %d %d \n' %(8*i+6, topology,  5*i+4, 5*i+8, 5*i+9))
    embeddedSurface.write('%d   %d  %d  %d %d \n' %(8*i+7, topology,  5*i+4, 5*i+9, 5*i+5))
    embeddedSurface.write('%d   %d  %d  %d %d \n' %(8*i+8, topology,  5*i+5, 5*i+9, 5*i+10))

embeddedSurface.write('Elements StickMovingSurface_9 using nodeset\n')
embeddedSurface.write('%d   %d  %d  %d %d \n' %(8*nPoints + 4*mPoints -7, topology,  5*nPoints + 4*mPoints - 2, 5*nPoints + 4*mPoints - 1, 5*nPoints + 4*mPoints - 3))
embeddedSurface.write('%d   %d  %d  %d %d \n' %(8*nPoints + 4*mPoints -6, topology,  5*nPoints + 4*mPoints - 1, 5*nPoints + 4*mPoints + 1, 5*nPoints + 4*mPoints - 3))
embeddedSurface.write('%d   %d  %d  %d %d \n' %(8*nPoints + 4*mPoints -5, topology,  5*nPoints + 4*mPoints + 1, 5*nPoints + 4*mPoints    , 5*nPoints + 4*mPoints - 3))
embeddedSurface.write('%d   %d  %d  %d %d \n' %(8*nPoints + 4*mPoints -4, topology,  5*nPoints + 4*mPoints - 2, 5*nPoints + 4*mPoints    , 5*nPoints + 4*mPoints - 3))

embeddedSurface.close()
##############################################################################################################################################
surfaceTop = open('surface.top','w')
surfaceTop.write('SURFACETOPO 1 SURFACE_THICKNESS %f\n' %(thickness))
topology = 3;
for i in range(nPoints-1):
    surfaceTop.write('%d   %d  %d  %d %d \n' %(8*i+1, topology,  5*i+1, 5*i+6, 5*i+7))
    surfaceTop.write('%d   %d  %d  %d %d \n' %(8*i+2, topology,  5*i+1, 5*i+7, 5*i+2))
    surfaceTop.write('%d   %d  %d  %d %d \n' %(8*i+3, topology,  5*i+2, 5*i+7, 5*i+8))
    surfaceTop.write('%d   %d  %d  %d %d \n' %(8*i+4, topology,  5*i+2, 5*i+8, 5*i+3))
    #Flip the edge
    surfaceTop.write('%d   %d  %d  %d %d \n' %(8*i+5, topology,  5*i+3, 5*i+8, 5*i+4))
    surfaceTop.write('%d   %d  %d  %d %d \n' %(8*i+6, topology,  5*i+4, 5*i+8, 5*i+9))
    surfaceTop.write('%d   %d  %d  %d %d \n' %(8*i+7, topology,  5*i+4, 5*i+9, 5*i+5))
    surfaceTop.write('%d   %d  %d  %d %d \n' %(8*i+8, topology,  5*i+5, 5*i+9, 5*i+10))


surfaceTop.write('*\n')
surfaceTop.close()
