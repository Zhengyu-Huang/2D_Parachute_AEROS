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
def straightLine():
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
    n = max(np.ceil(ds/cl), 2) #n small segments on each segment
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
        n = max(np.ceil(ds/cl), 2) #n small segments on each segment
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


#cl = 0.01
#num,x,y = candle()
#num,x,y = hilbertCurve(3,1,0.25)
#nPoints, xArray, yArray = curveRefine(num,x,y, 0.01,True)
nPoints, xArray, yArray = straightLine()

#nPoints, xArray, yArray = hatShape()

commonDataInclude = open('common.data.include','w')

# NodeId, x coordinate, y coordinate, z coordinate
commonDataInclude.write('NODES\n')
for i in range(nPoints):
    # i = 0, 1, 2 ..n-1
    commonDataInclude.write('%d   %f  %f  %f \n' %(4*i+1, xArray[i], yArray[i], 0.0))
    commonDataInclude.write('%d   %f  %f  %f \n' %(4*i+2, xArray[i], yArray[i], 0.01))
    commonDataInclude.write('%d   %f  %f  %f \n' %(4*i+3, xArray[i], yArray[i], 0.02))
    commonDataInclude.write('%d   %f  %f  %f \n' %(4*i+4, xArray[i], yArray[i], 0.03))
commonDataInclude.write('*\n')


# TopologyId, finite element type, node1Id, node2Id, node3Id ..
# (some element type has more nodes, but 129 is 3 nodes membrane element)
topology = 129;
commonDataInclude.write('TOPOLOGY\n')
for i in range(nPoints-1):
    commonDataInclude.write('%d   %d  %d  %d %d \n' %(6*i+1, topology,  4*i+1, 4*i+6, 4*i+2))
    commonDataInclude.write('%d   %d  %d  %d %d \n' %(6*i+2, topology,  4*i+1, 4*i+5, 4*i+6))
    commonDataInclude.write('%d   %d  %d  %d %d \n' %(6*i+3, topology,  4*i+2, 4*i+7, 4*i+3))
    commonDataInclude.write('%d   %d  %d  %d %d \n' %(6*i+4, topology,  4*i+2, 4*i+6, 4*i+7))
    commonDataInclude.write('%d   %d  %d  %d %d \n' %(6*i+5, topology,  4*i+3, 4*i+8, 4*i+4))
    commonDataInclude.write('%d   %d  %d  %d %d \n' %(6*i+6, topology,  4*i+3, 4*i+7, 4*i+8))
commonDataInclude.write('*\n')

# elementId, element attribute(material) id
attributes = 1;
commonDataInclude.write('ATTRIBUTS\n')
for i in range(nPoints-1):
    commonDataInclude.write('%d   %d \n' %(6*i+1, attributes))
    commonDataInclude.write('%d   %d \n' %(6*i+2, attributes))
    commonDataInclude.write('%d   %d \n' %(6*i+3, attributes))
    commonDataInclude.write('%d   %d \n' %(6*i+4, attributes))
    commonDataInclude.write('%d   %d \n' %(6*i+5, attributes))
    commonDataInclude.write('%d   %d \n' %(6*i+6, attributes))
commonDataInclude.write('*\n')



# Material specifies material
# material id, ....
youngsModulus = 6.08e8
poissonRatio = 0.4
density = 1153.4
thickness = 7.62e-5
commonDataInclude.write('MATERIAL\n')
commonDataInclude.write('1 0 %f %f %f 0 0 %f 0 0 0 0 0 0 0\n' %(youngsModulus, poissonRatio, density, thickness))
commonDataInclude.write('*\n')





commonDataInclude.close()

aerosMeshInclude = open('aeros.mesh.include','w')
# MATUSAGE specifies material
# start element number, end element number, material id
aerosMeshInclude.write('MATUSAGE\n')
aerosMeshInclude.write('1 %d 1\n' %(6*nPoints - 6))
aerosMeshInclude.write('*\n')

# MATLAW can specify nonlinear property of the material
# material id, material name, ...
aerosMeshInclude.write('MATLAW\n')
aerosMeshInclude.write('1 HyperElasticPlaneStress %f %f %f %f\n' %(density, youngsModulus, poissonRatio, thickness))
aerosMeshInclude.write('*\n')


# Pressure
#aerosMeshInclude.write('PRESSURE\n')
#aerosMeshInclude.write('1 %d %f\n' %(6*i+6, -4000.0))
#aerosMeshInclude.write('*\n')
#aerosMeshInclude.write('DISP\n')
for freedom in range(1,7):
    aerosMeshInclude.write('%d %d 0.0\n' %(1, freedom))
    aerosMeshInclude.write('%d %d 0.0\n' %(2, freedom))
    aerosMeshInclude.write('%d %d 0.0\n' %(3, freedom))
    aerosMeshInclude.write('%d %d 0.0\n' %(4, freedom))
    aerosMeshInclude.write('%d %d 0.0\n' %(4*nPoints-3, freedom))
    aerosMeshInclude.write('%d %d 0.0\n' %(4*nPoints-2, freedom))
    aerosMeshInclude.write('%d %d 0.0\n' %(4*nPoints-1, freedom))
    aerosMeshInclude.write('%d %d 0.0\n' %(4*nPoints, freedom))

aerosMeshInclude.write('* symmetry planes\n')
aerosMeshInclude.write('%d thru %d step %d 3 0.0\n' %(1, 4*nPoints-3, 4))
aerosMeshInclude.write('%d thru %d step %d 4 0.0\n' %(1, 4*nPoints-3, 4))
aerosMeshInclude.write('%d thru %d step %d 5 0.0\n' %(1, 4*nPoints-3, 4))
aerosMeshInclude.write('%d thru %d step %d 3 0.0\n' %(4, 4*nPoints, 4))
aerosMeshInclude.write('%d thru %d step %d 4 0.0\n' %(4, 4*nPoints, 4))
aerosMeshInclude.write('%d thru %d step %d 5 0.0\n' %(4, 4*nPoints, 4))
aerosMeshInclude.close()





structureTop = open('structure.top','w')

structureTop.write('Nodes nodeset\n')
for i in range(nPoints):
    # i = 0, 1, 2 ..n-1
    structureTop.write('    %d   %f  %f  %f \n' %(4*i+1, xArray[i], yArray[i], 0.0))
    structureTop.write('    %d   %f  %f  %f \n' %(4*i+2, xArray[i], yArray[i], 0.01))
    structureTop.write('    %d   %f  %f  %f \n' %(4*i+3, xArray[i], yArray[i], 0.02))
    structureTop.write('    %d   %f  %f  %f \n' %(4*i+4, xArray[i], yArray[i], 0.03))

topology = 4;
structureTop.write('Elements SlipMovingSurface_8 using nodeset\n')
for i in range(nPoints-1):
    structureTop.write('    %d   %d  %d  %d %d \n' %(6*i+1, topology,  4*i+1, 4*i+6, 4*i+2))
    structureTop.write('    %d   %d  %d  %d %d \n' %(6*i+2, topology,  4*i+1, 4*i+5, 4*i+6))
    structureTop.write('    %d   %d  %d  %d %d \n' %(6*i+3, topology,  4*i+2, 4*i+7, 4*i+3))
    structureTop.write('    %d   %d  %d  %d %d \n' %(6*i+4, topology,  4*i+2, 4*i+6, 4*i+7))
    structureTop.write('    %d   %d  %d  %d %d \n' %(6*i+5, topology,  4*i+3, 4*i+8, 4*i+4))
    structureTop.write('    %d   %d  %d  %d %d \n' %(6*i+6, topology,  4*i+3, 4*i+7, 4*i+8))
structureTop.close()


embeddedSurface = open('embedded.surface','w')

embeddedSurface.write('SURFACETOPO 1 SURFACE_THICKNESS %f\n' %(thickness))

topology = 3;
for i in range(nPoints-1):
    embeddedSurface.write('%d   %d  %d  %d %d \n' %(6*i+1, topology,  4*i+1, 4*i+6, 4*i+2))
    embeddedSurface.write('%d   %d  %d  %d %d \n' %(6*i+2, topology,  4*i+1, 4*i+5, 4*i+6))
    embeddedSurface.write('%d   %d  %d  %d %d \n' %(6*i+3, topology,  4*i+2, 4*i+7, 4*i+3))
    embeddedSurface.write('%d   %d  %d  %d %d \n' %(6*i+4, topology,  4*i+2, 4*i+6, 4*i+7))
    embeddedSurface.write('%d   %d  %d  %d %d \n' %(6*i+5, topology,  4*i+3, 4*i+8, 4*i+4))
    embeddedSurface.write('%d   %d  %d  %d %d \n' %(6*i+6, topology,  4*i+3, 4*i+7, 4*i+8))
embeddedSurface.write('*\n')
embeddedSurface.close()
