import numpy as np
import matplotlib.pyplot as plt

def Canopy(type, cl, xScale, yScale, *args):
    '''
    Generate a folded parachute canopy in (-xScale, xScale)(0, yScalse)
    :param type:
    :param cl:
    :param xScale:
    :param yScale:
    :param plotOrNot:
    :param closeOrNot:
    :return:
    '''

    if(type == 'sinCurve'):
        return sinCurve(cl, xScale, yScale, *args)
    elif(type == 'line'):
        num, xx, yy = 2, np.array([-xScale, xScale]), np.array([0.,0.])
    elif(type == 'candle'):
        num, xx, yy = candle( xScale, yScale)
    elif(type == 'sFolding'):
        num,xx,yy = sFolding(args[0], xScale,yScale)
    elif(type == 'hilbert'):
        num, xx, yy = hilbertCurve(args[0], xScale, yScale)
    elif(type == 'hat'):
        num, xx, yy = 3, np.array([-xScale,0., xScale]), np.array([0., yScale, 0.])
    elif(type == 'zCurve'):
        num, xx, yy = zCurve(xScale, yScale)

    return curveRefine(num, xx, yy,cl)









#This is a sin curve
#int  nPoints, number of nodes
#int* xArray, x coordinates of these nodes
#int* yArray, y coordinates of these nodes
def sinCurve(cl, xScale, yScale, plotOrNot, *args):
    k = args[0]

    nPoints = int( 2*xScale* np.sqrt((k*np.pi*yScale)**2)/cl)

    A = yScale;
    xArray = np.linspace(-xScale,xScale,num=nPoints)
    yArray = A*(np.cos(2*k*np.pi*xArray)+1.0)/2.0
    if(plotOrNot):
        plt.plot(xArray, yArray,'-*')
        plt.ylim([-0.5,2*yScale])
        plt.xlim([-2*xScale,2*xScale])
        plt.show()
    return nPoints, xArray, yArray

#This is a straight line
#int  nPoints, number of nodes
#int* xArray, x coordinates of these nodes
#int* yArray, y coordinates of these nodes
def straightLine(cl, xScale, yScale, plotOrNot):
    nPoints = int(2*xScale/cl+1)
    if(nPoints%2 == 0):
        nPoints += 1
    xArray = np.linspace(-xScale,xScale,num=nPoints)
    yArray = 0*xArray
    return nPoints, xArray, yArray





def candle(xScale, yScale):
    xx = np.array([-0.5, -0.07, -0.05, -0.45, -0.5, -0.05, -0.04, -0.5, -0.48, -0.38, -0.375, -0.25, -0.245, -0.13, -0.125, -0.05, -0.045,
                   0.045, 0.05, 0.125, 0.13, 0.245, 0.25, 0.375, 0.38, 0.48, 0.5, 0.04, 0.05, 0.5, 0.45, 0.05, 0.07, 0.5])*xScale;
    yy = np.array([0, 0, 0.125, 0.125, 0.25, 0.25, 0.375, 0.375, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5,
                   1, 1, 0.375, 0.375, 0.25, 0.25, 0.125, 0.125, 0, 0])*yScale;

    num = len(xx)
    return num, xx, yy


def zCurve(xScale,yScale):
    '''
    x = np.array([-0.5, -0.1, -0.5, 0.5, 0.1,  0.5])
    y = np.array([0.0,  0,   1,    1,   0.0 ,  0.0])
    '''

    hy =  1e-3  # thickness

    hx =  0.05 - hy  # canopy_cl
    x = np.array([-xScale/2.0, -xScale/4.0, -xScale/4.0 + hx, -xScale/4.0, -xScale/2.0, -xScale/2.0 - hx, -xScale/2.0,
                   xScale/2.0,  xScale/2.0 +hx,   xScale/2.0,  xScale/4.0,  xScale/4.0 - hx,  xScale/4.0,  xScale/2.0])

    y = np.array([0, 0, hy/2, hy, hy, 1.5*hy, 2*hy,
                  2*hy, 1.5*hy, hy, hy, 0.5*hy, 0,0])


    return len(x), x, y



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

def curveRefine(num, xx, yy,cl, closeOrNot = False, plotOrNot = True):

    segNum = num if closeOrNot else num - 1

    numArray = np.zeros(segNum + 1,np.int32)

    for i in range(segNum):
        ds = np.sqrt((yy[(i+1)%num]-yy[i])**2 + (xx[(i+1)%num] - xx[i])**2)
        n = max(np.ceil(ds/cl), 1) #n small segments on each segment
        numArray[i+1] = numArray[i] + n;

    nPoints = numArray[-1]+ 1
    xArray = np.zeros(nPoints)
    yArray = np.zeros(nPoints)

    for i in range(segNum):
        xArray[numArray[i]:(numArray[i+1]+1)] = np.linspace(xx[i],xx[(i+1)%num],num=numArray[i+1] - numArray[i] + 1)
        yArray[numArray[i]:(numArray[i+1]+1)] = np.linspace(yy[i],yy[(i+1)%num],num=numArray[i+1] - numArray[i] + 1)

    if(plotOrNot):
        plt.plot(xArray, yArray,'-*')
        plt.show()

    if(closeOrNot):
        cornerNode = numArray[0:-1]
    else:
        cornerNode = numArray[1:-1]
    print('cornerNode is ', cornerNode)
    return (nPoints-1, xArray[0:-1], yArray[0:-1]) if closeOrNot else (nPoints, xArray, yArray)


if __name__ == "__main__":
    cl = 0.01
    xScale, yScale = 0.1, 1.0
    type = 'sFolding'

    Canopy(type, cl, xScale, yScale, 3)
