import numpy as np
import matplotlib.pyplot as plt
from Folding import curveRefine

def AeroShell(type, x, y, cl):
    '''
    Generate aeroshell top nodes are at (-x, y) (x, y)
    :param type: AFL or square
    :param k: node number for the nose in AFL
    :return: num of points, x, y coordinate, 3 cable connected nodes
    '''
    if(type == 'AFL'):
        return AFL(-x,y,x,y,cl)
    elif(type == 'square'):
        return squareCurve(-x, y, x, y)
    elif(type == 'circle'):
        return circle(-x, y, x, y,cl)



def AFL(xa, ya, xb, yb,cl):
    x4 = 2.25
    x3 = x4 - 1.012*np.tan(36.9*np.pi/180)
    x2 = x3 - 0.506*np.tan(59*np.pi/180)
    x1 = x2 - 0.5*np.tan(33.95*np.pi/180)
    y1,y2,y3,y4 = 0, -0.5 , -0.5-0.506, -0.5-0.506-1.012

    o_x,o_y = 0.0, y4 - 2.25*np.tan(20*np.pi/180) + 0.5* 2.25/np.cos(20*np.pi/180)

    k = int(1.125*2*np.pi/9.0/cl) + 2 #40 degrees curve
    print('In AFL k is',k)

    nose_x, nose_y = o_x + 1.125*np.cos(np.linspace(-70,-110,k)*np.pi/180), o_y + 1.125*np.sin(np.linspace(-70,-110,k)*np.pi/180)

    x_p = np.append(np.append([-x1,0,x1,x2,x3,x4,],nose_x),[-x4,-x3,-x2])
    y_p = np.append(np.append([y1,0,y1,y2,y3,y4], nose_y), [y4,y3,y2])

    num = k+9
    scale = (xb-xa)/(2*x1)
    x,y = x_p*scale, y_p *scale
    x , y = x + xa - x[0], y + ya - y[0]
    print('x1 is %f, smallest y coord is %f',x1, np.amin(y))

    plt.fill(x, y,edgecolor='r', fill=False)
    plt.show()


    return num, x, y,[0,1,2]


def squareCurve(xa, ya, xb, yb):
    h = xb - xa

    x = np.array([xa, 0,  xb, xb, xa])
    y = np.array([ya, ya, ya, ya-h, ya-h])
    num = 5
    return num, x, y,[0,1,2]

def circle(xa, ya, xb, yb, cl = 0.025,r = 1.0):
    h = xb - xa
    theta = np.arcsin(h/(2*r))

    n1 = 2*int(theta*r/cl + 1) + 1
    theta_1 = np.linspace(-theta,theta,n1)
    n2 = int(2 * (np.pi-theta) * r / cl) + 2
    theta_2 = np.linspace(theta, 2*np.pi-theta,n2)

    num = n1+n2-2
    x,y = np.empty(num),np.empty(num)

    x[:n1],x[n1-1:] = r*np.cos(theta_1),r*np.cos(theta_2[:-1])
    y[:n1],y[n1-1:] = r*np.sin(theta_1),r*np.sin(theta_2[:-1])
    return num, x, y,[0,n1//2,n1]

if __name__ == "__main__":
    type = 'circle'
    cl = 0.05
    x , y , k= 0.311423704133, 1.0, 10
    nn,xx,yy,_ = AeroShell(type, x, y, cl)

    import Folding
    Folding.curveRefine(nn,xx,yy,cl,closeOrNot=True)