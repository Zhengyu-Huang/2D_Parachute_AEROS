import numpy as np

def Cables(type, As, Bs, cl, k, cable_r):
    n = len(type)
    nodes_n, cables_coord, phantoms_coord = [], [] ,[]
    for i in range(n):
        node_n, cable_coord, phantom_coord = One_Cable_coordinates(type[i], As[i,:], Bs[i,:], cl, k, cable_r)
        nodes_n.append(node_n)
        cables_coord.append(cable_coord)
        phantoms_coord.append(phantom_coord)
    return nodes_n, cables_coord, phantoms_coord




def One_Cable_coordinates(type, A, B, cl, k, cable_r):
    # cable AB
    #A = np.array([0.0,0.0,0.0],dtype=float)
    #B = np.array([0.0,0.0,1.0],dtype=float)

    #n = 20 # number of nodes on the cable AB
    #k = 6  # number of nodes on the phantom skeleton
    #cable_r = 0.1 # radius of cable
    #choose A cross e_1(1,0,0) as the first rigid phantom edge

    if(type == 'straight'):
        n = max(int(np.linalg.norm(B-A)/cl + 1) + 1,4)
        cable_coord = np.empty(shape=[n,3],dtype=float)
        phantom_coord = np.empty(shape=[n*k,3],dtype=float)
        for i in range(3):
            cable_coord[:,i] = np.linspace(A[i],B[i],num=n)

        nx,ny,nz = dir = (B-A)/np.linalg.norm(B-A) # direction of AB
        e1 = np.array([1.0,0.0,0.0],dtype=float)

        theta = 2*np.pi/k

        #Rotation matrix on https://en.wikipedia.org/wiki/Rotation_matrix
        R = np.array([[np.cos(theta) + nx*nx*(1-np.cos(theta))   ,nx*ny*(1-np.cos(theta)) - nz*np.sin(theta)   ,nx*nz*(1-np.cos(theta)) + ny*np.sin(theta)],
                      [ny*nx*(1-np.cos(theta))+nz*np.sin(theta)  ,np.cos(theta) + ny*ny*(1-np.cos(theta))      ,ny*nz*(1-np.cos(theta)) - nx*np.sin(theta)],
                      [nz*nx*(1-np.cos(theta))-ny*np.sin(theta)  ,nz*ny*(1-np.cos(theta)) + nx*np.sin(theta)   ,np.cos(theta) + nz*nz*(1-np.cos(theta)) ]],dtype=float)

        phontom_dr = np.empty(shape=[k,3],dtype=float)
        phontom_dr[0,:] = cable_r*np.cross(dir,e1)/np.linalg.norm(np.cross(dir,e1))

        for i in range(1,k):
            phontom_dr[i,:] = np.dot(R,phontom_dr[i-1,:])


        for i in range(n):
            for j in range(k):
                phantom_coord[i*k+j,:] = cable_coord[i,:] + phontom_dr[j,:]
        return n, cable_coord, phantom_coord
