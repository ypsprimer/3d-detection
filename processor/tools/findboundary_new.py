import numpy as np

def getSingleAxisBoundary(labelin, mean_id1, mean_id2, shape_id):
    x1 = 0
    x2 = 0
    labelx = labelin
    
    label_nx = labelin.copy()
    label_nx[label_nx>0]=0

    
    lmean_z = labelin.mean(mean_id1)
    lmean_z[lmean_z>0]=1
    
    xlist = lmean_z.mean(mean_id2)
    dpos_meanzy = np.where(xlist == 0)
    pmax = np.where(xlist == xlist.max())
    
    dpos_meanzy = np.array(dpos_meanzy)
    pmax = np.array(pmax)

    pleft = 0
    pright = labelin.shape[shape_id]

    for i in range(len(dpos_meanzy[0])):
        if pmax[0][0]>dpos_meanzy[0][i]:
#        print('k')
            if dpos_meanzy[0][i]>pleft:
                pleft = dpos_meanzy[0][i]
#            print('h')
        if pmax[0][0]<dpos_meanzy[0][i]:
            if dpos_meanzy[0][i]<pright:
                pright = dpos_meanzy[0][i]
            
####### x1, x2 in z-zip plane            
#    print(pleft, pright, np.array(pmax))

    if shape_id==2:
        label_nx[:,:,pleft:pright] = labelin[:,:,pleft:pright].copy()
    if shape_id==1:
        label_nx[:,pleft:pright,:] = labelin[:,pleft:pright,:].copy()
    if shape_id==0:
        label_nx[pleft:pright,:,:] = labelin[pleft:pright,:,:].copy()
    
    x1 = pleft
    x2 = pright
    
    labelx = label_nx
    
#    print(label_nx.sum())
    
    return x1, x2, labelx


def findBoundary(label):
    labelin = label.copy()
    x1, x2, labelx = getSingleAxisBoundary(labelin, 0, 0, 2)
    y1, y2, labely = getSingleAxisBoundary(labelx, 0, 1, 1)
    z1, z2, labelz = getSingleAxisBoundary(labely, 1, 1, 0)
    
    return z1, y1, x1, z2, y2, x2, labelz

def findBoundaryX(label):
#    labeln = label.copy()
#    labeln[labeln>0]=0
    
    z1, y1, x1, z2, y2, x2, _ = findBoundary(label)
    
#    print (z1, y1, x1, z2, y2, x2)
##    z1, y1, x1, z2, y2, x2, labeln = findBoundary(labeln)
    
#    print (z1, y1, x1, z2, y2, x2)
##    z1, y1, x1, z2, y2, x2, labeln = findBoundary(labeln)
    
    return z1, y1, x1, z2, y2, x2