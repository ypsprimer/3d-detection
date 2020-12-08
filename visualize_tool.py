import ipyvolume as ipv
import numpy as np

def compare(pred, lab, ignore_index=255):
    pred[pred==ignore_index]=0
    lab[lab==ignore_index]=0
    lab = lab>0
    tp = pred&lab
    fp = pred&(~lab)
    fn = lab&(~pred)

    ipv.figure()
    ipv.plot_isosurface(tp,color='green',level=0.9,controls=False)
    ipv.plot_isosurface(fp,color='blue',level=0.9,controls=False)
    ipv.plot_isosurface(fn,color='red',level=0.9,controls=False)
    ipv.xlabel('')
    ipv.ylabel('')
    ipv.zlabel('')
    ipv.show()
    
def show_centerline(pred):
    pred[pred==255]=0
    mesh = np.meshgrid(*[range(pred.shape[i]) for i in range(1,4)],indexing='ij')
    cents = mesh+pred[1:]
    centdots=cents.transpose([1,2,3,0])[pred[0]>0]
    ipv.figure()
    ipv.scatter(centdots[:,0],centdots[:,1],centdots[:,2], size=0.1)
    # ipv.volshow(pred[0])
    ipv.show()
    
def show_quiver(pred, sparse = 10):
    arrows=pred[1:].transpose([1,2,3,0])[pred[0]>0]
    coord = np.array(np.where(pred[0]>0)).astype('float').transpose()
    ipv.figure()
    ipv.quiver(coord[::sparse,0], coord[::sparse,1], coord[::sparse,2], arrows[::sparse,0], arrows[::sparse,1], arrows[::sparse,2],size=1)
    ipv.show()