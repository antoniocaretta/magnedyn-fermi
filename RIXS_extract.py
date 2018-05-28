import glob
import h5py
import numpy as np
import time
import warnings
import os

###############################################################################
###############################################################################
###############################################################################

vec_sum=[[-1,0],[0,1],[1,0],[0,-1]]

def new_pos(ind_y,ind_x,matrix,thresh,shape):
    
    partial_positions_y=[ind_y]
    partial_positions_x=[ind_x]
    
    final_positions_y=[partial_positions_y[0]]
    final_positions_x=[partial_positions_x[0]]
    final_positions_z=[matrix[ind_y,ind_x]]
    
    matrix[ind_y,ind_x]=0
    
    while len(partial_positions_y)!=0:
        
        vec_contr=[partial_positions_y[0]-1>=0,partial_positions_x[0]+1<shape[1],partial_positions_y[0]+1<shape[0],partial_positions_x[0]-1>=0]
        
        vec_contr=[i for i in range(len(vec_contr)) if vec_contr[i]]
        
        for ind in vec_contr:
            
            yi=vec_sum[ind][0]+partial_positions_y[0]
            xi=vec_sum[ind][1]+partial_positions_x[0]
            
            if matrix[yi,xi]>thresh:
                
                partial_positions_y.append(yi)
                partial_positions_x.append(xi)
                
                final_positions_y.append(yi)
                final_positions_x.append(xi)
                final_positions_z.append(matrix[yi,xi])
                
                matrix[yi,xi]=0
                
        del partial_positions_y[0]
        del partial_positions_x[0]
    
    return(final_positions_y,final_positions_x,final_positions_z)
    
###############################################################################

def new_label_pos(matrix,thresh,min_len):
    
    points_y=[]
    points_x=[]
    points_z=[]
    
    shape_mat=matrix.shape
    
    vector=np.where(matrix > thresh)
    
    for ind in range(len(vector[0])):
        
        if matrix[vector[0][ind],vector[1][ind]]!=0:
            
            [py,px,pz]=new_pos(vector[0][ind],vector[1][ind],matrix,thresh,shape_mat)
            
            if len(px)>min_len:
                
                points_y.append(py)
                points_x.append(px)
                points_z.append(pz)
    
    return(points_y,points_x,points_z)

###############################################################################

def totpoints(h5fn, image_dir, bunches_dir, thresh, min_len, test_unified, control_unified):
    
    tot_points_x=[]
    tot_points_y=[]
    tot_points_z=[]
    tot_images=[]
    
    f = h5py.File(h5fn , 'r')
    
    try:
        
        Image8 = f[image_dir][...]
        size = Image8.shape
        
        
    except:
        
        size = None
        warnings.warn("Warning: no images in "+ h5fn, stacklevel=2)
    
    if bunches_dir is not None:
        
        try:
            
            bunches = f[bunches_dir][...]
            
        except:
            
            if len(size) == 2:
                
                bunches = [0]
                
            else:
                
                bunches = np.arange(len(size[0]))
            
            warnings.warn("Warning: no bunches in "+ bunches_dir, stacklevel=2)
    
    if size is not None:
        
        if len(size)>2:
            
            for ind in np.arange(size[0]):
                
                [py,px,pz]=new_label_pos(Image8[ind],thresh,min_len)
                
                for label in range(len(px)):
                    
                    if len(px[label])>15 and test_unified:
                        
                        control=test_unified_blobs(px[label],py[label],pz[label],thresh)
                        
                        if control>control_unified:
                            
                            [min_y,max_y]=[min(py[label]),max(py[label])]
                            [min_x,max_x]=[min(px[label]),max(px[label])]
                            
                            new_y=(np.asarray(py[label])-min_y+1).tolist()
                            new_x=(np.asarray(px[label])-min_x+1).tolist()
                            
                            contr=[1*(min_y>0),1*(max_y-1<size[0]),1*(min_x>0),1*(max_x-1<size[1])]
                            
                            matrix=Image8[ind][min_y-contr[0]:max_y+1+contr[1],min_x-contr[2]:max_x+1+contr[3]]
                            
                            [min_y,min_x]=[min_y-contr[0],min_x-contr[2]]
                            
                            for pos in np.arange(len(new_x)):
                                
                                matrix[new_y[pos],new_x[pos]]=pz[label][pos]
                            
                            [new_x,new_y,new_z]=label_unified_blob(new_x,new_y,pz[label],matrix,min_len,min_y,min_x)
                            
                            if len(new_x)>0:
                                
                                del px[label]
                                del py[label]
                                del pz[label]
                                
                                for pos in np.arange(len(new_x)):
                                    
                                    px.append(new_x[pos])
                                    py.append(new_y[pos])
                                    pz.append(new_z[pos])
                
                for newlab in np.arange(len(px)):
                    
                    tot_images.append(bunches[ind])
                    tot_points_x.append(px[newlab])
                    tot_points_y.append(py[newlab])
                    tot_points_z.append(pz[newlab])
                    
        else:
            
            [py,px,pz]=new_label_pos(Image8,thresh,thresh)
            
            for label in range(len(px)):
                
                if len(px[label])>15 and test_unified:
                    
                    control=test_unified_blobs(px[label],py[label],pz[label],thresh)
                    
                    if control>control_unified:
                        
                        [min_y,max_y]=[min(py[label]),max(py[label])]
                        [min_x,max_x]=[min(px[label]),max(px[label])]
                        
                        new_y=(np.asarray(py[label])-min_y+1).tolist()
                        new_x=(np.asarray(px[label])-min_x+1).tolist()
                        
                        contr=[1*(min_y>0),1*(max_y-1<size[0]),1*(min_x>0),1*(max_x-1<size[1])]
                        
                        matrix=Image8[min_y-contr[0]:max_y+1+contr[1],min_x-contr[2]:max_x+1+contr[3]]
                        
                        [min_y,min_x]=[min_y-contr[0],min_x-contr[2]]
                        
                        for pos in np.arange(len(new_x)):
                            
                            matrix[new_y[pos],new_x[pos]]=pz[label][pos]
                        
                        [new_x,new_y,new_z]=label_unified_blob(new_x,new_y,pz[label],matrix,min_len,min_y,min_x)
                        
                        if len(new_x)>0:
                            
                            del px[label]
                            del py[label]
                            del pz[label]
                            
                            for pos in np.arange(len(new_x)):
                                
                                px.append(new_x[pos])
                                py.append(new_y[pos])
                                pz.append(new_z[pos])
                        
            for newlab in np.arange(len(px)):
                
                tot_images.append(bunches[0])
                tot_points_x.append(px[newlab])
                tot_points_y.append(py[newlab])
                tot_points_z.append(pz[newlab])
    
    return(tot_points_y,tot_points_x,tot_points_z,tot_images)

###############################################################################

def all_h5fns(fullh5path = None, runnr = None, image_dir = None, bunches_dir = None, thresh = None, min_len = None, test_unified = False, control_unified = None):
    
    print('')
    
    
    if fullh5path is None:
        
        raise ValueError('No file directory in input')
    
    if type(fullh5path) is not str:
            
            raise ValueError(fullh5path, 'is not a directory.')
    
    else:
        
        raw_data = '*.h5'
        
        if runnr is None:
            
            runnr = ''
            
            warnings.warn('No h5 run name in input; all .h5 will be considered.')
            
            h5fns = sorted(glob.glob(fullh5path + '/' + raw_data))
            
        else:
                
            if type(runnr) is not str:
                
                runnr = ''
                
                warnings.warn('h5 run name not valid; all .h5 will be considered.')
                
                h5fns = sorted(glob.glob(fullh5path + '/' + raw_data))
        
        
        if runnr != '':
            
            h5fns = sorted(glob.glob(fullh5path + '/' + runnr + raw_data))
        
        time0 = time.time()
        
        tot_ys = []
        tot_xs = []
        tot_zs = []
        tot_imgs = []
        
        if len(h5fns) == 0:
            
            raise ValueError('Files not found.')
    
    if image_dir is None:
        
        raise ValueError('No h5 images directory in input')
    
    if type(image_dir) is not str:
        
        raise ValueError(image_dir, 'is not a directory.')
    
    if bunches_dir is None:
        
        warnings.warn('No h5 bunches directory in input; switching to image index')
    
    if type(bunches_dir) is not str:
        
        bunches_dir = None
        
        warnings.warn(image_dir, 'is not a directory; switching to image index')
    
    if thresh is None:
        
        thresh = 10
        
        warnings.warn('No threshold configured; set to default value 10.', stacklevel=2)
        
    else:
        
        if not isinstance(thresh, (int, float)):
            
            raise ValueError(thresh, 'is unsuitable as threshold value.')
    
    if thresh<=0 or int(round(thresh)) == 0:
        
        thresh = 10
        
        warnings.warn('Threshold or its rounding is not positive defined; set to default 10.', stacklevel=2)
    
    if  round(thresh) - thresh != 0:
        
        thresh = int(round(thresh))
        
        warnings.warn('Threshold is not integer defined; rounded to nearest integer'+ str(thresh), stacklevel=2)
    
    thresh = int(thresh)
    
    if min_len is None:
        
        min_len = 2
        
        warnings.warn('No minimum dimension configured; set to default value 2.', stacklevel=2)
        
    else:
        
        if not isinstance(min_len, (int, float)):
            
            raise ValueError(min_len, 'is unsuitable as minimum dimension value.')
    
    if min_len<=0 or int(round(min_len)) == 0:
        
        min_len = 2
        
        warnings.warn('Minimum dimension or its rounding is not positive defined; set to default 2.', stacklevel=2)
    
    if  round(min_len) - min_len != 0:
        
        min_len = int(round(min_len))
        
        warnings.warn('Minimum dimension is not integer defined; rounded to nearest integer' + str(min_len), stacklevel=2)
    
    min_len = int(min_len)
    
    if control_unified is None and test_unified:
        
        control_unified = 1.
        
        warnings.warn('No control unified configured; set to default value 1.', stacklevel=2)
    
    if control_unified < 1. and test_unified:
        
        warnings.warn('Setting control unified with a value minor than 1 may cause unneccessary tests of unified events.', stacklevel=2)
    
    print('')
    print('Found ', len(h5fns), 'files.')
    print('')
    
    for ind in np.arange(len(h5fns)):
        
        [py,px,pz,imgs]=totpoints(h5fns[ind],image_dir,bunches_dir,thresh,min_len,test_unified,control_unified)
        
        tot_ys.append(py)
        tot_xs.append(px)
        tot_zs.append(pz)
        tot_imgs.append(imgs)
        
        est_time=(len(h5fns) - ind - 1)*(time.time() - time0)/(ind + 1)
        
        if len(h5fns) - ind > 1:
            
            print('Estimated remaining time: ',int(est_time/60),' m ',int(est_time - 60*int(est_time/60)),' s')
    
    print('')
    print('Saving data in h5...')
    print('')
    
    if runnr == '':
        
        namefile='/results'
        
    else:
        
        namefile = '/' + runnr
    
    extension = '.h5'
    
    cwd = os.getcwd()
    
    if len(glob.glob(cwd + namefile + extension)) == 1:
        
        cont = 0
        
        while len(glob.glob(cwd + namefile + '_' + str(cont) + extension)) == 1:
            
            cont += 1
        
        namefile = namefile + '_' + str(cont)
    
    namefile=namefile[1::]
    
    f = h5py.File(namefile + extension, "w")
    
    cont = 1
    
    grp0 = f.create_group("Events")
    
    for ind in np.arange(len(tot_imgs)):
        
        if len(tot_imgs[ind])>0:
            
            name = h5fns[ind][h5fns[ind].rfind('\\')+1::]
            
            for pos in np.arange(len(tot_imgs[ind])):
                
                grpi = grp0.create_group(str(cont))
                
                grpi.create_dataset("name_folder", data=name)
                grpi.create_dataset("image_number", data=tot_imgs[ind][pos])
                
                grpi.create_dataset("spots_x", data=tot_xs[ind][pos], compression="gzip")
                grpi.create_dataset("spots_y", data=tot_ys[ind][pos], compression="gzip")
                grpi.create_dataset("spots_z", data=tot_zs[ind][pos], compression="gzip")
                
                cont+=1
    
    grp = f.create_group("Stats")
    
    if runnr == '':
        
        name = h5fns[0][h5fns[0].rfind('\\')+1:len(h5fns[0])-3]
        
    else:
        
        name = runnr
    
    grp.create_dataset("Run_number", data=name)
    grp.create_dataset("Threshold", data=thresh)
    grp.create_dataset("Minimum_size", data=min_len)
    grp.create_dataset("Events_collected", data=cont-1)
    grp.create_dataset("N_of_files", data=len(h5fns))
    
    if test_unified:
        
        grp1 = grp.create_group("Test_unified")
        
        grp1.create_dataset("Test", data='True')
        grp1.create_dataset("Control", data=control_unified)
        
    else:
        
        grp1 = grp.create_group("Test_unified")
        
        grp1.create_dataset("Test", data='False')
    
    f.close()
    
    est_time = time.time() - time0
    
    print('Total time:',int(est_time/60),' m ',int(est_time - 60*int(est_time/60)),' s')
    print('')
    print('Results saved in file ',namefile + extension, 'in directory', cwd)
    
    return(tot_ys,tot_xs,tot_zs,tot_imgs,namefile + extension)

###############################################################################

def test_unified_blobs(xs,ys,values,thresh):
    
    contr=0.
    
    xcs=[]
    ycs=[]
    
    m=sorted(values)
    
    maximum=m[len(m)-1]
    
    m=[m[i] for i in np.arange(len(m)-1) if m[i]!=m[i+1]]
    
    m.append(maximum)
    
    for ind in np.arange(len(m)):
        
        xc=0.
        yc=0.
        pound=0.
        
        minimum=m[ind]-1
        
        for pos in np.arange(len(values)):
            
            if values[pos]>minimum:
                
                val = values[pos]-minimum
                
                xc += xs[pos]*val
                yc += ys[pos]*val
                pound += val
        
        xcs=np.append(xcs,xc/pound)
        ycs=np.append(ycs,yc/pound)
    
    contr=sum((xcs-(sum(xcs)/len(xcs)))**2)+sum((ycs-(sum(ycs)/len(ycs)))**2)#########/len(xcs)
    
    return(contr)

###############################################################################

def label_unified_blob(points_x,points_y,points_z,matrix,min_len,delta_y,delta_x):
    
    shape=matrix.shape
    
    mat_mean=np.zeros(shape)
    
    for pos in np.arange(len(points_x)):
        
        vec_contr=[points_y[pos]-1>=0,points_x[pos]+1<shape[1],points_y[pos]+1<shape[0],points_x[pos]-1>=0]
        
        vec_contr=[i for i in range(len(vec_contr)) if vec_contr[i]]
        
        vec_ind=[matrix[points_y[pos]+vec_sum[ind][0],points_x[pos]+vec_sum[ind][1]] for ind in vec_contr]
        
        mat_mean[points_y[pos],points_x[pos]]=sum(vec_ind)/len(vec_ind)
    
    mat_grad=np.zeros(shape)
    
    for pos in np.arange(len(points_x)):
        
        vec_contr=[points_y[pos]-1>=0,points_x[pos]+1<shape[1],points_y[pos]+1<shape[0],points_x[pos]-1>=0]
        
        vec_contr=[i for i in range(len(vec_contr)) if vec_contr[i]]
        
        vec_ind=[mat_mean[points_y[pos],points_x[pos]]-mat_mean[points_y[pos]+vec_sum[ind][0],points_x[pos]+vec_sum[ind][1]] for ind in vec_contr]
        
        mat_grad[points_y[pos],points_x[pos]]=sum(vec_ind)/len(vec_ind)
    
    maximum = np.max(mat_grad)
    
    mat_ref=matrix*(mat_grad>0)
    
    [new_points_y,new_points_x,new_points_z]=new_label_pos(mat_ref,maximum,min_len)
    
    labelled_image=np.zeros(matrix.shape).astype('uint8')
    
    frequency=[]
    
    if len(new_points_x)>0:
        
        frequency = [ind for ind in np.arange(len(new_points_x)) if len(new_points_x[ind])>2]
    
    if len(frequency)<2:
        
        return([],[],[])
    
    for lab in np.arange(len(frequency)):
        
        label=frequency[lab]
        
        for pos in np.arange(len(new_points_x[label])):
            
            labelled_image[new_points_y[label][pos],new_points_x[label][pos]]=lab+1
    
    all_points_x=[]
    all_points_y=[]
    all_points_z=[]
    
    for label in frequency:
        
        for pos in np.arange(len(new_points_x[label])):
            
            all_points_x.append(new_points_x[label][pos])
            all_points_y.append(new_points_y[label][pos])
            all_points_z.append(new_points_z[label][pos])
    
    not_points_x=[]
    not_points_y=[]
    not_points_z=[]
    
    for pos in np.arange(len(points_x)):
        
        cont=0
        
        for pos_new in np.arange(len(all_points_x)):
            
            if points_x[pos]==all_points_x[pos_new] and points_y[pos]==all_points_y[pos_new]:
                
                cont=1
                break
        
        if cont == 0:
            
            not_points_x.append(points_x[pos])
            not_points_y.append(points_y[pos])
            not_points_z.append(points_z[pos])
    
    new_points_x=[new_points_x[label] for label in frequency]
    new_points_y=[new_points_y[label] for label in frequency]
    new_points_z=[new_points_z[label] for label in frequency]
    
    contr_image=((labelled_image>0)).astype('uint8')*2
    
    while len(not_points_x)>0:
        
        cont=0
        
        for ind in np.arange(len(not_points_x)):
            
            x=not_points_x[ind-cont]
            y=not_points_y[ind-cont]
            z=not_points_z[ind-cont]
            
            vec_contr=[y-1>=0,x+1<shape[1],y+1<shape[0],x-1>=0]
            
            vec_contr=[i for i in range(len(vec_contr)) if vec_contr[i]]
            
            vec_ind=[ind for ind in vec_contr if labelled_image[y+vec_sum[ind][0],x+vec_sum[ind][1]]!=0 and contr_image[y+vec_sum[ind][0],x+vec_sum[ind][1]]>1]
            
            pos_y=[y+vec_sum[ind][0] for ind in vec_ind]
            pos_x=[x+vec_sum[ind][1] for ind in vec_ind]
            
            grad=[]
            
            for i in np.arange(len(pos_x)):
                
                grad.append(not_points_z[ind-cont]-matrix[pos_y[i],pos_x[i]])
            
            if len(grad)>0:
                
                max_x=pos_x[int(np.argmax(grad))]
                max_y=pos_y[int(np.argmax(grad))]
                
                labelled_image[y,x]=labelled_image[max_y,max_x]
                
                new_points_x[labelled_image[max_y,max_x]-1].append(x)
                new_points_y[labelled_image[max_y,max_x]-1].append(y)
                new_points_z[labelled_image[max_y,max_x]-1].append(z)
                
                contr_image[y,x]=1
                
                del not_points_x[ind-cont]
                del not_points_y[ind-cont]
                del not_points_z[ind-cont]
                
                cont+=1
        
        contr_image=contr_image*2
    
    for label in np.arange(len(new_points_x)):
        
        for pos in np.arange(len(new_points_x[label])):
            
            new_points_x[label][pos]+=delta_x
            new_points_y[label][pos]+=delta_y
    
    return(new_points_x,new_points_y,new_points_z)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################



print('            Raw data analysis on images in h5 file')
print('')
print('')


folder = input("Tell folder path: (folder containing .h5 files) ")
print('')

runnr = input("Tell Run nr.: (string identifying run .h5 files) ")
print('')

thresh = float(input("Tell threshold: (float > 0.) "))
print('')

min_len = float(input("Tell minimum dimension of structures: (int > 0) "))
print('')

test_unified = (input("Test unified structures? (True/False) ")) 
test_unified = (test_unified == 'True' or test_unified == 'T' or test_unified == '1')
print('')

if test_unified:
    
    control_unified = float(input("Control unified structures: (float > 0.) "))
    print('')




#thresh=8.
#min_len=4
#test_unified=True
#control_unified = 1.





directory='ext-dev/ccd-exp-01/Image8'
bunches_dir='bunches'

tot_ys=[]
tot_xs=[]
tot_zs=[]
tot_imgs=[]

[tot_ys,tot_xs,tot_zs,tot_imgs,file]=all_h5fns(folder,runnr,directory,bunches_dir,thresh,min_len,test_unified,control_unified)





###############################################################################




from scipy.optimize import leastsq as opt
from scipy.special import wofz


def parabola(x,p0,p1,p2):
    
    return(p0 + p1 * x + p2 * x**2)


###############################################################################

errfunc = lambda p, x, y: (y - parabola(x,*p))**2

###############################################################################

def data_analisys(file):
    
    f = h5py.File(file , 'r')
    
    max_y=0
    max_x=0
    
    events = f['Events']
    
    for label in np.arange(len(events)):
        
        mix = np.max(events[str(label+1)]['spots_x'][...])
        
        if max_x < mix: max_x = mix
        
        miy = np.max(events[str(label+1)]['spots_y'][...])
        
        if max_y < miy: max_y = miy
    
    [max_y,max_x] = [max_y+1,max_x+1]
    
    tot_img=np.zeros([max_y,max_x])
    
    print('')
    print('Creating total image...')
    print('')
    
    xcms = []
    ycms = []
    
    for label in np.arange(len(events)):
        
        xs = events[str(label+1)]['spots_x'][...]
        ys = events[str(label+1)]['spots_y'][...]
        zs = events[str(label+1)]['spots_z'][...]
        
        max_val = np.max(zs)
        
        k = np.log(max_val)/max_val
        
        pounds = np.exp(zs*k)
        
        xcms.append(sum(pounds*xs)/sum(pounds))
        
        ycms.append(sum(pounds*ys)/sum(pounds))
        
        for ind in np.arange(len(xs)):
            
            tot_img[ys[ind],xs[ind]] += zs[ind]
    
    print('')
    print('Shifting data...')
    print('')
    
    [py,px]=np.unravel_index(tot_img.argmax(),[max_y,max_x])
    
    pars = [py,0.,0.]
    
    maxs = []
    
    for col in np.arange(max_x):
        
        maxs.append(np.argmax(tot_img[:,col]))
    
    out =  opt(errfunc,pars,args=(np.arange(len(maxs)),maxs))[0]
        
    truemaxs=[]
    trueargs=[]
    
    for col in np.arange(len(maxs)):
        
        if (maxs[col]-parabola(col,*out))**2/parabola(col,*out) < 1.:
            
            truemaxs.append(maxs[col])
            trueargs.append(col)
            
    out =  opt(errfunc,pars,args=(np.asarray(trueargs)*1.0,np.asarray(truemaxs)*1.0))[0]
    
    x_0_pol = parabola(np.arange(max_x),*out)
    
    shift_0=max(x_0_pol)
    
    shape_row=max_y+int(max(x_0_pol))-int(min(x_0_pol))
    
    print('')
    print('Creating spectrum...')
    print('')
    
    spectre=np.zeros(shape_row)
    
    for label in np.arange(len(xcms)):
        
        delta = shift_0-parabola(xcms[label],*out)
        
        spectre[int(round(ycms[label] + delta))] += 1
    
    return(spectre)



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################




print('            Spectrum reconstruction')
print('')
print('')


input("Begin analisys (hit enter) ")
print('')


time0 = time.time()

fileh5 = os.getcwd() + '\\' + file

spectre = []

spectre = data_analisys(fileh5)

time1 = time.time()

print('')
print('Total time:', time1-time0)





import matplotlib.pyplot as plt

###############################################################################

def V(x, amp, alpha, gamma, x0):
    
    sigma = alpha / np.sqrt(2 * np.log(2))
    return amp*np.real(wofz(((x-x0) + 1j*np.abs(gamma))/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)

###############################################################################

def Vbis(x,ag,gg1,gg2,wg,bl,ag_,gg1_,gg2_,wg_):
    return V(x,ag,gg1,gg2,wg) + V(x,ag_,gg1_,gg2_,wg_) + np.abs(bl)

###############################################################################

errfunc_voigt = lambda p, x, y: (y - Vbis(x,*p))**2

###############################################################################

def data_fit(spectre):
    
    mu1 = np.argmax(spectre[0:int(np.argmax(spectre)*0.9)])
    mu2 = int(np.argmax(spectre)*1.2)+np.argmax(spectre[int(np.argmax(spectre)*1.1):len(spectre)-1])
    
    if spectre[mu1] > spectre[mu2]:
        
        pars = [np.max(spectre)*5.,1.,1.,np.argmax(spectre)*1.,0.1,spectre[mu1]*5.,1.,1.,mu1*1.]
        
    else:
        
        pars = [np.max(spectre)*5.,1.,1.,np.argmax(spectre)*1.,0.1,spectre[mu2]*5.,1.,1.,mu2*1.]
    
    print('')
    print('Parameters acquisition...')
    print('')
    
    out =  opt(errfunc_voigt,pars,args=(np.arange(len(spectre)),spectre))[0]
    
    out1 = out[0:4].tolist()
    out2 = out[5:].tolist()
    
    dist=int((abs(out2[3]-out1[3])/2.)*0.8)
    
    print('')
    print('Plotting spectre...')
    print('')
    
    plt.figure(100)
    
    plt.plot(spectre)
    plt.plot(V(np.arange(len(spectre)),*out1)+out[4])
    plt.plot(V(np.arange(len(spectre)),*out2)+out[4])
    plt.plot(Vbis(np.arange(len(spectre)),*out))
    
    integral1 = sum(spectre[int(out1[3]-dist)+1:int(out1[3]+dist)]) + (spectre[int(out1[3])-dist] + spectre[int(out1[3])+dist])*0.5
    
    integral2 = sum(spectre[int(out2[3]-dist)+1:int(out2[3]+dist)]) + (spectre[int(out2[3])-dist] + spectre[int(out2[3])+dist])*0.5
    
    return(integral1,out[3],integral2,out2[3])


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################




print('            Spectrum fit')
print('')
print('')


input("Begin fit of spectrum (hit enter) ")
print('')


time0 = time.time()

[int1, mu1, int2, mu2] = data_fit(spectre)

time1 = time.time()

print('')
print('Total time:', time1-time0)


print('')
print('')

if int1 > int2:
    
    print('Elastic position: ', mu1, ', integral ', int1)
    print('Inlastic position: ', mu2, ', integral ', int2)
    
else:
    
    print('Elastic position: ', mu2, ', integral ', int2)
    print('Inlastic position: ', mu1, ', integral ', int1)

