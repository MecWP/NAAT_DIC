import sys #check
sys.path.append(r'C:\Users\padil\Documents\UFPA\ProjetoDIC\DIC_thiago\Edições')

import numpy as np #check
import numpy.matlib as npl #check
import scipy.linalg as linalg #check
import Interp_spline as it
#from scipy.signal import gaussian #(M, std, sym=True) W = np.matmul(W,np.transpose(W))
#import DIC_froutines

#import pdb

def GN_opt(I_f, I_i, template, P, Center_position, InterpS, InterpI, Def_order,GaussianWindow,Template_mask):
# P = U0(1) + V0(2) + Ux(3) + Vx(4) + Uy(5) + Vy(6)
# Uxx(7) + Vxx(8) + Uyy(9) + Vyy(10) + Uxy(11) + Vxy(12) + w(13)
# differences
    w = (template.shape[0] - 1) / 2
    Diff_vector = range(int(-w),int(w)+1)
    dx = npl.repmat(Diff_vector, int(2 * w + 1), 1)
    dy = np.transpose( dx )

    template = template*GaussianWindow


    #template = double(template)
    i = Center_position[0]
    j = Center_position[1]

    Rxi = Center_position[3] - 1
    Ryi = Center_position[2] - 1


    Mesh_allX, Mesh_allY = np.meshgrid(range(int(i - w), int(i + w) + 1), range(int(j - w), int(j + w) + 1))  #Mesh_allY = np.transpose(Mesh_allY)

    Mesh_allX = Mesh_allX
    Mesh_allY = Mesh_allY

    # Calculate Gradient
    # Look more for sub2ind

    idx = np.ravel_multi_index( (Mesh_allY.flatten('F'),Mesh_allX.flatten('F')), [InterpI['Fx'].shape[0],InterpI['Fx'].shape[1]], order='F' )


    Fx_flat = InterpI['Fx'].flatten('F')
    Fy_flat = InterpI['Fy'].flatten('F')


    P_ori = P.copy()
    if True:
        x = 0
        Fx = np.reshape(Fx_flat[idx - x], (template.shape[0],template.shape[1]), order='F')
        Fy = np.reshape(Fy_flat[idx - x], (template.shape[0],template.shape[1]), order='F')

        # Insert the derivatives
        Fgrad =[]
        Fgrad.append(Fx)
        Fgrad.append(Fy)
        Fgrad.append(Fx * dx)
        Fgrad.append(Fy * dx)
        Fgrad.append(Fx * dy)
        Fgrad.append(Fy * dy)

        if Def_order == 1:

            Fgrad.append(1. / 2 * Fx * (dx**2) )
            Fgrad.append(1. / 2 * Fy * (dx**2))
            Fgrad.append(1. / 2 * Fx * (dy**2))
            Fgrad.append(1. / 2 * Fy * (dy**2))
            Fgrad.append(Fx * dx * dy)
            Fgrad.append(Fy * dx * dy)

        # Pre calculate Mean
        template_Mean = np.mean(template[Template_mask==1])
        template_square_mean = np.sqrt(np.sum( (template[Template_mask==1] - np.mean(template[Template_mask==1]) )**2 )  )

        Trials = 0

        #pdb.set_trace()
        dP = np.zeros(  (  len(Fgrad),1 )  )
        dP_old = np.zeros(  ( len(Fgrad),1 )  )
        P_old = P_ori[0:len(Fgrad)]
        OutPoints = np.ones((Fx.shape[0],Fx.shape[1]))
        OutPoints = OutPoints.flatten('F')
        C = 999
        C_old = 1
        Norm = 99
        Exit = 0
        CorrT = 0
        while Trials < 40 and Norm > 0.000001 and Exit < 2: #and CorrT < 0.998:


            ######################## Calculate the next step with the GN algorithm ############
            # Calculate the Hessian Just once
            if Trials > 0:

                if Trials == 1:

                    Hessian = HessianGN(template,Fgrad,Template_mask)

                    #invHessian = linalg.inv(Hessian)

                dP = -linalg.solve(Hessian,GradientGN(template, I_f_indexed,Fgrad,Template_mask),sym_pos=True, overwrite_b=True,check_finite=False)#-np.matmul(invHessian,GradientGN(template, I_f_indexed,Fgrad))
                dP = np.reshape(dP,( len(Fgrad),1 ))


                Norm = linalg.norm(2 * dP)

                if np.array_equal(dP,dP_old): # Check if repeated gradient
                    Exit += 1
                else:
                    Exit = 0

            # Get the deformed image in a matrix
            #U0(1) + V0(2) + Ux(3) + Vx(4) + Uy(5) + Vy(6)
            #defvector_init_u = P_old[0]
            #defvector_init_v = P_old[1]
            #defvector_init_dudx = P_old[2]
            #defvector_init_dvdx = P_old[3]
            #defvector_init_dudy = P_old[4]
            #defvector_init_dvdy = P_old[5]

            #gradient_buffer = np.zeros((6,1))
            #gradient_buffer[0] = dP[0]
            #gradient_buffer[1] = dP[1]
            #gradient_buffer[2] = dP[2]
            #gradient_buffer[4] = dP[3]
            #gradient_buffer[3] = dP[4]
            #gradient_buffer[5] = dP[5]

            #P_old[0] = defvector_init_u - ((defvector_init_dudx + 1)*(gradient_buffer[0] + gradient_buffer[0]*gradient_buffer[5] - gradient_buffer[1]*gradient_buffer[3]))/(gradient_buffer[2] + gradient_buffer[5] + gradient_buffer[2]*gradient_buffer[5] - gradient_buffer[3]*gradient_buffer[4] + 1) - (defvector_init_dudy*(gradient_buffer[1] - gradient_buffer[0]*gradient_buffer[4] + gradient_buffer[1]*gradient_buffer[2]))/(gradient_buffer[2] + gradient_buffer[5] + gradient_buffer[2]*gradient_buffer[5] - gradient_buffer[3]*gradient_buffer[4] + 1) # u
            #P_old[1] = defvector_init_v - ((defvector_init_dvdy + 1)*(gradient_buffer[1] - gradient_buffer[0]*gradient_buffer[4] + gradient_buffer[1]*gradient_buffer[2]))/(gradient_buffer[2] + gradient_buffer[5] + gradient_buffer[2]*gradient_buffer[5] - gradient_buffer[3]*gradient_buffer[4] + 1) - (defvector_init_dvdx*(gradient_buffer[0] + gradient_buffer[0]*gradient_buffer[5] - gradient_buffer[1]*gradient_buffer[3]))/(gradient_buffer[2] + gradient_buffer[5] + gradient_buffer[2]*gradient_buffer[5] - gradient_buffer[3]*gradient_buffer[4] + 1)#v
            #P_old[2] = ((gradient_buffer[5] + 1)*(defvector_init_dudx + 1))/(gradient_buffer[2] + gradient_buffer[5] + gradient_buffer[2]*gradient_buffer[5] - gradient_buffer[3]*gradient_buffer[4] + 1) - (gradient_buffer[4]*defvector_init_dudy)/(gradient_buffer[2] + gradient_buffer[5] + gradient_buffer[2]*gradient_buffer[5] - gradient_buffer[3]*gradient_buffer[4] + 1) - 1 # du/dx
            #P_old[4] = (defvector_init_dudy*(gradient_buffer[2] + 1))/(gradient_buffer[2] + gradient_buffer[5] + gradient_buffer[2]*gradient_buffer[5] - gradient_buffer[3]*gradient_buffer[4] + 1) - (gradient_buffer[3]*(defvector_init_dudx + 1))/(gradient_buffer[2] + gradient_buffer[5] + gradient_buffer[2]*gradient_buffer[5] - gradient_buffer[3]*gradient_buffer[4] + 1) # du/dy
            #P_old[3] = (defvector_init_dvdx*(gradient_buffer[5] + 1))/(gradient_buffer[2] + gradient_buffer[5] + gradient_buffer[2]*gradient_buffer[5] - gradient_buffer[3]*gradient_buffer[4] + 1) - (gradient_buffer[4]*(defvector_init_dvdy + 1))/(gradient_buffer[2] + gradient_buffer[5] + gradient_buffer[2]*gradient_buffer[5] - gradient_buffer[3]*gradient_buffer[4] + 1) # dv/dx
            #P_old[5] = ((gradient_buffer[2] + 1)*(defvector_init_dvdy + 1))/(gradient_buffer[2] + gradient_buffer[5] + gradient_buffer[2]*gradient_buffer[5] - gradient_buffer[3]*gradient_buffer[4] + 1) - (gradient_buffer[3]*defvector_init_dvdx)/(gradient_buffer[2] + gradient_buffer[5] + gradient_buffer[2]*gradient_buffer[5] - gradient_buffer[3]*gradient_buffer[4] + 1) - 1 # dv/dy


            P_old = P_old - dP

            Tm_X, Tm_Y = DeformationGrad(P_old, Mesh_allX, Mesh_allY, dx, dy, I_i, I_f, Def_order)

                                      #-1                #-3
            Points = np.transpose( np.vstack((Tm_X.flatten('F'), Tm_Y.flatten('F'))) )

            #try:
            Values = it.Interp_spline(Points, InterpS['coeff']) # USE INTERP S
            #except:
            #    print('Error in the interpolation')
            #    return np.zeros((12,1)),0

            I_f_indexed = np.reshape(Values,(template.shape[0],template.shape[1]),order='F')
            I_f_indexed = I_f_indexed*GaussianWindow #* OutPoints.astype(float)


            Trials = Trials + 1

            #print 'Interation: '+ str(Trials)+ '. Correlation: '+ str(CorrT)
            #print 'norm dP: '+ str(norm(dP)), '\n')


            dP_old = dP

            P = P_old

    # Correlation

    C = np.sum((((I_f_indexed[Template_mask==1] - np.mean(I_f_indexed[Template_mask==1])) / np.sqrt(
                np.sum((I_f_indexed[Template_mask==1] - np.mean(I_f_indexed[Template_mask==1])) ** 2))) - (
                        template[Template_mask==1] - template_Mean) / template_square_mean) ** 2)


    CorrT = 1 - 0.5 * C
    return P,CorrT

#@jit(numba.f8[:,:](numba.f8[:,:],numba.f8[:,:]))
def GradientGN(I_i, I_f,Fgrad,Template_mask): #Grad

    # GetGradient

    import pdb
    #pdb.set_trace()
    try:
        Normalized_difference = ((I_i[Template_mask==1]) - np.mean(I_i[Template_mask==1])) / np.sqrt( np.sum( (I_i[Template_mask==1] -
                                np.mean(I_i[Template_mask==1]))** 2 ))  - (I_f[Template_mask==1] - np.mean(I_f[Template_mask==1])).flatten() / \
                               np.sqrt(np.sum( ( (I_f[Template_mask==1]) -np.mean(I_f[Template_mask==1]))** 2) )
    except:
        pdb.set_trace()

    Grad = np.zeros( ( len(Fgrad),1) )

    for idx in range(len(Fgrad)):
        Grad[idx] = 2 / np.sqrt(np.sum((I_i[Template_mask==1] - np.mean(I_i[Template_mask==1])) ** 2)) * np.sum( Normalized_difference * Fgrad[idx][Template_mask==1])


    return Grad

#@jit(numba.f8[:,:](numba.f8[:,:]))
def HessianGN(I_i, Fgrad,Template_mask): #Hessian


    Dif_factor = 2 / np.sum( (I_i[Template_mask==1] - np.mean(I_i[Template_mask==1]) )** 2  )
    Hessian = np.zeros((len(Fgrad),len(Fgrad)))


    #for idxX in range(len(Fgrad)):
        #for idxY in range(len(Fgrad)):
            #Hessian[idxX, idxY] = Dif_factor * np.sum( Fgrad[idxX] * Fgrad[idxY] )
    for idxY in range(len(Fgrad)):
        for idxX in range(idxY+1):

            Hessian[idxX, idxY] = Dif_factor * np.sum( Fgrad[idxX] * Fgrad[idxY] )

    return Hessian

#@jit(nopython=True)
def DeformationGrad(P, Mesh_allX, Mesh_allY, delta_x, delta_y, I_i, I_f,Def_order):

    U0 = P[0]
    V0 = P[1]
    du_x = P[2]
    dv_x = P[3]
    du_y = P[4]
    dv_y = P[5]

    if Def_order == 0:
        du_x2= 0.
        dv_x2= 0.
        du_y2= 0.
        dv_y2= 0.
        du_xy= 0.
        dv_xy= 0.

    elif Def_order == 1:
        du_x2= P[6]
        dv_x2= P[7]
        du_y2= P[8]
        dv_y2= P[9]
        du_xy= P[10]
        dv_xy= P[11]


        # New position and Deformation
    de_x =  (U0  + du_x * delta_x+ du_y * delta_y+ 1. / 2 * du_x2 * (delta_x**2) + 1. / 2 * du_y2 * delta_y** 2.+
             du_xy * delta_x * delta_y)
    #de_x = np.zeros((delta_x.shape[0],delta_x.shape[1]), dtype=np.float64)
    #de_y = np.zeros((delta_x.shape[0],delta_x.shape[1]), dtype=np.float64)

    #for idxX in range(delta_x.shape[0]):
        #for idxY in range(delta_x.shape[1]):
            #de_x[idxX,idxY] = U0 + du_x * delta_x[idxX,idxY] + du_y * delta_y[idxX,idxY] + 1. / 2 * du_x2 * (delta_x[idxX,idxY] ** 2) + 1. / 2 * du_y2 * delta_y[idxX,idxY] ** 2. +\
                              #du_xy * delta_x[idxX,idxY] * delta_y[idxX,idxY]

    de_y =    (V0 + dv_x * delta_x + dv_y * delta_y + 1. / 2 * dv_x2 * (delta_x** 2.)+1. / 2 * dv_y2 * delta_y**2.+
        dv_xy * delta_x * delta_y)

    #for idxX in range(delta_x.shape[0]):
        #for idxY in range(delta_x.shape[1]):
            #de_y[idxX,idxY] = V0 + dv_x * delta_x[idxX,idxY] + dv_y * delta_y[idxX,idxY] + 1. / 2 * dv_x2 * (delta_x[idxX,idxY] ** 2.) + 1. / 2 * dv_y2 * delta_y[idxX,idxY] ** 2. +\
            #dv_xy * delta_x[idxX,idxY] * delta_y[idxX,idxY]

    Tm_X =  Mesh_allX+de_x
    Tm_Y =  Mesh_allY+de_y

    # Check for outside points
    #OutPoints_Y = np.logical_or(Tm_Y < 1 , Tm_Y > I_i.shape[0])
    #OutPoints_X = np.logical_or(Tm_X < 1 , Tm_X > I_i.shape[1])

    #OutPoints =  ~np.logical_and(OutPoints_Y , OutPoints_X)

    # Avoid distortion over the image
    #if np.max(Tm_X) > I_f.shape[1]:

        #Tm_X[Tm_X > I_f.shape[1]] = 0


    #if np.max(Tm_Y) > I_f.shape[0]:
        #Tm_Y[Tm_Y > I_f.shape[0]] = 0



    #Tm_X[Tm_X < 1] = 1
    #Tm_Y[Tm_Y < 1] = 1

    return Tm_X, Tm_Y#, OutPoints

def find(A,x):
    temp = []
    indexes = []
    for idx in range(len(A)):
        if A[idx] == x:
            temp.append(A[idx])
            indexes.append(idx)
    A = np.array(temp)
    return A,indexes

def getInterp(I,Padding,flag):


            # Interpolate
            I_i_padded = np.lib.pad(I.astype(float), (Padding,Padding),'constant', constant_values=0) #edges

           # Collums
            b0 = np.array([1./120., 13./60., 11./20., 13./60., 1./120., 0])
            b0 = b0.reshape((1, 6))
            b0 = np.hstack( (b0,np.zeros((1, I_i_padded.shape[0]-6 ))) )
            Kernel_FFT = np.fft.fft(b0)

            C_coef = np.zeros((I_i_padded.shape[0], I_i_padded.shape[1]))

            for idxR in range((I_i_padded.shape[1])):
               C_coef[:,idxR] = np.abs( np.fft.ifft( np.fft.fft(I_i_padded[:,idxR]) /Kernel_FFT ) )

            # Save data
            coeff = C_coef #InterpS.
            pad = Padding #InterpS.

            np.savetxt("coeff_python.csv",coeff,delimiter=',')

            #QK
            QK = np.array( [[1./120,  13./60,  11./20, 13./60,  1./120, 0], #InterpS.
             [-1./24,   -5./12,    0,    5./12,  1./24,  0],
              [1./12,    1./6,   -1./2,   1./6,   1./12,  0],
             [-1./12,    1./6 ,    0,   -1./6 ,  1./12,  0],
              [1./24,   -1./6,    1./4,  -1./6 ,  1./24,  0],
             [-1./120,   1./24,  -1./12,  1./12, -1./24, 1./120] ]  )


            # Pre calculate things
            CoCoeff = []

            for idxXX in range(I.shape[0]):
                CoCoeff.append([])
                for idxYY in range(I.shape[1]):
                    CoCoeff[idxXX].append([]) #InterpS.

            if flag == 1: # If reference image, if not, no need for gradients
                Fx = np.zeros((I.shape[0],I.shape[1])) #InterpS.
                Fy = np.zeros((I.shape[0],I.shape[1])) #InterpS.


            for idxXX in range(I.shape[0]):
                for idxYY in range(I.shape[1]):

                    y_tilda_floor = idxXX
                    x_tilda_floor = idxYY

                    top = int(y_tilda_floor +Padding-2)
                    left = int(x_tilda_floor +Padding-2)
                    bottom = int(y_tilda_floor +Padding+3)
                    right = int(x_tilda_floor +Padding+3)

                    CoCoeff[idxXX][idxYY] = np.matmul(np.matmul(QK,coeff[np.ix_([range(top+1,bottom+1+1)][0],[range(left+1,right+1+1)][0])]) , np.transpose(QK)) #InterpS.

                    Fx[idxXX,idxYY] = np.matmul(  np.matmul(  np.matmul(  np.matmul( np.array([1,0 ,0, 0, 0, 0]),QK) , coeff[np.ix_([range(top+1,bottom+1+1)][0],[range(left+1,right+1+1)][0])]) ,np.transpose(QK)) , np.transpose( np.array([0, 1, 0, 0, 0, 0])) ) #InterpS.
                    Fy[idxXX,idxYY] = np.matmul(  np.matmul(  np.matmul(  np.matmul( np.array([0,1 ,0, 0, 0, 0]),QK) , coeff[np.ix_([range(top+1,bottom+1+1)][0],[range(left+1,right+1+1)][0])]) ,np.transpose(QK)) , np.transpose( np.array([1, 0, 0, 0, 0, 0])) )  #InterpS.

            InterpS = {'Fx': Fx, 'Fy': Fy, 'QK': QK,'coeff':CoCoeff,'pad':pad }
            return InterpS

# If one just want to calculate DIC
def DIC_direct(P,I_f, I_i, template, Center_position, InterpS, Def_order):

    w = (template.shape[0] - 1) / 2
    Diff_vector = range(int(-w),int(w)+1)
    dx = npl.repmat(Diff_vector, 2 * w + 1, 1)
    dy = np.transpose( dx )


    #template = double(template)
    i = Center_position[0]
    j = Center_position[1]

    Mesh_allX, Mesh_allY = np.meshgrid(range(int(i - w), int(i + w) + 1), range(int(j - w), int(j + w) + 1))  #Mesh_allY = np.transpose(Mesh_allY)

    Mesh_allX = Mesh_allX
    Mesh_allY = Mesh_allY

    Tm_X, Tm_Y = DeformationGrad(P, Mesh_allX, Mesh_allY, dx, dy, I_i, I_f, Def_order)

    # -1                #-3
    Points = np.transpose(np.vstack((Tm_X.flatten('F'), Tm_Y.flatten('F'))))

    Values = it.Interp_spline(Points, InterpS['coeff'])  # USE INTERP S
    I_f_indexed = np.reshape(Values, (template.shape[0], template.shape[1]), order='F')
    I_f_indexed = I_f_indexed  # * OutPoints.astype(float)

    # Correlation


    template_Mean = np.mean(template)
    template_square_mean = np.sqrt(np.sum((template - np.mean(template)) ** 2))

    C = np.sum((((I_f_indexed - np.mean(I_f_indexed)) / np.sqrt(np.sum((I_f_indexed - np.mean(I_f_indexed)) ** 2))) - (
        template - template_Mean) / template_square_mean) ** 2)

    return C


