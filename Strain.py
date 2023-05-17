import wx #check
import os #check
import numpy as np #check
import joblib #check
import numpy.matlib as npl #check
import scipy.linalg as linalg #check
import scipy #check

import wx.xrc #checar necessidade
import wx.grid #checar necessidade

class StrainMixin:
    
    def Mouse_movement(self,e):
        try:
            self.StatusBar.SetStatusText('X: ' + str(np.round(100 * e.xdata) / 100) + '. Y: ' + str(
                np.round(100 * e.ydata) / 100) + 'Value: ' +str( np.round(100 * e.ydata)) , 1)
        except:
            print("bup")
            return
        
    def CB_gaussianCorrection_Callback(self,e):

        if self.CB_gaussianCorrection.IsChecked():
            # Exx
            dlg = wx.FileDialog(self, "Choose the Exx regressor:", 'C:', "", "*.*", wx.FD_OPEN)
            if dlg.ShowModal() == wx.ID_OK:
                self.filename = dlg.GetFilename()
                self.dirname = dlg.GetDirectory()
                f = os.path.join(self.dirname, self.filename)
                self.GP_x = joblib.load(f)
                
            # Eyy
            dlg = wx.FileDialog(self, "Choose the Eyy regressor:", 'C:', "", "*.*", wx.FD_OPEN)
            if dlg.ShowModal() == wx.ID_OK:
                self.filename = dlg.GetFilename()
                self.dirname = dlg.GetDirectory()
                f = os.path.join(self.dirname, self.filename)
                self.GP_y = joblib.load(f)

    def PB_filterStrain_Callback(self,e):

        # Parameter Inputs
        Filter_size = int(self.ET_filterSizeStrain.GetLineText(0))
        Algo = self.CB_filtersDisp1.GetSelection()
        Let_us = self.ET_usChooseStrain.IsChecked()
        Smooth = self.CB_smoothStrain.IsChecked()
        Filter = self.CB_filtersStrain.GetSelection()
        Value = float(self.ET_filterValueStrain.GetLineText(0))
        All_images = self.CB_allImages.IsChecked()

        if Algo == 0: # Direct Gradient Algorithm

            Length = 1
            if All_images == True:
                Length = self.Len_images
                
            ######### Filter Deformation########
            for idx in range(Length):

                if All_images == True:  # If not all, just pass the actual index
                    self.Actual_Image = idx

                # X
                Def_xy,Def_xx = np.gradient(self.disX_smooth[self.Actual_Image])

                # Y
                Def_yy,Def_yx = np.gradient(self.disY_smooth[self.Actual_Image])

                self.Exx[self.Actual_Image] = 1. / 2 * (2 * Def_xx + Def_xx** 2 + Def_xy** 2) *self.DIC_calibration_factor
                self.Exy[self.Actual_Image] = 1. / 2 * (Def_xy + Def_yx + Def_xx* Def_xy + Def_yx* Def_yy) *self.DIC_calibration_factor
                self.Eyy[self.Actual_Image] = 1. / 2 * (2 * Def_yy + Def_xy** 2 + Def_yy** 2) *self.DIC_calibration_factor

        elif Algo == 1: # Plane Fit Algorithm

            Length = 1
            if All_images == True:
                Length = self.Len_images
                
            ######### Filter Displacements########
            for idx in range(Length):

                if All_images == True:  # If not all, just pass the actual index
                    self.Actual_Image = idx

                # Compensate the NAN
                # X
                U = self.disX_smooth[self.Actual_Image].copy()
                Def_xxU, Def_xyU = self.sgolay2d(U,Filter_size,order=2,derivative='both' )#self.Plane_fit(U,Filter_size)

                V = self.disX_smooth[self.Actual_Image].copy() * 0 + 1
                V[U != self.disX_smooth[self.Actual_Image].copy()] = 0
                Def_xxV, Def_xyV = self.sgolay2d(V,Filter_size,order=2,derivative='both' )#self.Plane_fit(V, Filter_size)

                Def_xxV[Def_xxV==0] = 1
                Def_xyV[Def_xyV == 0] = 1

                Def_xx = Def_xxU #/Def_xxV
                Def_xy = Def_xyU #/Def_xyV

                # Y
                U = self.disY_smooth[self.Actual_Image].copy()
                Def_yxU, Def_yyU = self.sgolay2d(U,Filter_size,order=2,derivative='both')#self.Plane_fit(U, Filter_size)

                V = self.disY_smooth[self.Actual_Image].copy() * 0 + 1
                V[U != self.disY_smooth[self.Actual_Image].copy()] = 0
                Def_yxV, Def_yyV = self.sgolay2d(V,Filter_size,order=2,derivative='both')#self.Plane_fit(V, Filter_size)

                Def_yxV[Def_yxV == 0] = 1
                Def_yyV[Def_yyV == 0] = 1

                Def_yx = Def_yxU #/Def_yxV
                Def_yy = Def_yyU #/Def_yyV

                # Calculate deformation
                self.Exx[self.Actual_Image] =  1. / 2 * (2 * Def_xx + Def_xx ** 2 + Def_xy ** 2) *self.DIC_calibration_factor
                self.Exy[self.Actual_Image] =  1. / 2 * (Def_xy + Def_yx + Def_xx * Def_xy + Def_yx * Def_yy) *self.DIC_calibration_factor
                self.Eyy[self.Actual_Image] =  1. / 2 * (2 * Def_yy + Def_xy ** 2 + Def_yy ** 2) *self.DIC_calibration_factor

        if Smooth:
            self.Exx[self.Actual_Image] = self.SmoothT(self.Exx[self.Actual_Image], Filter, Value)
            self.Eyy[self.Actual_Image] = self.SmoothT(self.Eyy[self.Actual_Image], Filter, Value)
            self.Exy[self.Actual_Image] = self.SmoothT(self.Exy[self.Actual_Image], Filter, Value)


        if self.CB_gaussianCorrection.IsChecked():

            X = np.hstack( ( self.All_parameters[0].reshape(-1,1),self.All_parameters[1].reshape(-1,1),
                           self.All_parameters[2].reshape(-1, 1),self.All_parameters[3].reshape(-1,1),
                           self.All_parameters[4].reshape(-1, 1),self.All_parameters[5].reshape(-1,1)))
            
            Indexes = ~np.isnan(X[:,0])
            X = X[Indexes,:] #.reshape(-1,5)
            self.Exx_correction = self.GP_x.predict(X)
            self.Eyy_correction = self.GP_y.predict(X)

            Dumb = np.zeros((self.Exx[self.Actual_Image].shape[0]*self.Exx[self.Actual_Image].shape[1],1))
            Dumb[Indexes] += self.Exx_correction.reshape(-1,1)
            self.Exx_correction = Dumb*self.DIC_calibration_factor_ML

            if self.ET_usChooseStrain.IsChecked():
                self.Exx_correction = self.SmoothT(self.Exx_correction, Filter, Filter_size)

            self.Exx[self.Actual_Image] += Dumb.reshape((self.Exx[self.Actual_Image].shape[0],self.Exx[self.Actual_Image].shape[1]))

            Dumb = np.zeros((self.Exx[self.Actual_Image].shape[0] * self.Exx[self.Actual_Image].shape[1], 1))
            Dumb[Indexes] += self.Eyy_correction.reshape(-1,1)

            self.Eyy_correction = Dumb*self.DIC_calibration_factor_ML

            if self.ET_usChooseStrain.IsChecked():
                self.Eyy_correction = self.SmoothT(self.Eyy_correction, Filter, Filter_size)

            self.Eyy[self.Actual_Image] += Dumb.reshape((self.Exx[self.Actual_Image].shape[0],self.Exx[self.Actual_Image].shape[1]))

            self.Exy[self.Actual_Image] += (1/2*(self.Eyy_correction+self.Exx_correction)).reshape((self.Exx[self.Actual_Image].shape[0],self.Exx[self.Actual_Image].shape[1]))

        if False:
            self.Exx[self.Actual_Image] = self.SmoothT(self.Exx[self.Actual_Image], Filter, Value)
            self.Eyy[self.Actual_Image] = self.SmoothT(self.Eyy[self.Actual_Image], Filter, Value)
            self.Exy[self.Actual_Image] = self.SmoothT(self.Exy[self.Actual_Image], Filter, Value)

        masked_array = np.ma.array(self.Eyy[self.Actual_Image], mask=self.MaskY)
        self.plot1Strain.draw(self.I_all[self.Actual_Image], self.Y + self.disX_smooth[self.Actual_Image],
                                  self.X + self.disY_smooth[self.Actual_Image], masked_array, 'Eyy', 'X', 'Y')

        self.plot1Res.draw(self.I_all[self.Actual_Image], self.Y + self.disX_smooth[self.Actual_Image],
                               self.X + self.disY_smooth[self.Actual_Image], masked_array, 'Eyy', 'X', 'Y')

        self.plot1DataStrain = masked_array
        self.plot1Strain.canvas.mpl_connect('motion_notify_event', self.VisuStrain1)

        # Save Eyy
        Temp = np.ma.array(self.Eyy[self.Actual_Image], mask=self.MaskY)
        np.savetxt('Eyy_Filter.csv', np.ma.filled(Temp, np.nan), delimiter=',')


    def Plane_fit(self,I,Radius):

        Def_y = np.zeros((I.shape[0],I.shape[1]))
        Def_x = np.zeros((I.shape[0],I.shape[1]))

        I_i_padded = np.lib.pad(I.astype(float), (Radius, Radius), 'edge')

        t = 0.
        for idxI in range(0,I.shape[0]):
             for idxJ in range(0,I.shape[1]):

                 win = (Radius) /2
                 dx = npl.repmat(range(-(win),win+1), 2 * win + 1, 1)
                 dy = np.transpose(dx)

                 A = np.hstack( (np.ones((len(dx.flatten()),1)),np.reshape(dx.flatten(),(len(dx.flatten()),1))
                                 ,np.reshape(dy.flatten(),(len(dy.flatten()),1)) ))

                # X
                 Bx = I_i_padded[np.ix_( range(int(idxI),int(idxI + Radius)),range(idxJ,int(idxJ + Radius))  ) ]


                 C = np.matmul(linalg.pinv(A),Bx.flatten())

                 Def_x[idxI,idxJ] = C[1]
                 Def_y[idxI,idxJ] = C[2]

                 print(float(t)/(I.shape[0]*I.shape[1]))
                 t += 1

        return Def_y,Def_x

    def sgolay2d(self,z, window_size, order, derivative=None):
        
        # number of terms in the polynomial expression
        n_terms = (order + 1) * (order + 2) / 2.0

        if window_size % 2 == 0:
            raise ValueError('window_size must be odd')

        if window_size ** 2 < n_terms:
            raise ValueError('order is too high for the window size')

        half_size = window_size // 2

        # exponents of the polynomial.
        # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
        # this line gives a list of two item tuple. Each tuple contains
        # the exponents of the k-th term. First element of tuple is for x
        # second element for y.
        # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
        exps = [(k - n, n) for k in range(order + 1) for n in range(k + 1)]

        # coordinates of points
        ind = np.arange(-half_size, half_size + 1, dtype=np.float64)
        dx = np.repeat(ind, window_size)
        dy = np.tile(ind, [window_size, 1]).reshape(window_size ** 2, )

        # build matrix of system of equation
        A = np.empty((window_size ** 2, len(exps)))
        for i, exp in enumerate(exps):
            A[:, i] = (dx ** exp[0]) * (dy ** exp[1])

        # pad input array with appropriate values at the four borders
        new_shape = z.shape[0] + 2 * half_size, z.shape[1] + 2 * half_size
        Z = np.zeros((new_shape))
        # top band
        band = z[0, :]
        Z[:half_size, half_size:-half_size] = band - np.abs(np.flipud(z[1:half_size + 1, :]) - band)
        # bottom band
        band = z[-1, :]
        Z[-half_size:, half_size:-half_size] = band + np.abs(np.flipud(z[-half_size - 1:-1, :]) - band)
        # left band
        band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
        Z[half_size:-half_size, :half_size] = band - np.abs(np.fliplr(z[:, 1:half_size + 1]) - band)
        # right band
        band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
        Z[half_size:-half_size, -half_size:] = band + np.abs(np.fliplr(z[:, -half_size - 1:-1]) - band)
        # central band
        Z[half_size:-half_size, half_size:-half_size] = z

        # top left corner
        band = z[0, 0]
        Z[:half_size, :half_size] = band - np.abs(np.flipud(np.fliplr(z[1:half_size + 1, 1:half_size + 1])) - band)
        # bottom right corner
        band = z[-1, -1]
        Z[-half_size:, -half_size:] = band + np.abs(
            np.flipud(np.fliplr(z[-half_size - 1:-1, -half_size - 1:-1])) - band)

        # top right corner
        band = Z[half_size, -half_size:]
        Z[:half_size, -half_size:] = band - np.abs(np.flipud(Z[half_size + 1:2 * half_size + 1, -half_size:]) - band)
        # bottom left corner
        band = Z[-half_size:, half_size].reshape(-1, 1)
        Z[-half_size:, :half_size] = band - np.abs(np.fliplr(Z[-half_size:, half_size + 1:2 * half_size + 1]) - band)

        # solve system and convolve
        if derivative == None:
            m = np.linalg.pinv(A)[0].reshape((window_size, -1))
            return scipy.signal.fftconvolve(Z, m, mode='valid')#convolve_fft(Z, m)#
        elif derivative == 'col':
            c = np.linalg.pinv(A)[1].reshape((window_size, -1))
            return scipy.signal.fftconvolve(Z, -c, mode='valid')#convolve_fft(Z, -c)#
        elif derivative == 'row':
            r = np.linalg.pinv(A)[2].reshape((window_size, -1))
            return scipy.signal.fftconvolve(Z, -r, mode='valid')#convolve_fft(Z, -r)#
        elif derivative == 'both':
            c = np.linalg.pinv(A)[1].reshape((window_size, -1))
            r = np.linalg.pinv(A)[2].reshape((window_size, -1))
            return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid') #AAA,BBB#

    def PB_visuStrain_Callback(self,e):

        Value = self.CB_strainShowChoice.GetSelection()

        if Value == 0: # Exx
            masked_array = np.ma.array(self.Exx[self.Actual_Image], mask=self.MaskY)
            self.plot1Strain.draw(self.I_all[self.Actual_Image],self.Y+ self.disX_smooth[self.Actual_Image], self.X+ self.disY_smooth[self.Actual_Image], masked_array, 'Exx', 'X', 'Y')
        elif Value == 1: # Eyy
            masked_array = np.ma.array(self.Eyy[self.Actual_Image], mask=self.MaskY)
            self.plot1Strain.draw(self.I_all[self.Actual_Image],self.Y+ self.disX_smooth[self.Actual_Image], self.X+ self.disY_smooth[self.Actual_Image], masked_array, 'Eyy', 'X', 'Y')
        elif Value == 2: # Exy
            masked_array = np.ma.array(self.Exy[self.Actual_Image], mask=self.MaskY)
            self.plot1Strain.draw(self.I_all[self.Actual_Image], self.Y+ self.disX_smooth[self.Actual_Image], self.X+ self.disY_smooth[self.Actual_Image], masked_array, 'Exy', 'X', 'Y')
        elif Value == 3: # fit Exx
            masked_array = np.ma.array(self.Exx_fit[self.Actual_Image], mask=self.MaskY)
            self.plot1Strain.draw(self.I_all[self.Actual_Image], self.Y+ self.disX_smooth[self.Actual_Image], self.X+ self.disY_smooth[self.Actual_Image], masked_array, 'Exx fit', 'X', 'Y')
        elif Value == 4: # fit Eyy
            masked_array = np.ma.array(self.Eyy_fit[self.Actual_Image], mask=self.MaskY)
            self.plot1Strain.draw(self.I_all[self.Actual_Image], self.Y+ self.disX_smooth[self.Actual_Image], self.X+ self.disY_smooth[self.Actual_Image], masked_array, 'Eyy fit', 'X', 'Y')
        elif Value == 5: # fit Exy
            masked_array = np.ma.array(self.Exy_fit[self.Actual_Image], mask=self.MaskY)
            self.plot1Strain.draw(self.I_all[self.Actual_Image], self.Y+ self.disX_smooth[self.Actual_Image], self.X+ self.disY_smooth[self.Actual_Image], masked_array, 'Exy fit', 'X', 'Y')
        elif Value == 6: #Exx_correction
            masked_array = np.ma.array(np.abs(self.Exx_correction.reshape((self.Exx[self.Actual_Image].shape[0],self.Exx[self.Actual_Image].shape[1])) ),  mask=self.MaskY)
            self.plot1Strain.draw(self.I_all[self.Actual_Image], self.Y + self.disX_smooth[self.Actual_Image],self.X + self.disY_smooth[self.Actual_Image], masked_array, 'Exx Deviation', 'X', 'Y')
        elif Value == 7:  # Exx_correction
            masked_array = np.ma.array(np.abs(self.Eyy_correction.reshape((self.Exx[self.Actual_Image].shape[0],self.Exx[self.Actual_Image].shape[1])) ), mask=self.MaskY)
            self.plot1Strain.draw(self.I_all[self.Actual_Image], self.Y + self.disX_smooth[self.Actual_Image],self.X + self.disY_smooth[self.Actual_Image], masked_array, 'Eyy Deviation', 'X', 'Y')

        self.plot1DataStrain = masked_array
        self.plot1Strain.canvas.mpl_connect('motion_notify_event', self.VisuStrain1)
