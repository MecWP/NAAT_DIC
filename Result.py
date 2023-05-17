import wx #check
import numpy as np #check
from sklearn.linear_model import LinearRegression as LN
from sklearn.gaussian_process import GaussianProcessRegressor as GP
import scipy.stats as st  #check
import matplotlib.pyplot as plt  #check

import wx.xrc #checar necessidade
import wx.grid #checar necessidade

class ResultMixin:
    
    def VisuDisp1(self,e):

        try:
            iY = e.ydata - self.Ryi
            iX = e.xdata - self.Rxi

            print('iY ' + str(iY))
            print('iX ' + str(iX))

            Disp = float(self.plot1Data[int(iY), int(iX)])
            self.ST_GUIStatus.SetLabel('Displacement Value: '+str(np.round(100*Disp)/100))
        except:
            a = 1

    def VisuStrain1(self,e):

        try:
            iY = e.ydata - self.Ryi
            iX = e.xdata - self.Rxi

            print('iY ' + str(iY))
            print('iX ' + str(iX))

            strain = float(self.plot1DataStrain[int(iY), int(iX)])
            self.ST_GUIStatus.SetLabel('Strain Value in microstrains: ' + str( np.round(100*strain)/100))
        except:
            a = 1

    def PB_SinglePointResult_Callback(self,e):
        self.Strain = 0
        self.plot1Res.canvas.mpl_connect('button_press_event', self.onClickPointResult)

        # Update status button
        self.ST_GUIStatus.SetLabel('Select point')

    def onClickPointResult(self,e):
        self.Strain_one_Y = e.ydata - self.Ryi
        self.Strain_one_X = e.xdata - self.Rxi

        print('First: ' + str(self.Strain_one_Y))
        print('Second: ' + str(self.Strain_one_X))
        print('Ryi: ' + str(self.Ryi) + 'Rxi' + str(self.Rxi))
        print('y data: ' + str(e.ydata) + 'X data' + str(e.xdata))


        Exx = np.zeros((self.Len_images, 1))
        Eyy = np.zeros((self.Len_images, 1))
        Exy = np.zeros((self.Len_images, 1))

        # Check to see if fit result is used
        Fitted = self.CB_resultsFit.IsChecked()

        for idx in range(self.Len_images):
            if Fitted == False:
                Exx[idx] = self.Exx[idx][int(self.Strain_one_Y), int(self.Strain_one_X)]
                Eyy[idx] = self.Eyy[idx][int(self.Strain_one_Y), int(self.Strain_one_X)]
                Exy[idx] = self.Exy[idx][int(self.Strain_one_Y), int(self.Strain_one_X)]
            else:
                Exx[idx] = self.Exx_fit[idx][int(self.Strain_one_Y), int(self.Strain_one_X)]
                Eyy[idx] = self.Eyy_fit[idx][int(self.Strain_one_Y), int(self.Strain_one_X)]
                Exy[idx] = self.Exy_fit[idx][int(self.Strain_one_Y), int(self.Strain_one_X)]

        # Strength/Tension vector
        dF = float(self.ET_forceStep.GetLineText(0))
        Area = float(self.ET_areaResult.GetLineText(0))
        F = np.arange(dF, dF * self.Len_images + dF, dF)
        F = np.reshape(F/Area, (len(F), 1))

        # Fit the deformation with a linear regression
        ### X
        # Gauss
        regX = GP()
        regX = regX.fit(F, Exx)
        Exx_fit, stdX = regX.predict(F, return_std=True)
        # Linear Regression
        regX = LN()
        regX = regX.fit(F, Exx)
        Exx_fit = regX.predict(F)
        stdX = np.std(abs(Exx_fit-Exx))

        ### Y
        # Gauss
        regY = GP()
        regY = regY.fit(F, Eyy)
        Eyy_fit, stdY = regY.predict(F, return_std=True)
        # Linear Regression
        regY = LN()
        regY = regY.fit(F, Eyy)
        Eyy_fit = regY.predict(F)
        stdY = np.std(abs(Eyy-Eyy_fit))

        ### XY
        regXY = GP()
        regXY = regXY.fit(F, Exy)
        Exy_fit, stdXY = regXY.predict(F, return_std=True)
        # Linear Regression
        regXY = LN()
        regXY = regXY.fit(F, Exy)
        Exy_fit = regXY.predict(F)
        stdXY = np.std(abs(Exy_fit-Exy))

        User_conf = float(self.ET_confidence.GetLineText(0))
        Confidence = st.norm.ppf(1 - (1 - User_conf) / 2)

        # Get fitted plot if requested
        if Fitted:
            stdX = self.StdX_fit[int(self.Strain_one_Y), int(self.Strain_one_X)]
            stdY = self.StdY_fit[int(self.Strain_one_Y), int(self.Strain_one_X)]
            stdXY = self.StdXY_fit[int(self.Strain_one_Y), int(self.Strain_one_X)]

        # PLot
        Override = 1
        self.plot2Res.draw(F, [Exx, Eyy, Exy], 'Virtual Strain Gauge Deformation', 'Strength', 'Deformation',
                           ['Ex', 'Ey', 'Exy'], Override)
        Override = 0
        self.plot2Res.draw(F, [Exx_fit, Eyy_fit, Exy_fit], 'Virtual Strain Gauge Deformation', 'Strength', 'Deformation',
                           [['Ex','Ex fit'], ['Ey','Ey fit'], ['Exy','Exy fit']], Override)
        # Std
        if self.CB_uncertainty.IsChecked() == True:
            self.plot2Res.draw(F,
                               [Exx_fit + stdX * Confidence, Eyy_fit + stdY * Confidence, Exy_fit + stdXY * Confidence],
                               'Virtual Strain Gauge Deformation', 'Strength', 'Deformation', ['Ex', 'Ey', 'Exy'],
                               Override)
            self.plot2Res.draw(F,
                               [Exx_fit - stdX * Confidence, Eyy_fit - stdY * Confidence, Exy_fit - stdXY * Confidence],
                               'Virtual Strain Gauge Deformation', 'Strength'
                               , 'Deformation', [['Ex', 'Ex fit'], ['Ey', 'Ey fit'], ['Exy', 'Exy fit']], Override)

        # Save in root
        self.Tree_results.DeleteChildren(self.StrainGates_root)

        self.StrainGates_X = self.Tree_results.AppendItem(self.StrainGates_root, '1')

        # Get the E
        YoungX = (F[-1] - dF) / Exx_fit[-1]
        YoungY = (F[-1] - dF) / Eyy_fit[-1]
        Poisson = - np.mean(Exx_fit / Eyy_fit)
        Poisson2 = - np.mean(Eyy_fit / Exx_fit)
        # Exx Eyy Exy
        self.YoungX = self.Tree_results.AppendItem(self.StrainGates_X, 'Young Modulus[Using Ex]: ' + str(
                np.round(YoungX) / (10 ** 7)) + ' GPa')
        self.YoungY = self.Tree_results.AppendItem(self.StrainGates_X, 'Young Modulus[Using Ey]: ' + str(
                np.round(YoungY) / (10 ** 7)) + ' GPa')
        self.Poisson = self.Tree_results.AppendItem(self.StrainGates_X,
                                                        'Poisson Coefficient[Ex/Ey]: ' + str(Poisson))
        self.Poisson = self.Tree_results.AppendItem(self.StrainGates_X,
                                                        'Poisson Coefficient[Ey/Ex]: ' + str(Poisson2))

    def PB_fitAll_Callback(self, e):

        # Force vector to regression
        dF = float(self.ET_forceStep.GetLineText(0))
        Area = float(self.ET_areaResult.GetLineText(0))
        F = np.arange(dF, dF * self.Len_images + dF, dF)
        F = np.reshape(F / Area, (len(F), 1))

        self.StdX_fit = np.zeros((self.Exx[0].shape[0],self.Exx[0].shape[1]))
        self.StdY_fit = np.zeros((self.Exx[0].shape[0], self.Exx[0].shape[1]))
        self.StdXY_fit = np.zeros((self.Exx[0].shape[0], self.Exx[0].shape[1]))

        # Initialize
        Exx = np.zeros((self.Len_images,1))
        Eyy = np.zeros((self.Len_images, 1))
        Exy = np.zeros((self.Len_images, 1))


        for idx in range(self.Len_images):
            self.Exx_fit[idx] = np.zeros((self.Exx[idx].shape[0],self.Exx[idx].shape[1]))
            self.Eyy_fit[idx] = np.zeros((self.Exx[idx].shape[0], self.Exx[idx].shape[1]))
            self.Exy_fit[idx] = np.zeros((self.Exx[idx].shape[0], self.Exx[idx].shape[1]))

        t = 0
        for idxX in range(self.Exx[0].shape[0]):
            for idxY in range(self.Exx[0].shape[1]):

                for idxI in range(self.Len_images):
                    Exx[idx] = self.Exx[idx][idxX, idxY]
                    Eyy[idx] = self.Eyy[idx][idxX, idxY]
                    Exy[idx] = self.Exy[idx][idxX, idxY]


                # Linear Regression
                # Exx
                if any(Exx == np.nan):

                    Exx_fit = np.zeros((self.Len_images,1))*np.nan
                    stdX = 0
                else:
                    regX = LN()
                    regX = regX.fit(F, Exx)
                    Exx_fit = regX.predict(F)
                    stdX = np.std(abs(Exx_fit - Exx))

                #Eyy
                if any(Eyy == np.nan):

                    Eyy_fit = np.zeros((self.Len_images, 1)) * np.nan
                    stdY = 0
                else:
                    regY = LN()
                    regY = regY.fit(F, Eyy)
                    Eyy_fit = regY.predict(F)
                    stdY = np.std(abs(Eyy_fit - Eyy))

                #Exy
                if any(Exy == np.nan):

                    Exy_fit = np.zeros((self.Len_images, 1)) * np.nan
                    stdXY = 0
                else:
                    regXY = LN()
                    regXY = regXY.fit(F, Exy)
                    Exy_fit = regXY.predict(F)
                    stdXY = np.std(abs(Exy_fit - Exy))

                # Save in the right Image
                for idxI in range(self.Len_images):
                    self.Exx_fit[idxI][idxX,idxY] = Exx_fit[idxI]
                    self.Eyy_fit[idxI][idxX,idxY] = Eyy_fit[idxI]
                    self.Exy_fit[idxI][idxX,idxY] = Exy_fit[idxI]

                self.StdX_fit[idxX,idxY] = stdX
                self.StdY_fit[idxX, idxY] = stdY
                self.StdXY_fit[idxX, idxY] = stdXY

                t += 1
                print(float(t)/(self.Exx[0].shape[0]*self.Exx[0].shape[1]))

    def PB_virtualStrainGate_Callback( self, event ):
        self.Strain = 0
        self.plot1Res.canvas.mpl_connect('button_press_event', self.onClickStrain_gate)

        # Update status button
        self.ST_GUIStatus.SetLabel('Select First point')

    def onClickStrain_gate(self,e):

        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (e.button, e.x, e.y, e.xdata, e.ydata))

        self.Strain += 1

        # Get Strain Gate point
        if self.Strain == 1:
            self.Strain_one_Y = e.ydata - self.Ryi
            self.Strain_one_X = e.xdata - self.Rxi
            self.ST_GUIStatus.SetLabel('Select Second point')
        if self.Strain == 2:
            self.Strain_two_Y = e.ydata - self.Ryi
            self.Strain_two_X = e.xdata - self.Rxi

            # Calculate deformation
            Lx = abs(self.Strain_one_X-self.Strain_two_X)
            Ly = abs(self.Strain_one_Y - self.Strain_two_Y)

            dx = np.zeros((self.Len_images,1))
            dy = np.zeros((self.Len_images, 1))

            for idx in range(self.Len_images):

                dx[idx] = abs(-self.disX_smooth[idx][int(self.Strain_one_Y),int(self.Strain_one_X)] + self.disX_smooth[idx][int(self.Strain_two_Y),int(self.Strain_two_X)])
                dy[idx] = abs(-self.disY_smooth[idx][int(self.Strain_one_Y), int(self.Strain_one_X)] + self.disY_smooth[idx][int(self.Strain_two_Y), int(self.Strain_two_X)])

            Exx = dx/Lx
            Eyy = dy/Ly
            Exy = 1./2*(Exx+Eyy)

            # Strength/Tension vector
            dF = float(self.ET_forceStep.GetLineText(0))
            F = np.arange(dF,dF*self.Len_images+dF,dF)
            F = np.reshape(F,(len(F),1))/float(self.ET_areaResult.GetLineText(0))

            # Fit the deformation with a linear regression
            ### X
            #Gauss
            regX = GP()
            regX = regX.fit(F,Exx)
            Exx_fit,stdX = regX.predict(F,return_std=True)
            stdX = np.std(Exy)
            # Linear Regression
            regX = LN()
            regX = regX.fit(F, Exx)
            Exx_fit = regX.predict(F)

            ### Y
            # Gauss
            regY = GP()
            regY = regY.fit(F,Eyy)
            Eyy_fit,stdY = regY.predict(F,return_std=True)
            stdY = np.std(Exy)
            # Linear Regression
            regY = LN()
            regY = regY.fit(F, Eyy)
            Eyy_fit = regY.predict(F)

            ### XY
            regXY = GP()
            regXY = regXY.fit(F, Exy)
            Exy_fit, stdXY = regXY.predict(F,return_std=True)
            stdXY = np.std(Exy)
            # Linear Regression
            regXY = LN()
            regXY = regXY.fit(F, Exy)
            Exy_fit = regXY.predict(F)


            User_conf = float(self.ET_confidence.GetLineText(0))
            Confidence = st.norm.ppf(1 - (1 - User_conf) / 2)

            Override = 1
            self.plot2Res.draw(F,[Exx,Eyy,Exy],'Virtual Strain Gauge Deformation','Strength','Deformation',['Ex','Ey','Exy'],Override)
            Override = 0
            self.plot2Res.draw(F,[Exx_fit,Eyy_fit,Exy_fit],'Virtual Strain Gauge Deformation','Strength','Deformation',[['Ex','Ex fit'],['Ey','Ey fit'],['Exy','Exy fit']],Override)


            Override = 1
            self.plot2Res.draw(F,[Exx,Eyy,Exy],'Virtual Strain Gauge Deformation','Strength','Deformation',['Ex','Ey','Exy'],Override)
            Override = 0
            self.plot2Res.draw(F,[Exx_fit,Eyy_fit,Exy_fit],'Virtual Strain Gauge Deformation','Strength','Deformation',[['Ex','Ex fit'],['Ey','Ey fit'],['Exy','Exy fit']],Override)

            #Std
            if self.CB_uncertainty.IsChecked() == True:
                self.plot2Res.draw(F,[Exx_fit+stdX*Confidence,Eyy_fit+stdY*Confidence,Exy_fit+stdY*Confidence],'Virtual Strain Gauge Deformation','Strength','Deformation',['Ex','Ey','Exy'],Override)
                self.plot2Res.draw(F,[Exx_fit-stdX*Confidence,Eyy_fit-stdY*Confidence,Exy_fit-stdY*Confidence],'Virtual Strain Gauge Deformation','Strength'
                                   ,'Deformation',[ ['Ex','Ex fit'],['Ey','Ey fit'],['Exy','Exy fit']],Override)


            self.ST_GUIStatus.SetLabel('Strain gauge inserted')

            # Save in root
            self.Tree_results.DeleteChildren(self.StrainGates_root)

            self.StrainGates_X = self.Tree_results.AppendItem(self.StrainGates_root, '1')

            # Get the E
            YoungX = (F[-1])/Exx[-1]#_fit[-1]
            YoungY = (F[-1]) / Eyy[-1]#_fit[-1]
            Poisson = - np.mean(Exx/Eyy)
            Poisson2 = - np.mean(Eyy/Exx)

            # Exx Eyy Exy
            self.Eytree = self.Tree_results.AppendItem(self.StrainGates_X,'Ey: ' + str(Eyy[-1]))
            self.Eytree = self.Tree_results.AppendItem(self.StrainGates_X, 'Ex: ' + str(Exx[-1]))
            self.YoungX = self.Tree_results.AppendItem(self.StrainGates_X, 'Young Modulus[Using Ex]: '+str(np.round(YoungX)/(10**7)) +' GPa')
            self.YoungY = self.Tree_results.AppendItem(self.StrainGates_X, 'Young Modulus[Using Ey]: ' + str(np.round(YoungY)/(10**7)) +' GPa')
            self.Poisson = self.Tree_results.AppendItem(self.StrainGates_X, 'Poisson Coefficient[Ex/Ey]: ' + str(Poisson))
            self.Poisson = self.Tree_results.AppendItem(self.StrainGates_X,'Poisson Coefficient[Ey/Ex]: ' + str(Poisson2))

    ########### Line Callback ##########
    def PB_lineDeformation_Callback(self,e):

        # Get the position
        dlg = wx.TextEntryDialog(self, 'Select Position to get Horizontal Line:', 'Position ')
        dlg.SetValue("50")
        if dlg.ShowModal() == wx.ID_OK:
            Pos = int(dlg.GetValue())
        dlg.Destroy()

        # Deformations
        masked_array = np.ma.array(self.Exx[self.Actual_Image], mask=self.MaskY)
        Exx = masked_array[Pos, :]

        masked_array = np.ma.array(self.Eyy[self.Actual_Image], mask=self.MaskY)
        Eyy = masked_array[Pos, :]

        masked_array = np.ma.array(self.Exy[self.Actual_Image], mask=self.MaskY)
        Exy = masked_array[Pos, :]

        plt.subplot(3,1,1)
        plt.plot(Exx)
        plt.subplot(3, 1, 2)
        plt.plot(Eyy)
        plt.subplot(3, 1, 3)
        plt.plot(Exy)
        plt.show()

        np.savetxt('Exx.csv', np.ma.filled(Exx, np.nan), delimiter=',')
        np.savetxt('Eyy.csv', np.ma.filled(Eyy, np.nan), delimiter=',')
        np.savetxt('Exy.csv', np.ma.filled(Exy, np.nan), delimiter=',')
