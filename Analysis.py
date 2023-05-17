import wx #check
import os #check
import numpy as np #check
import pickle #check
import matplotlib.pyplot as plt #check

import wx.xrc #checar necessidade
import wx.grid #checar necessidade

class AnalysisMixin:
    def Menu_GridAnalysis_Callback(self,e):

        dlg = wx.TextEntryDialog(self, 'Enter Window Values', 'Values:')
        dlg.SetValue("[10,20,30,40,50]")
        if dlg.ShowModal() == wx.ID_OK:
            w = eval('np.array('+dlg.GetValue()+')')
            Comp_vec = w-np.max(w)
            print('One'+str(Comp_vec))
            Comp_vec = Comp_vec[::-1]
            print('Two' + str(Comp_vec))
        dlg.Destroy()

        dlg = wx.TextEntryDialog(self, 'Enter step Values', 'Values:')
        dlg.SetValue("[1,2,5,10,20,30,50]")
        if dlg.ShowModal() == wx.ID_OK:
            steps = eval('np.array(' + dlg.GetValue() + ')')
        dlg.Destroy()

        # Get the position
        dlg = wx.TextEntryDialog(self, 'Select Position to get Horizontal Line:', 'Position ')
        dlg.SetValue("50")
        if dlg.ShowModal() == wx.ID_OK:
            Pos = int(dlg.GetValue())
        dlg.Destroy()

        # Loop to calculate

        N_grids = len(steps)*len(w)

        for idxW in range(len(w)):

            for idxS in range(len(steps)):
                self.Toolbar_cplus_Callback(e,w[idxW],steps[idxS])
                self.PB_filterDisp_Callback(e)
                self.PB_filterStrain_Callback(e)

                # Deformations
                masked_array = np.ma.array(self.Exx[self.Actual_Image], mask=self.MaskY)
                Exx = masked_array[Pos+Comp_vec[idxW], :]

                masked_array = np.ma.array(self.Eyy[self.Actual_Image], mask=self.MaskY)
                Eyy = masked_array[Pos+Comp_vec[idxW], :]

                masked_array = np.ma.array(self.Exy[self.Actual_Image], mask=self.MaskY)
                Exy = masked_array[Pos+Comp_vec[idxW], :]

                plt.subplot(3, 1, 1)
                plt.plot(Exx)
                plt.subplot(3, 1, 2)
                plt.plot(Eyy)
                plt.subplot(3, 1, 3)
                plt.plot(Exy)
                plt.show()

                np.savetxt('Exx_'+str(w[idxW])+'_'+str(steps[idxS])+'.csv', np.ma.filled(Exx, np.nan), delimiter=',')
                np.savetxt('Eyy_'+str(w[idxW])+'_'+str(steps[idxS])+'.csv', np.ma.filled(Eyy, np.nan), delimiter=',')
                np.savetxt('Exy_'+str(w[idxW])+'_'+str(steps[idxS])+'.csv', np.ma.filled(Exy, np.nan), delimiter=',')

    def Menu_CompareResult_Callback(self,e):
        dlg = wx.FileDialog(self, "Choose the X data:", 'C:', "", "*.*", wx.FD_OPEN)

        if dlg.ShowModal() == wx.ID_OK:
            # Update Statusbar
            self.StatusBar.SetStatusText('Loading Project...')

            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            f = os.path.join(self.dirname, self.filename)
            # Load project and data
            with wx.FD_OPEN(f) as fa:
                X = pickle.load(fa)

        dlg = wx.FileDialog(self, "Choose the Y data:", 'C:', "", "*.*", wx.FD_OPEN)

        if dlg.ShowModal() == wx.ID_OK:
            # Update Statusbar
            self.StatusBar.SetStatusText('Loading Project...')

            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            f = os.path.join(self.dirname, self.filename)
            # Load project and data
            with wx.FD_OPEN(f) as fa:
                Y = pickle.load(fa)

        dlg = wx.FileDialog(self, "Choose the Eyy data:", 'C:', "", "*.*", wx.FD_OPEN)

        if dlg.ShowModal() == wx.ID_OK:
            # Update Statusbar
            self.StatusBar.SetStatusText('Loading Project...')

            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            f = os.path.join(self.dirname, self.filename)
            # Load project and data
            with wx.FD_OPEN(f) as fa:
                Eyy_Ansys = pickle.load(fa)

        if True:

            dlg = wx.TextEntryDialog(self, 'Enter Window Values', 'Values:')
            dlg.SetValue("[30]")
            if dlg.ShowModal() == wx.ID_OK:
                w = eval('np.array(' + dlg.GetValue() + ')')
                Comp_vec = w - np.max(w)
                print('One' + str(Comp_vec))
                Comp_vec = Comp_vec[::-1]
                print('Two' + str(Comp_vec))
            dlg.Destroy()

            dlg = wx.TextEntryDialog(self, 'Enter step Values', 'Values:')
            dlg.SetValue("[10]")
            if dlg.ShowModal() == wx.ID_OK:
                steps = eval('np.array(' + dlg.GetValue() + ')')
            dlg.Destroy()

            # Folder
            dlg = wx.DirDialog(None, "Choose directory to save results", "",
                               wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
            dlg.ShowModal()
            Path = dlg.GetPath()

            # Loop to calculate
            N_grids = len(steps) * len(w)
            X_ori = X.copy()
            Y_ori = Y.copy()

            for idxW in range(len(w)):

                for idxS in range(len(steps)):

                    self.Toolbar_cplus_Callback(e, w[idxW], steps[idxS])
                    self.PB_filterDisp_Callback(e)

                    Points_X = np.arange(self.Rxi, self.Rxf +1).astype(int)
                    Points_Y = np.arange(self.Ryi , self.Ryf +1).astype(int)

                    X = -X_ori[np.ix_(Points_Y, Points_X)]
                    Y = -Y_ori[np.ix_(Points_Y, Points_X)]
                    Eyy_Ansys = Eyy_Ansys[np.ix_(Points_Y, Points_X)]

                    Mask_pre = self.MaskY.copy()
                    Mask_pre[Mask_pre == 0] = np.nan

                    X = self.SmoothT(X*Mask_pre,0,3.)
                    Y = self.SmoothT(Y*Mask_pre, 0,3.)

                    ################ Plane Fit###########
                    Filter_size = 21
                    U = X*Mask_pre
                    Def_xxU, Def_xyU = self.sgolay2d(U, Filter_size, order=2,
                                                     derivative='both')  # self.Plane_fit(U,Filter_size)

                    Def_xx = Def_xxU   #/Def_xxV
                    Def_xy = Def_xyU   #/Def_xyV

                    # Y
                    U = Y* Mask_pre
                    Def_yxU, Def_yyU = self.sgolay2d(U, Filter_size, order=2,
                                                     derivative='both')  # self.Plane_fit(U,Filter_size)

                    Def_yx = Def_yxU   #/ Def_yxV
                    Def_yy = Def_yyU   #/ Def_yyV

                    # Calculate deformation
                    Exx_b = 1. / 2 * (2 * Def_xx + Def_xx ** 2 + Def_xy ** 2) * 10 ** 5
                    Exy_b = 1. / 2 * (Def_xy + Def_yx + Def_xx * Def_xy + Def_yx * Def_yy) * 10 ** 5
                    Eyy_b = 1. / 2 * (2 * Def_yy + Def_xy ** 2 + Def_yy ** 2) * 10 ** 5

                    Exx_b = np.ma.array(Exx_b, mask=self.MaskY)
                    Exy_b = np.ma.array(Exy_b, mask=self.MaskY)
                    Eyy_b = np.ma.array(Eyy_b, mask=self.MaskY)

                    Eyy_Ansys = np.ma.array(Eyy_Ansys, mask=self.MaskY)

                    Eyy_b = Eyy_Ansys

                    if idxW == 0 and idxS == 0:
                        # Save original deformations
                        plt.imshow(Exx_b)
                        plt.title('Exx Original ')
                        plt.colorbar()
                        plt.savefig(Path + '/Exx_original.png',
                                    bbox_inches='tight', dpi=600)

                        plt.clf()
                        plt.imshow(Eyy_b)
                        plt.title('Eyy Original ')
                        plt.colorbar()
                        plt.savefig(Path + '/Eyy_original.png',
                                    bbox_inches='tight', dpi=600)
                        plt.clf()
                        plt.imshow(Exy_b)
                        plt.title('Exy Original ')
                        plt.colorbar()
                        plt.savefig(Path + '/Exy_original.png',
                                    bbox_inches='tight', dpi=600)


                    #### Measurement
                    disX = np.ma.array(self.disX_smooth[self.Actual_Image], mask=self.MaskY)
                    disY = np.ma.array(self.disY_smooth[self.Actual_Image], mask=self.MaskY)
                    ################ Plane Fit###########
                    Filter_size = float(self.ET_filterSizeStrain.GetLineText(0))

                    U =  self.disX_smooth[self.Actual_Image].copy()
                    Def_xxU, Def_xyU = self.sgolay2d(U, Filter_size, order=2,
                                                     derivative='both')  # self.Plane_fit(U,Filter_size)

                    Def_xx = Def_xxU #/ Def_xxV
                    Def_xy = Def_xyU #/ Def_xyV

                    # Y
                    U = self.disY_smooth[self.Actual_Image].copy()
                    Def_yxU, Def_yyU = self.sgolay2d(U, Filter_size, order=2,
                                                     derivative='both')  # self.Plane_fit(U,Filter_size)

                    Def_yx = Def_yxU #/ Def_yxV
                    Def_yy = Def_yyU #/ Def_yyV

                    # Calculate deformation
                    Exx = 1. / 2 * (2 * Def_xx + Def_xx ** 2 + Def_xy ** 2) *self.DIC_calibration_factor
                    Exy = 1. / 2 * (Def_xy + Def_yx + Def_xx * Def_xy + Def_yx * Def_yy) *self.DIC_calibration_factor
                    Eyy = 1. / 2 * (2 * Def_yy + Def_xy ** 2 + Def_yy ** 2) *self.DIC_calibration_factor

                    Exx = np.ma.array(Exx, mask=self.MaskY)
                    Exy = np.ma.array(Exy, mask=self.MaskY)
                    Eyy = np.ma.array(Eyy, mask=self.MaskY)

                    # Use the own code deformation calculation
                    self.PB_filterStrain_Callback(e)
                    Exx = np.ma.array(self.Exx[0], mask=self.MaskY)
                    Exy = np.ma.array(self.Exy[0], mask=self.MaskY)
                    Eyy = np.ma.array(self.Eyy[0], mask=self.MaskY)

                    DiffX = Exx_b - Exx
                    DiffY = Eyy_b - Eyy
                    DiffXY = Exy_b - Exy
                    
                    # Variable needs attention
                    Displacement = False

                    if Displacement == True:


                        plt.imshow(DiffY)
                        plt.title('Difference[rms]: ' + str(
                            np.round(1000 * np.sqrt(np.mean(DiffY ** 2))) / 1000)+'. w:'+str(w[idxW])+'. s:'+str(steps[idxS]))
                        plt.colorbar()

                        plt.savefig(Path+'/Error_Y_Ansys_benchmark_w'+str(w[idxW])+'_s'+str(steps[idxS])+'.png', bbox_inches='tight',dpi=600)

                        plt.clf()
                        plt.imshow(DiffX)
                        plt.title(
                            'X Difference. RMS[Pixel Error]: ' + str(np.round(1000 * np.sqrt(np.mean(DiffX ** 2))) / 10000)
                            + '. w:' + str(w[idxW]) + '. s:' + str(steps[idxS]))
                        plt.colorbar()
                        plt.savefig(Path+'/Error_X_Ansys_benchmark_w'+str(w[idxW])+'_s'+str(steps[idxS])+'.png', bbox_inches='tight',dpi=600)
                        plt.clf()
                    else:

                        # Eyy
                        plt.subplot(1, 2, 1)
                        plt.imshow(DiffY)
                        plt.title('Difference[rms]: ' + str(
                            np.round(1000 * np.sqrt(np.mean(DiffY ** 2))) / 1000) + '. w:' + str(
                            w[idxW]) + '. s:' + str(steps[idxS]))
                        plt.colorbar()

                        plt.subplot(1, 2, 2)
                        plt.imshow(Eyy)
                        plt.title('Eyy Measured ')
                        plt.colorbar()

                        plt.savefig(
                            Path + '/Error_Eyy_Ansys_benchmark_w' + str(w[idxW]) + '_s' + str(steps[idxS]) + '.png',
                            bbox_inches='tight', dpi=600)

                        # Exx
                        plt.clf()
                        plt.subplot(1, 2, 1)
                        plt.imshow(DiffX)
                        plt.title(
                            'Difference[rms]: ' + str(
                                np.round(1000 * np.sqrt(np.mean(DiffX ** 2))) / 1000)
                            + '. w:' + str(w[idxW]) + '. s:' + str(steps[idxS]))
                        plt.colorbar()

                        plt.subplot(1, 2, 2)
                        plt.imshow(Exx)
                        plt.title('Exx Measured ')
                        plt.colorbar()

                        plt.savefig(
                            Path + '/Error_Exx_Ansys_benchmark_w' + str(w[idxW]) + '_s' + str(steps[idxS]) + '.png',
                            bbox_inches='tight', dpi=600)
                        plt.clf()

                        # Exy
                        plt.subplot(1, 2, 1)
                        plt.imshow(DiffXY)
                        plt.title('Exy Difference. RMS[me]: ' + str(
                            np.round(1000 * np.sqrt(np.mean(DiffXY ** 2))) / 1000) + '. w:' + str(
                            w[idxW]) + '. s:' + str(steps[idxS]))
                        plt.colorbar()


                        plt.subplot(1, 2, 2)
                        plt.imshow(Exy)
                        plt.title('Exy Measured ')
                        plt.colorbar()


                        plt.savefig(
                            Path + '/Error_Exy_Ansys_benchmark_w' + str(w[idxW]) + '_s' + str(steps[idxS]) + '.png',
                            bbox_inches='tight', dpi=600)
                        plt.clf()

                    np.savetxt('Exx_' + str(w[idxW]) + '_' + str(steps[idxS]) + '.csv', np.ma.filled(Exx, np.nan),
                               delimiter=',')
                    np.savetxt('Eyy_' + str(w[idxW]) + '_' + str(steps[idxS]) + '.csv', np.ma.filled(Eyy, np.nan),
                               delimiter=',')
                    np.savetxt('Exy_' + str(w[idxW]) + '_' + str(steps[idxS]) + '.csv', np.ma.filled(Exy, np.nan),
                               delimiter=',')

                    np.savetxt('DIff_Exx_' + str(w[idxW]) + '_' + str(steps[idxS]) + '.csv', np.ma.filled(DiffX, np.nan),
                               delimiter=',')
                    np.savetxt('DIff_Eyy_' + str(w[idxW]) + '_' + str(steps[idxS]) + '.csv', np.ma.filled(DiffY, np.nan),
                               delimiter=',')
                    np.savetxt('DIff_Exy_' + str(w[idxW]) + '_' + str(steps[idxS]) + '.csv', np.ma.filled(DiffXY, np.nan),
                               delimiter=',')

                    np.savetxt('U_' + str(w[idxW]) + '_' + str(steps[idxS]) + '.csv',
                               np.ma.filled(disX, np.nan),
                               delimiter=',')
                    np.savetxt('V_' + str(w[idxW]) + '_' + str(steps[idxS]) + '.csv',
                               np.ma.filled(disY, np.nan),
                               delimiter=',')
        else:

            ### Get the data which was calculated
            w = float(self.ET_sizeWindow.GetLineText(0))
            print("The used window was " + str(w))

            Points_X = np.arange(self.Rxi + w, self.Rxf - w).astype(int)
            Points_Y = np.arange(self.Ryi + w, self.Ryf - w ).astype(int)

            X_ori = X.copy()
            Y_ori = Y.copy()

            X =  -X[np.ix_(Points_Y,Points_X)]
            Y =  -Y[np.ix_(Points_Y, Points_X)]

            X = np.ma.array(X, mask=self.MaskY)
            Y = np.ma.array(Y, mask=self.MaskY)

            disX = np.ma.array(self.disX_smooth[self.Actual_Image], mask=self.MaskY)
            disY = np.ma.array(self.disY_smooth[self.Actual_Image], mask=self.MaskY)

            DiffX = (disX-np.mean(disX)) - (X-np.mean(X))
            DiffY = (disY - np.mean(disY) ) - (Y-np.mean(Y))

            # Difference
            plt.figure(0)
            plt.subplot(1,2,1)
            plt.imshow(DiffY)
            plt.title('Y Difference. RMS[Pixel Error]: '+str(   np.round(10000*np.sqrt(np.mean(DiffY**2)))/10000    ))
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow(DiffX)
            plt.title('X Difference. RMS[Pixel Error]: '+str(  np.round(1000* np.sqrt(np.mean(DiffX**2)) )/10000     ))
            plt.colorbar()
            plt.show()

            # Y comparison
            plt.figure(1)
            plt.subplot(1, 2, 1)
            plt.imshow(Y-np.mean(Y))
            plt.title('Y real')
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow((disY-np.mean(disY)))
            plt.title('Y measured')
            plt.colorbar()
            plt.show()

            # X Comparison
            plt.figure(2)
            plt.subplot(1, 2, 1)
            plt.imshow(X - np.mean(X))
            plt.title('X real')
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow((disX-np.mean(disX)))
            plt.title('X measured')
            plt.colorbar()
            plt.show()