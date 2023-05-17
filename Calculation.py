import wx #check
import cv2 #check
import numpy as np #check
from scipy.signal import gaussian as gaussianWindow #check
import numpy.matlib as npl #check
import GN_opt
import Interp_spline as it
from scipy import interpolate #check


import wx.xrc #checar necessidade
import wx.grid #checar necessidade

class CalculationMixin:
    def Toolbar_calculate_Callback(self,e,w=None,step=None):

        # Parameter Inputs
        if w == None:
            w = float(self.ET_sizeWindow.GetLineText(0))
            step = np.array(eval(self.ET_step.GetLineText(0)))
            step = step.astype(float)
        else:
            step = np.array([step,step]).astype(float)

        Algo = self.CB_algChoice.GetSelection()
        All_images = self.CB_allImages.IsChecked()
        Treshold = float(self.ET_tresholdMain.GetLineText(0))
        New_template_corr = 1
        self.StatusBar.SetStatusText('Calculating images coefficients...')

        # Create Gaussian Center Window
        if self.CB_gaussCentered.IsChecked() == True:
            GaussianWindow = gaussianWindow(2 * w + 1, int(w ), sym=True)
            GaussianWindow = npl.repmat(GaussianWindow, len(GaussianWindow), 1)
            GaussianWindow = GaussianWindow * np.transpose(GaussianWindow)
            print('Window is centered')
        else:
            GaussianWindow = 1

        # Get interpolations if val != None
        Padding = 6

        Mesh_allX, Mesh_allY = np.meshgrid(range(0, int(self.I_ref.shape[1])),range(0, int(self.I_ref.shape[0]) ))  # Mesh_allY = np.transpose(Mesh_allY)
        Points = np.transpose(np.vstack((Mesh_allX.flatten('F'), Mesh_allY.flatten('F'))))

        # REFERENCE INTERPOLATION COEFFICIENTS
        if self.I_ref_interp[0] is None:
            self.InterpI = GN_opt.getInterp(self.I_ref, Padding, 1)
            Values = it.Interp_spline(Points, self.InterpI['coeff'])
            self.I_ref_interp = np.reshape(Values, (self.I_ref.shape[0], self.I_ref.shape[1]), order='F')

        # Positions Vector
        Points_X = np.arange(self.Rxi + w, self.Rxf - w + step[0], step[0])
        Points_Y = np.arange(self.Ryi + w, self.Ryf - w + step[1], step[1])

        # Initialize Displacement, Correlation and position

        disX = np.full((len(Points_Y) * len(Points_X), 1), np.nan)  # Reverse
        disY = np.full((len(Points_Y) * len(Points_X), 1), np.nan)

        Xpos = np.full((len(Points_Y), len(Points_X)), np.nan)
        Ypos = np.full((len(Points_Y), len(Points_X)), np.nan)

        points = np.zeros((len(Points_X) * len(Points_Y), 2))

        C_all = np.full((len(Points_Y), len(Points_X)), np.nan)

        P_temp = []
        for idxXX in range(len(Points_Y)):
            P_temp.append([])
            for idxYY in range(len(Points_X)):
                P_temp[idxXX].append([])

        # MAIN LOOP
        Length = 1
        if All_images == True:
            Length = self.Len_images


        # Initialize Waitbar
        progressMax = len(Points_X)*len(Points_Y)
        dialog = wx.ProgressDialog("Calculating...", "Time remaining", progressMax*Length,
                                   style= wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME | wx.PD_AUTO_HIDE) #wx.PD_CAN_ABORT |

        for idxImages in range(Length):

            # Initialize Displacement, Correlation and position

            disX = np.full((len(Points_Y) * len(Points_X), 1), np.nan)  # Reverse
            disY = np.full((len(Points_Y) * len(Points_X), 1), np.nan)

            Xpos = np.full((len(Points_Y), len(Points_X)), np.nan)
            Ypos = np.full((len(Points_Y), len(Points_X)), np.nan)

            points = np.zeros((len(Points_X) * len(Points_Y), 2))

            C_all = np.full((len(Points_Y), len(Points_X)), np.nan)


            if All_images == True: # IF not true, just use the actual selected image
                self.Actual_Image = idxImages

            countWB = 0 # Reinitialize every time. For indexing

            if self.InterpS[self.Actual_Image] is None:
                self.InterpS[self.Actual_Image] = GN_opt.getInterp(self.I_all[self.Actual_Image], Padding, 1)
                Values = it.Interp_spline(Points, self.InterpS[self.Actual_Image]['coeff'])  # USE INTERP S
                self.I_f_interp[self.Actual_Image] = np.reshape(Values, (self.I_ref.shape[0], self.I_ref.shape[1]), order='F')


            self.StatusBar.SetStatusText('Calculating displacements.... Image '+str(idxImages+1)+'/'+str(self.Len_images))

            # DIC Loop
            for idxX in range(len(Points_Y)):
                for idxY in range(len(Points_X)):

                    # Get position from index loop
                    i = int(Points_Y[idxX])
                    j = int(Points_X[idxY])

                    Xpos[idxX,idxY] = i
                    Ypos[idxX, idxY] = j

                    # If mask, dont run
                    if self.mask[i, j] == 0:

                        C_all[idxX, idxY] = np.nan
                        points[countWB, 0] = i
                        points[countWB, 1] = j
                        disX[countWB] = np.nan
                        disY[countWB] = np.nan

                        countWB += 1
                        dialog.Update(countWB + progressMax*idxImages)
                        continue

                    # Get template self.I_ref
                    Template = self.I_ref_interp[np.ix_([ range(int(i - w),int(i + w+1))][0], [range(int(j - w),int(j + w+1))][0])]
                    Template_mask = self.mask[np.ix_([ range(int(i - w),int(i + w+1))][0], [range(int(j - w),int(j + w+1))][0])]

                    # Apply template Matching
                    if (idxX == 0 and idxY == 0 ) or New_template_corr == 1:#and idxImages == 0
                        dX,dY,max_val = self.TemplateMatch( self.I_f_interp[self.Actual_Image].astype('float32'),Template.astype('float32'), cv2.TM_CCOEFF_NORMED,[j,i])#'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED'
                                                                #self.I_all[0]
                        dX2,dY2,max_val2 = self.TemplateMatch( self.I_all[self.Actual_Image].astype('float32'),Template.astype('float32'), cv2.TM_CCOEFF_NORMED,[j,i])#['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

                        Original_I = True
                        if abs(dX-dX2) > 10 or abs(dX-dX2) > 10:
                            Original_I = True
                            dX = dX2
                            dY = dY2
                            max_val = max_val2

                        P = np.zeros((12, 1))

                        if abs(dX) < 150 and abs(dY) < 150 and max_val > 0.7:
                            P[0] = dX
                            P[1] = dY

                        New_template_corr = 0
                        print('New template Corr')

                    # If use the original image to calculate or the interpolated one
                    if Original_I == True:
                        P, C = GN_opt.GN_opt(self.I_f_interp[self.Actual_Image], self.I_ref_interp, Template, P,
                                                    np.array([j , i , self.Ryi, self.Rxi]),
                                                     self.InterpS[self.Actual_Image],
                                                     self.InterpI, Algo,GaussianWindow,Template_mask)
                    else:
                        P,C = GN_opt.GN_opt(self.I_f_interp[self.Actual_Image], self.I_ref_interp, Template, P, np.array( [j,i,self.Ryi,self.Rxi]), self.InterpS[self.Actual_Image],
                                        self.InterpI, Algo,GaussianWindow,Template_mask)

                    if C < 0.9 :
                        P = np.zeros((12, 1))
                        P[0] = dX
                        P[1] = dY
                        print('Stop 1')
                        if Original_I == False:
                            dX, dY, max_val = self.TemplateMatch(self.I_all[self.Actual_Image].astype('float32'),
                                                                 Template.astype('float32'), cv2.TM_CCOEFF_NORMED, [j,i])  # 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED'

                            P, C = GN_opt.GN_opt(self.I_f_interp[self.Actual_Image], self.I_ref_interp, Template, P,
                                                 [j, i, self.Ryi, self.Rxi],
                                                 self.InterpS[self.Actual_Image], self.InterpI, Algo,GaussianWindow,Template_mask)

                        else:
                            dX, dY, max_val = self.TemplateMatch(self.I_all[self.Actual_Image].astype('float32'),
                                                                   Template.astype('float32'), cv2.TM_CCOEFF_NORMED, [j, i])#'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED'

                            P, C = GN_opt.GN_opt(self.I_f_interp[self.Actual_Image], self.I_ref_interp, Template, P, [j, i, self.Ryi, self.Rxi],
                                             self.InterpS[self.Actual_Image],self.InterpI, Algo,GaussianWindow,Template_mask)

                    # Save displacements and plot correlation
                    if C < Treshold: # If correlation below threshold, save as nan
                        P[:] = np.nan
                        New_template_corr = 1

                    P_temp[idxX][idxY] = P
                    points[countWB,0] = i
                    points[countWB , 1] = j
                    disX[countWB] = P[0]
                    disY[countWB] = P[1]
                    C_all[idxX,idxY] = C
                    print('Correlation: ' + str(C))

                    # Update waitbar
                    countWB += 1
                    dialog.Update(countWB + progressMax*idxImages)

            # Close WB
            dialog.Destroy()
            countWB = 0
            
            # Get X values
            MeshLocX,MeshLocY = np.meshgrid(Points_X,Points_Y)
            self.MeshLocX = np.transpose(MeshLocX)
            self.MeshLocY = np.transpose(MeshLocY)

            # Interpolate
            self.X,self.Y = np.meshgrid([range(int(self.Ryi+w),int(self.Ryf-w))][0],[range(int(self.Rxi+w),int(self.Rxf-w))][0])
            self.X = np.transpose(self.X)
            self.Y = np.transpose(self.Y)


            BBB,IndexX = GN_opt.find(np.isnan(disX), False)
            BBB, IndexY = GN_opt.find(np.isnan(disY), False)
            disX = disX[~np.isnan(disX)]  # eliminate any NaN
            disY = np.squeeze(disY[~np.isnan(disY)])

            pointsX = points[IndexX,:]
            pointsY = points[IndexY, :]

            # Grid Data
            self.disX[self.Actual_Image] =  interpolate.griddata( pointsX, disX, (self.X, self.Y), method='linear') # linear / cubic
            self.disY[self.Actual_Image] =  interpolate.griddata(pointsY, disY, (self.X, self.Y), method='linear') # linear

            # Clean memory

###################################### FINISH IMAGES LOOP
        # Get Mask
        Mask_pre = self.mask[np.ix_([range(int(self.Ryi+w),int(self.Ryf-w))][0],[range(int(self.Rxi+w),int(self.Rxf-w))][0])]
        Mask_pre[Mask_pre==0] = np.nan
        self.disY[self.Actual_Image] = self.disY[self.Actual_Image]  * Mask_pre
        self.disX[self.Actual_Image] = self.disX[self.Actual_Image] * Mask_pre

        self.MaskY = np.isnan(self.disY[self.Actual_Image] * Mask_pre)#np.isnan(self.disY[self.Actual_Image])#
        self.MaskX = np.isnan(self.disX[self.Actual_Image] * Mask_pre)#np.isnan(self.disX[self.Actual_Image])##

        masked_array = np.ma.array(self.disY[self.Actual_Image], mask= self.MaskY)


        # Plot Y in the Disp field
        self.plot1Disp.draw(self.I_all[self.Actual_Image],self.Y+ self.disX[self.Actual_Image], self.X+ self.disY[self.Actual_Image],masked_array,'V Displacement','X','Y')
        self.plot1Data = masked_array
        self.plot1Disp.canvas.mpl_connect('motion_notify_event', self.VisuDisp1)
        self.StatusBar.SetStatusText('Displacement Calculated!')

    def TemplateMatch(self,I_f, Template, Method, Center):

        res = cv2.matchTemplate(I_f, Template,
                                Method)  # ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        disX = max_loc[0] + len(Template) / 2 - Center[0]
        disY = max_loc[1] + len(Template) / 2 - Center[1]
        C = max_val

        return disX, disY, C