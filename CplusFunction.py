import numpy as np #check
import time #check
import GN_opt
import scipy #check
from scipy import interpolate #check
import matplotlib.pyplot as plt #check

class CplusMixin:
    def Toolbar_cplus_Callback(self,e,w=None,step=None):
        # Parameter Inputs
        if self.CB_stepAutomatic.IsChecked():
            Auto_step = 1
        else:
            Auto_step = 0

        if w == None:
            w = float(self.ET_sizeWindow.GetLineText(0))
            step = np.array(eval(self.ET_step.GetLineText(0)))
            step = step.astype(float)
        else:
            step = np.array([step, step]).astype(float)

        Algo = self.CB_algChoice.GetSelection()
        All_images = self.CB_allImages.IsChecked()
        Treshold = float(self.ET_tresholdMain.GetLineText(0))
        New_template_corr = 1
        N_parallel = int(self.ET_parallelCores.GetLineText(0) )

        self.StatusBar.SetStatusText('Calculating images coefficients...')

        # Create Gaussian Center Window
        if self.CB_gaussCentered.IsChecked() == True:
            GaussianWindow = 1#gaussianWindow(2 * w + 1, int(w), sym=True)
            #GaussianWindow = npl.repmat(GaussianWindow, len(GaussianWindow), 1)
            #GaussianWindow = GaussianWindow * np.transpose(GaussianWindow)
            print('Window is centered')
        else:
            GaussianWindow = 0

        # Get interpolations if val != None
        Padding = 6

        Mesh_allX, Mesh_allY = np.meshgrid(range(0, int(self.I_ref.shape[1])),
                                           range(0, int(self.I_ref.shape[0])))  # Mesh_allY = np.transpose(Mesh_allY)
        import pdb
        print(Mesh_allX)
        #pdb.set_trace()
        Points = np.transpose(np.vstack((Mesh_allY.flatten('C'), Mesh_allX.flatten('C'))))


        # Trying to change the points to include the extreme values
        #Points_X = np.arange(self.Rxi + w, self.Rxf - w + step[0], step[0])
        #Points_Y = np.arange(self.Ryi + w, self.Ryf - w + step[1], step[1])

        Points_X = np.arange(self.Rxi , self.Rxf  + step[0], step[0])
        Points_Y = np.arange(self.Ryi , self.Ryf  + step[1], step[1])

        Length = 1
        if All_images == True:
            Length = self.Len_images
            All_images = 1
        else:
            All_images = 0

        # Inputs
        Input = [w,step[0],step[1],self.Rxi,self.Rxf,self.Ryi,self.Ryf,self.Actual_Image,All_images,N_parallel,GaussianWindow,Algo,Treshold]
        np.savetxt('Inputs.txt', Input, delimiter=' ')
        # Save Reference Image for the algorithm
        np.savetxt('I.txt',self.I_ref,delimiter=' ')
        #np.savetxt('Mesh_allX.txt', Mesh_allX, delimiter=' ')
        #np.savetxt('Mesh_allY.txt', Mesh_allY, delimiter=' ')
        np.savetxt('Points.txt',Points,delimiter=' ')
        np.savetxt('Mask.txt',self.mask,delimiter=' ')
        import os
        for idxImages in range(Length):

            if All_images == 1: # IF not true, just use the actual selected image
                self.Actual_Image = idxImages

            ## Save Inputs
            np.savetxt('I_f.txt',self.I_all[self.Actual_Image])

            ## End of input save

            start_time = time.time()

            Processors = int(self.ET_parallelCores.GetLineText(0))
            print('Is running')
            cmd1 = r'"set OMP_NUM_THREADS = 4"'
            os.system(cmd1)
            cmd = r'"DIC_Fast_implementation.exe"'#r'"D:/Projetos/Python GUIs/GUI_s/DIC GUI/DIC C++ #Search/DIC_Fast_implementation/bin/Release/DIC_Fast_implementation.exe"'

            #dirname = os.path.dirname(__file__)
            #cmd2  = '"'+dirname+ '/DIC C++ Search/DIC_Fast_implementation/bin/Release/DIC_Fast_implementation.exe'+'"'
            #cmd2 = '%r' %cmd2[1:-1]

            try:
                os.system(cmd)
            except:
                cmd = r'"DIC_Fast_implementation.exe"'
                os.system(cmd)

            elapsed_time = time.time() - start_time
            print('Real time taked was:')
            print(elapsed_time)

            points = np.genfromtxt('Points.csv',delimiter=',')
            disX = np.genfromtxt('disX.csv',delimiter=',')
            disY = np.genfromtxt('disY.csv',delimiter=',')
            All_parameters = np.genfromtxt('All_parameters.csv',delimiter=',')

            # Get X values
            MeshLocX, MeshLocY = np.meshgrid(Points_X, Points_Y)
            self.MeshLocX = np.transpose(MeshLocX)
            self.MeshLocY = np.transpose(MeshLocY)

            # self.disX[self.Actual_Image] = disX
            # self.disY[self.Actual_Image] = disY
            # self.disS[self.Actual_Image] = np.sqrt(disX**2+disY**2)

            # Interpolate
            #self.X, self.Y = np.meshgrid([range(int(self.Ryi + w), int(self.Ryf - w))][0],
                                        # [range(int(self.Rxi + w), int(self.Rxf - w))][0])
            self.X, self.Y = np.meshgrid([range(int(self.Ryi ), int(self.Ryf+1 ))][0],
                                         [range(int(self.Rxi ), int(self.Rxf+1 ))][0])
            self.X = np.transpose(self.X)
            self.Y = np.transpose(self.Y)

            # points = np.transpose(np.vstack((MeshLocX.flatten('F'),MeshLocY.flatten('F'))))

            BBB, IndexX = GN_opt.find(np.isnan(disX), False)
            BBB, IndexY = GN_opt.find(np.isnan(disY), False)
            disX = disX[~np.isnan(disX)]  # eliminate any NaN
            disY = disY[~np.isnan(disY)]#np.squeeze(disY[~np.isnan(disY)])

            BBB, IndexA = GN_opt.find(np.isnan(All_parameters[:, 0]), False)
            All_parameters = All_parameters[~np.isnan(All_parameters[:, 0]),:]
            #self.
            self.IsGaussianProcess = 1

            pointsX = points[IndexX, :]
            pointsY = points[IndexY, :]
            pointsA = points[IndexA, :]


            ######## If dynamic window, loop the program until it converges #######
            if self.CB_stepAutomatic.IsChecked():
                Treshold = 0.05

                Converged = False

                while Converged == False:

                    Vertices = np.hstack((pointsX, pointsY))
                    Disp_vector = np.hstack((disX, disY, All_parameters))
                    Triangles = scipy.delaunay(Vertices)
                    Points_new = []
                    t = 0

                    for triangle in Triangles:
                        Dis_max = np.max( [np.linalg.norm(Disp_vector[triangle[0],:]-Disp_vector[triangle[1],:]),
                                           np.linalg.norm(Disp_vector[triangle[0], :] - Disp_vector[triangle[2], :]),
                                           np.linalg.norm(Disp_vector[triangle[1], :] - Disp_vector[triangle[2], :])])
                        if Dis_max > Treshold:
                            Points_new.append([ np.mean([pointsX[triangle[0]]+pointsX[triangle[1]]+pointsX[triangle[2]] ]),
                                                np.mean([pointsY[triangle[0]] + pointsY[triangle[1]] + pointsY[triangle[2]]])
                                                ])
                        t += 1

                    if not Points_new: # Check if over convergence
                        Converged = True
                    else:
                        Input = [w, step[0], step[1], self.Rxi, self.Rxf, self.Ryi, self.Ryf, self.Actual_Image,
                                 All_images, N_parallel, GaussianWindow]
                        np.savetxt('Inputs.txt', Input, delimiter=' ')
                        np.savetxt('Points.txt', Points, delimiter=' ')

                        start_time = time.time()

                        print('Is Increasing Convergence')
                        cmd = r'"D:/Projetos/Python GUIs/GUI_s/DIC GUI/DIC C++ Search/DIC_Fast_implementation/bin/Release/DIC_Fast_implementation.exe"'
                        os.system(cmd)

                        elapsed_time = time.time() - start_time
                        print('Real time taked was:')
                        print(elapsed_time)

                        points = np.genfromtxt('Points.csv', delimiter=',')
                        disX = np.genfromtxt('disX.csv', delimiter=',')
                        disY = np.genfromtxt('disY.csv', delimiter=',')
                        All_parameters = np.genfromtxt('All_parameters.csv', delimiter=',')

                        All_parameters = All_parameters[~np.isnan(All_parameters), :]
                        disX = disX[~np.isnan(disX)]  # eliminate any NaN
                        disY = disY[~np.isnan(disY)]  # np.squeeze(disY[~np.isnan(disY)])


                        BBB, IndexX = GN_opt.find(np.isnan(disX), False)
                        BBB, IndexY = GN_opt.find(np.isnan(disY), False)
                        BBB, IndexA = GN_opt.find(np.isnan(All_parameters[:,0]), False)

                        pointsX = points[IndexX, :]
                        pointsY = points[IndexY, :]
                        pointsA = points[IndexA, :]
                        ########################## END mesh Convergence Loop #########
                # From here C


            # Grid Data
            self.All_parameters = []
            if self.CB_stepAutomatic.IsChecked():
                a = 2 # Scatter interpolation to be implemented

            else:  # Cubic interpolation
                self.disX[self.Actual_Image] = interpolate.griddata(pointsX, disX, (self.X, self.Y),
                                                                    method='cubic')  # linear / cubic
                self.disY[self.Actual_Image] = interpolate.griddata(pointsY, disY, (self.X, self.Y), method='cubic')  # linear

                # Interpolate All other parameters
                Mask_pre = self.mask[np.ix_([range(int(self.Ryi), int(self.Ryf + 1))][0], [range(int(self.Rxi), int(self.Rxf + 1))][0])]
                Mask_pre[Mask_pre == 0] = np.nan
                
                self.All_parameters.append(interpolate.griddata(pointsA, All_parameters[:,0], (self.X, self.Y),method='cubic')* Mask_pre ) # U
                self.All_parameters.append(interpolate.griddata(pointsA, All_parameters[:, 1], (self.X, self.Y), method='cubic')* Mask_pre)# V
                self.All_parameters.append(interpolate.griddata(pointsA, All_parameters[:, 2], (self.X, self.Y), method='cubic')* Mask_pre)  # dU/dx
                self.All_parameters.append(interpolate.griddata(pointsA, All_parameters[:, 3], (self.X, self.Y), method='cubic')* Mask_pre)  # dV/dx
                self.All_parameters.append(interpolate.griddata(pointsA, All_parameters[:, 4], (self.X, self.Y), method='cubic')* Mask_pre) # dU/dy
                self.All_parameters.append(interpolate.griddata(pointsA, All_parameters[:, 5], (self.X, self.Y), method='cubic')* Mask_pre)  # dV/dy

                np.savetxt('sigma_U_.csv', np.ma.filled(self.All_parameters[0], np.nan),delimiter=',') #' + str(w) + '_' + str(step) + '
                np.savetxt('sigma_V_.csv', np.ma.filled(self.All_parameters[1], np.nan),delimiter=',')
                np.savetxt('sigma_Udx_.csv', np.ma.filled(self.All_parameters[2], np.nan),delimiter=',')
                np.savetxt('sigma_Vdx_.csv', np.ma.filled(self.All_parameters[3], np.nan),delimiter=',')
                np.savetxt('sigma_Udy_.csv', np.ma.filled(self.All_parameters[4], np.nan),delimiter=',')
                np.savetxt('sigma_Vdy_.csv', np.ma.filled(self.All_parameters[5], np.nan),delimiter=',')


        #Mask_pre = self.mask[np.ix_([range(int(self.Ryi + w), int(self.Ryf - w))][0], [range(int(self.Rxi + w), int(self.Rxf - w))][0])]
        Mask_pre = self.mask[np.ix_([range(int(self.Ryi), int(self.Ryf+1 ))][0], [range(int(self.Rxi ), int(self.Rxf+1 ))][0])]
        Mask_pre[Mask_pre == 0] = np.nan

        self.disY[self.Actual_Image] = self.disY[self.Actual_Image] * Mask_pre
        self.disX[self.Actual_Image] = self.disX[self.Actual_Image] * Mask_pre

        self.MaskY = np.isnan(self.disY[self.Actual_Image] * Mask_pre)  # np.isnan(self.disY[self.Actual_Image])#
        self.MaskX = np.isnan(self.disX[self.Actual_Image] * Mask_pre)  # np.isnan(self.disX[self.Actual_Image])##

        if step[0] == -999:
            masked_array = np.ma.array(self.All_stdX, mask=self.MaskY)

            fig, axes = plt.subplots(nrows=1, ncols=2)
            im1 = axes[0].imshow(masked_array)
            masked_array = np.ma.array(self.All_stdY, mask=self.MaskY)

            im2 = axes[1].imshow(masked_array)

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im1, cax=cbar_ax)

            plt.show()


        masked_array = np.ma.array(self.disY[self.Actual_Image], mask=self.MaskY)

        # Plot Y in the Disp field
        self.plot1Disp.draw(self.I_all[self.Actual_Image], self.Y + self.disX[self.Actual_Image],
                            self.X + self.disY[self.Actual_Image], masked_array, 'V Displacement', 'X', 'Y')

        self.plot1Data = masked_array
        self.plot1Disp.canvas.mpl_connect('motion_notify_event', self.VisuDisp1)
        # dlg = wx.MessageDialog(self, 'Calculation Completed.', 'Done',
        #                       wx.OK | wx.ICON_INFORMATION)
        # dlg.ShowModal()
        # dlg.Destroy()

        self.StatusBar.SetStatusText('Displacement Calculated!')