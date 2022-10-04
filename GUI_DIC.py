import matplotlib
matplotlib.use('WXAgg')
import math
#from astropy.convolution import convolve_fft
# importing wx files
import wx
import wx.xrc
import wx.grid
# import the created GUI file
import gui
import time
# importing external libraries
import numpy as np
import numpy.matlib as npl

import scipy
from scipy import interpolate
#import pdb
import cv2
import os
import scipy.ndimage as ndimage
import pickle
import scipy.stats as st
from sklearn.linear_model import LinearRegression as LN
from sklearn.gaussian_process import GaussianProcessRegressor as GP
# Matplotlib
#matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import MatplotlibWxClasses as pltWx
from scipy.signal import gaussian as gaussianWindow
import scipy.linalg as linalg
#from sklearn.externals import joblib
#import joblib
import pdb
# My libraries
import GN_opt
import Interp_spline as it

class CalcFrame(gui.MainFrame):


############################# C++ Call Field ####################################
#################################################################################
#################################################################################
#################################################################################
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

#############################################################################
#################################################################################
#################################################################################
#################################################################################

    # constructor
    def __init__(self, parent):

        gui.MainFrame.__init__(self, parent)

        self.frame = gui.MainFrame

        # Icon
        #icon = wx.EmptyIcon()
        #icon.CopyFromBitmap(wx.Bitmap("D:\\Projetos\\Arasy Solutions\\Logo\\Arasy_36x36x.png", wx.BITMAP_TYPE_ANY))
        #self.SetIcon(icon)

        self.DIC_calibration_factor = 10**6
        self.DIC_calibration_factor_ML = self.DIC_calibration_factor/(10**5)
        ######################### MAIN TREE CHILDS #################
        self.PhotosMain_root = self.Tree_photosMain.AddRoot('Images')
        self.Results_root = self.Tree_results.AddRoot('Results')
        self.StrainGates_root = self.Tree_results.AppendItem(self.Results_root, 'Strain Gates')
        self.Line_deformation_root = self.Tree_results.AppendItem(self.Results_root, 'Line Deformation')
        self.Point_deformation_root = self.Tree_results.AppendItem(self.Results_root, 'Single Points')
        ###################### MATPLOTLIB CANVAS ####################
        self.plot1Image = pltWx.CanvasPanel2D_Image(self.Panel_plot1)
        self.plot2Image = pltWx.CanvasPanel2D_Image(self.Panel_plot2)
        self.plot1ROI = pltWx.CanvasPanel2D_Image(self.Panel_plot1ROI)#pltWx.CanvasPanel2D_Image(self.Panel_plot1ROI)
        self.plot2ROI = pltWx.CanvasPanel2D_Image(self.Panel_plot2ROI)
        self.plot1Disp = pltWx.CanvasPanel3D_Image(self.Panel_plot1Disp)
        self.plot1Strain = pltWx.CanvasPanel3D_Image(self.Panel_plot1Strain)
        self.plot1Res = pltWx.CanvasPanel3D_Image(self.Panel_plot1Res)
        self.plot2Res = pltWx.CanvasPanel2D(self.Panel_plot2Res)

        self.IsGaussianProcess = 0

    ############################    MENU FUNCTIONS ###################

    def MenuFolder_Callback(self,e):

        Treshold_size = int(self.ET_treshImage.GetLineText(0) )#1500
        Ratio = float(self.ET_treshRate.GetLineText(0) )
        print(Ratio.type)

        self.StatusBar.SetStatusText('Loading Images...')
        # Load Reference Image
        dlg = wx.FileDialog(self, "Choose The First Image", "D:", "", "*.*", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()

            self.I_ref = cv2.imread(os.path.join(self.dirname, self.filename), 0)  # Read grayscale

            ####### Bilateral Filter
            diag = np.sqrt(self.I_ref.shape[0] ** 2 + self.I_ref.shape[1] ** 2)
            sigmaSpace = 0.08 * diag
            sigmaColor = 75
            #self.I_ref = cv2.bilateralFilter(self.I_ref, -1, sigmaColor, sigmaSpace)
            #self.I_ref = cv2.GaussianBlur(self.I_ref, (10, 10), 0)
            ####### Bilateral Filter

            if np.max(self.I_ref.shape) > Treshold_size:
                #from scipy.misc import imresize
                #self.I_ref = imresize(self.I_ref, Ratio, interp='bilinear', mode=None)
                #self.I_ref = cv2.resize(src=self.I_ref, dsize=Ratio, interpolation=cv2.INTER_CUBIC)
                from PIL import Image
                self.I_ref = np.array(Image.fromarray(obj=self.I_ref, mode=None).resize(size=Ratio, resample=Image.BICUBIC))
                

                # Plot Image
            self.plot1Image.draw(self.I_ref, 'Reference Image', 'X', 'Y')
            self.plot1ROI.draw(self.I_ref, 'Select ROI', 'X', 'Y')

            self.mask = np.zeros((self.I_ref.shape[0], self.I_ref.shape[1]))
            self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y')

        dlg.Destroy()  # Close file dialog

        # Plot Image
        self.plot1Image.draw(self.I_ref, 'Reference Image', 'X', 'Y')
        self.plot1ROI.draw(self.I_ref, 'Select ROI', 'X', 'Y')

        self.mask = np.zeros((self.I_ref.shape[0], self.I_ref.shape[1]))
        self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y')

        # Load all other images
        dlg = wx.DirDialog(None, "Choose input directory", "",
                           wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        dlg.ShowModal()

        Files = os.listdir(dlg.GetPath())
        print(Files)
        self.I_all = []
        for image in Files:
            temp = cv2.imread(os.path.join(dlg.GetPath(), image), 0)

            ####### Bilateral Filter
            diag = np.sqrt(temp.shape[0] ** 2 + temp.shape[1] ** 2)
            sigmaSpace = 0.08 * diag
            sigmaColor = 75
            #temp = cv2.bilateralFilter(temp, -1, sigmaColor, sigmaSpace)
            #temp = cv2.GaussianBlur(temp, (10, 10), 0)
            ####### Bilateral Filter

            if np.max(temp.shape) > Treshold_size:
                temp = imresize(temp, Ratio, interp='bilinear', mode=None)
            self.I_all.append(temp)

        self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y')

        dlg.Destroy()

        self.plot2Image.draw(self.I_all[0], 'Image 1', 'X', 'Y')

        # Update tree
        self.Tree_photosMain.DeleteChildren(self.PhotosMain_root)
        self.PhotoRef = self.Tree_photosMain.AppendItem(self.PhotosMain_root,
                                                     'Reference Image')

        self.PhotosTree = []
        for idx in range(len(self.I_all)):
            self.PhotosTree.append( self.Tree_photosMain.AppendItem(self.PhotosMain_root,
                                                     str(idx+1)) )
        # Update Statusbar
        self.StatusBar.SetStatusText(
            "Images Loaded! Dimensions: " + str(self.I_ref.shape[0]) + " x " + str(
                self.I_ref.shape[1]))

        dlg = wx.MessageDialog(self, 'Images Loaded!', 'Loaded',  # Second argument is title
                               wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

        self.StatusBar.SetStatusText(
            "Images Loaded! Dimensions: " + str(self.I_ref.shape[0]) + " x " + str(
                self.I_ref.shape[1]))

        self.Actual_Image = 0

        # Create Empty variables
        Len_images = len(self.I_all)
        self.Exx = [None] * Len_images
        self.Eyy = [None] * Len_images
        self.Exy = [None] * Len_images
        self.disX= [None] * Len_images
        self.disY= [None] * Len_images
        self.disX_smooth = [None] * Len_images
        self.disY_smooth = [None] * Len_images
        self.InterpI = [None]
        self.I_ref_interp = [None]
        self.InterpS = [None] * Len_images
        self.I_f_interp = [None] * Len_images
        self.Len_images = Len_images

        self.Exx_fit = [None] * self.Len_images
        self.Eyy_fit = [None] * self.Len_images
        self.Exy_fit = [None] * self.Len_images

    def Menu_TwoImages_Callback(self, e):
        self.StatusBar.SetStatusText('Loading Images...')

        Treshold_size = int(self.ET_treshImage.GetLineText(0) )#1500
        Ratio = float(self.ET_treshRate.GetLineText(0) )
        print("-=-=-=-=-=-=-=-=-=-")
        print(Ratio)
        print("-=-=-=-=-=-=-=-=-=-")

        # Load Reference Image
        dlg = wx.FileDialog(self, "Choose The First Image", "D:", "", "*.*", wx.FD_OPEN)
        print("-=-=-=-=-=-=-=-=-=-")
        print(dlg.CharWidth)
        print(dlg.CharHeight)
        print("-=-=-=-=-=-=-=-=-=-")
        teste = (math.ceil(dlg.CharWidth*Ratio), math.ceil(dlg.CharHeight*Ratio))

        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()

            self.I_ref = cv2.imread(os.path.join(self.dirname, self.filename),0)  # Read grayscale

            ####### Bilateral Filter
            diag = np.sqrt(self.I_ref.shape[0] ** 2 + self.I_ref.shape[1] ** 2)
            sigmaSpace = 0.02 * diag
            sigmaColor = 20
            #self.I_ref = cv2.bilateralFilter(self.I_ref, -1, sigmaColor, sigmaSpace)
            self.I_ref = cv2.GaussianBlur(self.I_ref, (3, 3), 0)
            ####### Bilateral Filter

            if np.max(self.I_ref.shape) > Treshold_size:
                #from scipy.misc import imresize
                #self.I_ref = imresize(self.I_ref, Ratio, interp='bilinear', mode=None)
                ##self.I_ref = cv2.resize(src=heatmap, dsize=input_dims, interpolation=cv2.INTER_CUBIC)
                ##self.I_ref = cv2.resize(src=self.I_ref, dsize=Ratio, interpolation=cv2.INTER_CUBIC)
                from PIL import Image
                self.I_ref = np.array(Image.fromarray(obj=self.I_ref, mode=None).resize(size=teste, resample=Image.BICUBIC))
                
                # Plot Image
            self.plot1Image.draw(self.I_ref, 'Reference Image','X','Y')
            self.plot1ROI.draw(self.I_ref, 'Select ROI', 'X', 'Y')

            self.mask = np.zeros((self.I_ref.shape[0],self.I_ref.shape[1]))
            self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y')

        dlg.Destroy()  # Close file dialog


        # Load Second Image
        self.I_all = []
        dlg = wx.FileDialog(self, "Choose The Last Image", "D:", "", "*.*", wx.FD_OPEN)

        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()


            temp = cv2.imread(os.path.join(self.dirname, self.filename), 0)

            if np.max(temp.shape) > Treshold_size:
                #
                #
                #
                #####temp = imresize(temp, Ratio, interp='bilinear', mode=None)
                temp = np.array(Image.fromarray(obj=self.I_ref, mode=None).resize(size=teste, resample=Image.BICUBIC))
                #
                #
                #
                
            ####### Bilateral Filter
            diag = np.sqrt(temp.shape[0] ** 2 + temp.shape[1] ** 2)
            sigmaSpace = 0.02 * diag
            sigmaColor = 20
            #temp = cv2.bilateralFilter(temp, -1, sigmaColor, sigmaSpace)
            temp = cv2.GaussianBlur(temp, (3, 3), 0)
            ####### Bilateral Filter

            self.I_all.append(temp) # Read grayscale

            # Plot Image
            self.plot2Image.draw(self.I_all[0], 'Image 1', 'X', 'Y')

        dlg.Destroy()  # Close file dialog

        dlg = wx.MessageDialog(self, 'Mesh Loaded!', 'Loaded',  # Second argument is title
                               wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

        # Update tree

        self.Tree_photosMain.DeleteChildren(self.PhotosMain_root)
        self.PhotoRef = self.Tree_photosMain.AppendItem(self.PhotosMain_root,
                                                           'Reference Image' )
        self.Photosec = self.Tree_photosMain.AppendItem(self.PhotosMain_root,
                                                     '1')
        # Update Statusbar


        self.StatusBar.SetStatusText(
            "Images Loaded! Dimensions: " + str(self.I_ref.shape[0]) + " x " + str(
                self.I_ref.shape[1]))

        self.Actual_Image = 0

        # Create Empty variables
        Len_images = len(self.I_all)
        self.Exx = [None] * Len_images
        self.Eyy = [None] * Len_images
        self.Exy = [None] * Len_images
        self.disX= [None] * Len_images
        self.disY= [None] * Len_images
        self.disX_smooth = [None] * Len_images
        self.disY_smooth = [None] * Len_images
        self.InterpI = [None]
        self.I_ref_interp = [None]
        self.InterpS = [None] * Len_images
        self.I_f_interp = [None] * Len_images
        self.Len_images = Len_images

        self.StdX_fit = []#np.zeros((self.Exx[0].shape[0],self.Exx[0].shape[1]))
        self.StdY_fit = []#np.zeros((self.Exx[0].shape[0], self.Exx[0].shape[1]))
        self.StdXY_fit = []#np.zeros((self.Exx[0].shape[0], self.Exx[0].shape[1]))

        self.Exx_fit = [None] * self.Len_images
        self.Eyy_fit = [None] * self.Len_images
        self.Exy_fit = [None] * self.Len_images

    def SaveDictionary(self,key,value,Project=None):
        try:
        

            Project[key] = value

            print('Data')
        except:
            print('No data')

        return Project

    def Menu_saveProject_Callback(self,e):

        # Create dictionary with everything
        Project = {'Version: ': 1}
        try:
            Project = self.SaveDictionary( 'disX',self.disX ,Project)
            Project =self.SaveDictionary( 'disY',self.disY ,Project)
        except:
            pass

        try:
            Project =self.SaveDictionary( 'I_ref',self.I_ref,Project )
            Project =self.SaveDictionary( 'I_all',self.I_all,Project )
        except:
            pass

        try:
            Project =self.SaveDictionary( 'MaskX',self.MaskX ,Project)
            Project =self.SaveDictionary( 'MaskY',self.MaskY ,Project)
        except:
            pass

        try:
            Project =self.SaveDictionary( 'X',self.X,Project )
            Project =self.SaveDictionary( 'Y',self.Y,Project )
        except:
            pass

        try:
            Project =self.SaveDictionary( 'disX_smooth',self.disX_smooth,Project )
            Project =self.SaveDictionary( 'disY_smooth',self.disY_smooth ,Project)
        except:
            pass

        try:
            Project =self.SaveDictionary( 'mask',self.mask ,Project)
            Project =self.SaveDictionary( 'Rxi',self.Rxi ,Project)
            Project =self.SaveDictionary( 'Ryi',self.Ryi ,Project)
            Project =self.SaveDictionary( 'Rxf',self.Rxf ,Project)
            Project =self.SaveDictionary( 'Ryf',self.Ryf ,Project)
        except:
            pass 

        try:
            Project =self.SaveDictionary( 'Exx',self.Exx ,Project)
            Project =self.SaveDictionary( 'Eyy',self.Eyy ,Project)
            Project =self.SaveDictionary( 'Exy',self.Exy ,Project)
        except:
            pass 
        
        try:
            Project =self.SaveDictionary( 'Exx_fit',self.Exx_fit ,Project)
            Project =self.SaveDictionary( 'Eyy_fit',self.Eyy_fit ,Project)
            Project =self.SaveDictionary( 'Exy_fit',self.Exy_fit ,Project)
            Project =self.SaveDictionary( 'StdX_fit',self.StdX_fit ,Project)
            Project =self.SaveDictionary( 'StdY_fit',self.StdY_fit ,Project)
            Project =self.SaveDictionary( 'StdXY_fit',self.StdXY_fit ,Project)
        except:
            pass       
        
        



        #Project = {'disX':self.disX,'disY':self.disY,'I_ref':self.I_ref,'I_all':self.I_all,
        #           'MaskX':self.MaskX,'MaskY':self.MaskY,'X':self.X,'Y':self.Y,'disX_smooth':self.disX_smooth,
        #           'disY_smooth':self.disY_smooth,'mask':self.mask,'Rxi':self.Rxi,'Ryi':self.Ryi,'Rxf':self.Rxf,'Ryf':self.Ryf,
        #           'Exx':self.Exx,'Eyy':self.Eyy,'Exy':self.Exy,'Exx_fit':self.Exx_fit,'Eyy_fit':self.Eyy_fit,'Exy_fit':self.Exy_fit,
        #            'StdX_fit':self.StdX_fit,'StdY_fit':self.StdY_fit,'StdXY_fit':self.StdXY_fit}#,'InterpS':self.InterpS,'InterpI':self.InterpI,'I_f_interp':self.I_f_interp}

        # FD_OPEN save prompt
        dlg = wx.FileDialog(self, "Save project as:", 'C:', "", "*.*", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory() #GetPath
            f = os.path.join(self.dirname, self.filename)
            # Save into a file
            with FD_OPEN(f, 'w') as fa:
                pickle.dump(Project, fa)
            fa.close()

        dlg.Destroy()

        dlg = wx.MessageDialog(self, 'Done!', 'Saved',
                               wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

    def Menu_loadProject_Callback(self,e):

        dlg = wx.FileDialog(self, "Choose the project to load:", 'C:', "", "*.*", wx.FD_OPEN)

        if dlg.ShowModal() == wx.ID_OK:
            # Update Statusbar
            self.StatusBar.SetStatusText('Loading Project...')

            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            f = os.path.join(self.dirname, self.filename)
            # Load project and data
            with FD_OPEN(f) as fa:
                Project = pickle.load(fa)

            try: 
                self.disX = Project['disX'] 
            except: 
                "No data for disX"
            
            try: 
                self.disY = Project['disY']
            except: 
                "No data for disY"

            try: 
                self.disX_smooth = Project['disX_smooth']
            except: 
                "No data for disX_smooth"
            
            try: 
                self.disY_smooth = Project['disY_smooth']
            except: 
                "No data for disY_smooth"

            try: 
                self.mask = Project['mask']
            except: 
                "No data for mask"

            try: 
                self.MaskX = Project['MaskX']
            except: 
                "No data for MaskX"

            try: 
                self.MaskY = Project['MaskY']
            except: 
                "No data for MaskY"

            try: 
                self.I_ref = Project['I_ref']
                self.I_all = Project['I_all']
                self.Len_images = len(self.I_all)
                
            except: 
                "No data for I_ref and I_all"


            try: 
                self.X = Project['X']
                self.Y = Project['Y']
            except: 
                "No data for X and Y"

            try: 
                self.Rxi = Project['Rxi']
                self.Rxf = Project['Rxf']
                self.Ryi = Project['Ryi']
                self.Ryf = Project['Ryf']
            except: 
                "No data for R's"

            try: 
                self.Exx = Project['Exx']  
                self.Eyy = Project['Eyy']
                self.Exy = Project['Exy']
            except: 
                "No data for Deformations"

           
            

            try:
                self.StdX_fit = Project['StdX_fit']
                self.StdY_fit = Project['StdY_fit']
                self.StdXY_fit = Project['StdXY_fit']
                self.Exx_fit = Project['Exx_fit']
                self.Eyy_fit = Project['Eyy_fit']
                self.Exy_fit = Project['Exy_fit']
            except:
                self.StdX_fit = []  # np.zeros((self.Exx[0].shape[0],self.Exx[0].shape[1]))
                self.StdY_fit = []  # np.zeros((self.Exx[0].shape[0], self.Exx[0].shape[1]))
                self.StdXY_fit = []  # np.zeros((self.Exx[0].shape[0], self.Exx[0].shape[1]))

                #self.Exx_fit = [None] * self.Len_images
                #self.Eyy_fit = [None] * self.Len_images
                #self.Exy_fit = [None] * self.Len_images

            try:
                self.I_ref_interp = Project['I_ref_interp']
                self.InterpS = Project['InterpS']
                self.InterpI = Project['InterpI']
                self.I_f_interp = Project['I_f_interp']
            except:
                self.I_ref_interp = [None]
                self.InterpS = [None]
                self.InterpI = [None]
                self.I_f_interp = [None]
            # Do the plots and set variables

            # Images
            self.plot1Image.draw(self.I_ref, 'Reference Image','X','Y')
            self.plot2Image.draw(self.I_all[0], 'Image 1', 'X', 'Y')
            self.Tree_results.DeleteChildren(self.PhotosMain_root)
            self.PhotoRef = self.Tree_results.AppendItem(self.PhotosMain_root,
                                                         'Reference Image')
            self.Photosec = self.Tree_results.AppendItem(self.PhotosMain_root,'1')

            # ROI
            try:
                self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y')
                self.plot1ROI.draw(self.I_ref, 'Select ROI', 'X', 'Y')
            except:
                a=1
            #Displacement
            try:
                masked_array = np.ma.array(self.disY_smooth[0], mask=self.MaskY)
                self.plot1Disp.draw(self.I_ref, self.Y, self.X, masked_array, 'V Displacement', 'X', 'Y')
                self.plot1Data = masked_array
                self.plot1Disp.canvas.mpl_connect('motion_notify_event', self.VisuDisp1)
            except:
                a = 1

            
            # Update tree
            try:
                self.Tree_photosMain.DeleteChildren(self.PhotosMain_root)
                self.PhotoRef = self.Tree_photosMain.AppendItem(self.PhotosMain_root,
                                                                'Reference Image')

                self.PhotosTree = []
                for idx in range(len(self.I_all)):
                    self.PhotosTree.append(self.Tree_photosMain.AppendItem(self.PhotosMain_root,
                                                                        str(idx + 1)))
            except:
                a = 1


            # Results

        dlg.Destroy()

        dlg = wx.MessageDialog(self, 'Done!', 'Loaded',
                               wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

        # Update Statusbar
        self.StatusBar.SetStatusText('Project loaded')
        self.Actual_Image = 0
    ######################### TOOLBAR FUNCTIONS ##################




    ############################ Calculate result##################################

    def Toolbar_animate_Callback(self,e):

        Wait_time = float( self.ET_timeAnim.GetLineText(0) )
        First_image = int(self.ET_firstAnim.GetLineText(0))
        Last_image = int(self.ET_lastAnim.GetLineText(0) )
        if Last_image == 0:
            Last_image = self.Len_images
        Property = self.CB_propertyAnimation.GetSelection() # Exx, Eyy, Exy, U, V
        if Property == 0:
            Plot = self.Exx
        elif Property == 1:
            Plot = self.Eyy
        elif Property == 2:
            Plot = self.Exy
        elif Property == 3:
            Plot = self.disX_smooth
        elif Property == 4:
            Plot = self.disY_smooth
        elif Property == 5:
            Plot = self.Exx_fit
        elif Property == 6:
            Plot = self.Eyy_fit
        elif Property == 7:
            Plot = self.Exy_fit

        masked_array = np.ma.array(Plot[Last_image-1], mask=self.MaskY)

        Plot_max =  np.max(masked_array)
        Plot_min =  np.min(masked_array)

        print(Plot_max)
        print(Plot_min)

        Array_plot = np.arange(First_image,Last_image, dtype = int)
        Array_plot = np.hstack( (Array_plot,np.arange(Last_image-2,First_image-1,-1, dtype = int) ))

        for idx in Array_plot:#range(First_image,Last_image):

            masked_array = np.ma.array(Plot[idx], mask=self.MaskY)

            if self.CB_adapTreshAnim.IsChecked() == True:
                self.plot1Res.draw(self.I_all[idx], self.Y+self.disX_smooth[idx],
                                self.X+self.disY_smooth[idx], masked_array, 'Animation', 'X', 'Y')
            else:
                self.plot1Res.draw(self.I_all[idx], self.Y + self.disX_smooth[idx],
                                   self.X + self.disY_smooth[idx], masked_array, 'Animation', 'X', 'Y',Plot_min,Plot_max)
            plt.pause(Wait_time/100.)


    def Toolbar_Calculate_Callback(self, e):


        self.StatusBar.SetStatusText('Done Calculation.')

        # Mesage box
        dlg = wx.MessageDialog(self, 'Done Calculation (Yeah!)', 'Finish',  # Second argument is title
                               wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()


    ################## Context Menu Callbacks ###################
    def Tree_control_Callback(self, e):
        Sel_Index = self.Tree_control.GetFocusedItem()
        Sel_Index = self.Tree_control.GetItemText(Sel_Index)

        ### Launcher creates wxMenu. ###
        menu = wx.Menu()
        menu.Append(0, 'Displacement constrain')
        menu.Append(1, 'Impedance Constrain')
        wx.EVT_MENU(menu, 0, self.Context_menu_B_fixed)

        ### 5. Launcher displays menu with call to PopupMenu, invoked on the source component, passing event's GetPoint. ###
        self.PopupMenu(menu, e.GetPoint())
        menu.Destroy()  # destroy to avoid mem leak

    def Context_menu_B_fixed(self, e):
        a = 1
        print(5)

    ############################# Trees Functions ######################
    def Tree_photosMain_Callback(self,e):

        try:
            temp = self.Tree_photosMain.GetFocusedItem()
            Index = int(self.Tree_photosMain.GetItemText(temp)) #Tree_photosMain
            self.Actual_Image = Index-1
            self.plot2Image.draw(self.I_all[self.Actual_Image], 'Image '+str(Index), 'X', 'Y')
        except:
            return

    ################################## Buttons callback ##################

    ######### ROI ##############


    # Clear all
    def PB_clearAll_Callback(self,e):

        self.rec_counter = 0
        self.mask[:] = 0
        self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y')

        # Update status button
        self.ST_GUIStatus.SetLabel('ROI Cleared')

    # Rectangle +
    def onclickRec(self,event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))

        if self.rec_counter == -1:
            return

        self.rec_counter += 1

        if self.rec_counter == 1:
            self.Ryi = int(np.round(event.ydata))
            self.Rxi = int(np.round(event.xdata))
            self.ST_GUIStatus.SetLabel('Select Down Right Corner Point')
        if self.rec_counter == 2:
            self.Ryf = int(np.round(event.ydata))
            self.Rxf = int(np.round(event.xdata))


            self.mask[self.Ryi:self.Ryf, self.Rxi:self.Rxf] = 1
            self.plot1ROI.draw(self.I_ref, 'Select ROI', 'X', 'Y')

            self.plot1ROI.axes.imshow(self.mask,cmap = 'gray',alpha=0.3)
            self.plot1ROI.canvas.draw()

            self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y')
            self.rec_counter = 0
            self.ST_GUIStatus.SetLabel('Done!')
            self.rec_counter = -1

    def PB_addRec_Callback(self,e):

        self.rec_counter = 0
        self.plot1ROI.canvas.mpl_connect('button_press_event', self.onclickRec)

        # Update status button
        self.ST_GUIStatus.SetLabel('Select Up Left Corner Point')

    # Rectangle -
    def onclickRecremove(self,event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))

        if self.rec_counter_remv == -1:
            return

        self.rec_counter_remv += 1

        if self.rec_counter_remv == 1:
            self.Ryri = int(np.round(event.ydata))
            self.Rxri = int(np.round(event.xdata))
            self.ST_GUIStatus.SetLabel('Select Down Right Corner Point')
        if self.rec_counter_remv == 2:
            self.Ryrf = int(np.round(event.ydata))
            self.Rxrf = int(np.round(event.xdata))


            self.mask[self.Ryri:self.Ryrf, self.Rxri:self.Rxrf] = 0
            self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y')

            self.plot1ROI.draw(self.I_ref, 'Select ROI', 'X', 'Y')

            self.plot1ROI.axes.imshow(self.mask, cmap='gray', alpha=0.3)
            self.plot1ROI.canvas.draw()

            self.rec_counter = 0
            self.ST_GUIStatus.SetLabel('Done!')
            self.rec_counter = -1

    def PB_removeRec_Callback(self,e):

        self.rec_counter_remv = 0
        self.plot1ROI.canvas.mpl_connect('button_press_event', self.onclickRecremove)

        # Update status button
        self.ST_GUIStatus.SetLabel('Select Up Left Corner Point')



    # Ellipse -
    def onclickRecEllRemove(self,event):

        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))

        if self.ell_counter == -1:
            return

        self.ell_counter += 1

        if self.ell_counter == 1:
            self.EcY = np.round(event.ydata)
            self.EcX = np.round(event.xdata)
            self.ST_GUIStatus.SetLabel('Select Vertical Radius')

        if self.ell_counter == 2:
            self.Vyy = np.round(event.ydata)
            self.Vyx = np.round(event.xdata)
            self.ST_GUIStatus.SetLabel('Select Horizontal Radius')

        if self.ell_counter == 3:

            self.Vxy = np.round(event.ydata)
            self.Vxx = np.round(event.xdata)

            rx = abs(self.EcX - self.Vxx)
            ry = abs(self.EcY - self.Vyy)

            ny = self.mask.shape[0]
            nx = self.mask.shape[1]
            #import pdb
            #pdb.set_trace()
            # Just to get the right ROI
            #self.EcY = 300#375.#476.
            #self.EcX = 375#150.#341.
            #ry = 75.#50.#48.
            #rx = 75.#50.#86.

            #y, x = np.ogrid[-self.EcX:nx - self.EcX, -self.EcY:ny - self.EcY]
            y, x = np.ogrid[0:nx, 0:ny ]
            x = x - self.EcY
            y = y - self.EcX
            #pdb.set_trace()
            mask = (x * x)/(ry**2) + (y * y)/(rx**2) <= 1
            mask = np.transpose(mask)

            # Save for easy buttons later
            self.mask_ell = self.mask.copy()
            self.ell_x = x
            self.ell_y = y
            self.ell_rx = rx
            self.ell_ry = ry

            # End saving for easy buttons :3

            self.mask[mask] = 0

            self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y')

            self.plot1ROI.draw(self.I_ref, 'Select ROI', 'X', 'Y')

            self.plot1ROI.axes.imshow(self.mask, cmap='gray', alpha=0.3)
            self.plot1ROI.canvas.draw()

            self.ell_counter = -1
            self.ST_GUIStatus.SetLabel('Done')


    def PB_remElli_Callback(self,e):

        self.ell_counter = 0
        self.plot1ROI.canvas.mpl_connect('button_press_event', self.onclickRecEllRemove)

        # Update status button
        self.ST_GUIStatus.SetLabel('Select Center Point')

    ### Easy buttons Ellipse
    def PB_Xplus_Callback(self,e):
        self.ell_y += 1
        self.Update_ellipse(e)

    def PB_Yplus_Callback(self, e):
        self.ell_x += 1
        self.Update_ellipse(e)

    def PB_RxPlus_Callback(self, e):
        self.ell_rx += 1
        self.Update_ellipse(e)

    def PB_RyPlus_Callback(self, e):
        self.ell_ry += 1
        self.Update_ellipse(e)

    def PB_Xminus_Callback(self,e):
        self.ell_y -= 1
        self.Update_ellipse(e)

    def PB_Yminus_Callback(self, e):
        self.ell_x -= 1
        self.Update_ellipse(e)

    def PB_RxMinus_Callback(self, e):
        self.ell_rx -= 1
        self.Update_ellipse(e)

    def PB_RyMinus_Callback(self, e):
        self.ell_ry -= 1
        self.Update_ellipse(e)



    def Update_ellipse(self,e):
        mask = (self.ell_x * self.ell_x) / (self.ell_ry ** 2) + (self.ell_y * self.ell_y) / (self.ell_rx ** 2) <= 1
        mask = np.transpose(mask)

        self.mask = self.mask_ell.copy()
        self.mask[mask] = 0

        self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y')

        self.plot1ROI.draw(self.I_ref, 'Select ROI', 'X', 'Y')

        self.plot1ROI.axes.imshow(self.mask, cmap='gray', alpha=0.3)
        self.plot1ROI.canvas.draw()

        self.ell_counter = -1
        self.ST_GUIStatus.SetLabel('Done')

    # Finish Easy buttons Ellipse

    # Ellipse +

    def onclickRecEll(self,event):

        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))

        if self.ell_counter_plus == -1:
            return

        self.ell_counter_plus += 1

        if self.ell_counter_plus == 1:
            self.EcY = np.round(event.ydata)
            self.EcX = np.round(event.xdata)
            self.ST_GUIStatus.SetLabel('Select Vertical Radius')

        if self.ell_counter_plus == 2:
            self.Vyy = np.round(event.ydata)
            self.Vyx = np.round(event.xdata)
            self.ST_GUIStatus.SetLabel('Select Horizontal Radius')

        if self.ell_counter_plus == 3:

            self.Vxy = np.round(event.ydata)
            self.Vxx = np.round(event.xdata)

            rx = abs(self.EcX - self.Vxx)
            ry = abs(self.EcY - self.Vyy)

            ny = self.mask.shape[0]
            nx = self.mask.shape[1]

            #y, x = np.ogrid[-self.EcX:nx - self.EcX, -self.EcY:ny - self.EcY]
            y, x = np.ogrid[0:nx, 0:ny ]
            x = x - self.EcY
            y = y - self.EcX
            #pdb.set_trace()
            mask = (x * x)/(ry**2) + (y * y)/(rx**2) <= 1
            mask = np.transpose(mask)

            self.mask[mask] = 1
            self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y')

            self.plot1ROI.draw(self.I_ref, 'Select ROI', 'X', 'Y')

            self.plot1ROI.axes.imshow(self.mask, cmap='gray', alpha=0.3)
            self.plot1ROI.canvas.draw()

            self.ell_counter = -1
            self.ST_GUIStatus.SetLabel('Done')


    def PB_addElli_Callback(self,e):

        self.ell_counter_plus = 0
        self.plot1ROI.canvas.mpl_connect('button_press_event', self.onclickRecEll)

        # Update status button
        self.ST_GUIStatus.SetLabel('Select Center Point')

        #pdb.set_trace()
    ########################### DISPLACEMENT ################
    def PB_visuDisp_Callback(self,e):

        Value = self.CB_dispShowChoice.GetSelection()

        if Value == 0: # Plot X
            masked_array = np.ma.array(self.disX_smooth[self.Actual_Image], mask=self.MaskX)
            self.plot1Disp.draw(self.I_all[self.Actual_Image], self.Y+ self.disX_smooth[self.Actual_Image], self.X+ self.disY_smooth[self.Actual_Image], masked_array, 'U Displacement Smooth', 'X', 'Y')

        elif Value == 1: #PlotY
            masked_array = np.ma.array(self.disY_smooth[self.Actual_Image], mask=self.MaskY)
            self.plot1Disp.draw(self.I_all[self.Actual_Image],self.Y+ self.disX_smooth[self.Actual_Image], self.X+ self.disY_smooth[self.Actual_Image], masked_array, 'V Displacement Smooth', 'X', 'Y')
        else: # Plot X+Y
            Plot = np.sqrt(self.disY_smooth[self.Actual_Image]**2+self.disX_smooth[self.Actual_Image]**2)
            masked_array = np.ma.array(Plot, mask=self.MaskY)
            self.plot1Disp.draw(self.I_all[self.Actual_Image],self.Y+ self.disX_smooth[self.Actual_Image], self.X+ self.disY_smooth[self.Actual_Image], masked_array, 'U+V Displacement Smooth', 'X', 'Y')

        self.plot1Data = masked_array
        self.plot1Disp.canvas.mpl_connect('motion_notify_event', self.VisuDisp1)
    ################################ STRAIN #############################

    def Mouse_movement(self,e):
        # print('you pressed', e.button, e.xdata, e.ydata)
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

                # Exy
        #dlg = wx.FileDialog(self, "Choose the Exy regressor:", 'C:', "", "*.*", wx.FD_OPEN)
        #if dlg.ShowModal() == wx.ID_OK:
         #   self.filename = dlg.GetFilename()
          #  self.dirname = dlg.GetDirectory()
           # f = os.path.join(self.dirname, self.filename)
            #self.GP_xy = joblib.load(f)

    def PB_filterStrain_Callback(self,e):

        # Parameter Inputs
        Filter_size = int(self.ET_filterSizeStrain.GetLineText(0))
        Algo = self.CB_filtersDisp1.GetSelection()
        Let_us = self.ET_usChooseStrain.IsChecked()
        Smooth = self.CB_smoothStrain.IsChecked()
        Filter = self.CB_filtersStrain.GetSelection()
        Value = float(self.ET_filterValueStrain.GetLineText(0))
        All_images = self.CB_allImages.IsChecked()



        if Algo == 0: # Direct Gradient

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

        elif Algo == 1: # Plane Fit

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
                #U[U != self.disX_smooth[self.Actual_Image].copy()] = 0
                Def_xxU, Def_xyU = self.sgolay2d(U,Filter_size,order=2,derivative='both')#self.Plane_fit(U,Filter_size)

                V = self.disX_smooth[self.Actual_Image].copy() * 0 + 1
                V[U != self.disX_smooth[self.Actual_Image].copy()] = 0
                Def_xxV, Def_xyV = self.sgolay2d(V,Filter_size,order=2,derivative='both')#self.Plane_fit(V, Filter_size)

                Def_xxV[Def_xxV==0] = 1
                Def_xyV[Def_xyV == 0] = 1

                Def_xx = Def_xxU#/Def_xxV
                Def_xy = Def_xyU# / Def_xyV

                # Y
                U = self.disY_smooth[self.Actual_Image].copy()
                #U[U != self.disY_smooth[self.Actual_Image].copy()] = 0
                Def_yxU, Def_yyU = self.sgolay2d(U,Filter_size,order=2,derivative='both')#self.Plane_fit(U, Filter_size)

                V = self.disY_smooth[self.Actual_Image].copy() * 0 + 1
                V[U != self.disY_smooth[self.Actual_Image].copy()] = 0
                Def_yxV, Def_yyV = self.sgolay2d(V,Filter_size,order=2,derivative='both')#self.Plane_fit(V, Filter_size)

                Def_yxV[Def_yxV == 0] = 1
                Def_yyV[Def_yyV == 0] = 1

                Def_yx = Def_yxU #/ Def_yxV
                Def_yy = Def_yyU #/ Def_yyV

                # Calculate deformation
                #self.Exx[self.Actual_Image] = Def_xx*10**5#1. / 2 * (2 * Def_xx + Def_xx ** 2 + Def_xy ** 2) *10**5
                #self.Exy[self.Actual_Image] = 1./2*(Def_yx+Def_xy)*10**5#1. / 2 * (Def_xy + Def_yx + Def_xx * Def_xy + Def_yx * Def_yy) *10**5
                #self.Eyy[self.Actual_Image] = Def_yy*10**5#1. / 2 * (2 * Def_yy + Def_xy ** 2 + Def_yy ** 2) *10**5

                self.Exx[self.Actual_Image] =  1. / 2 * (2 * Def_xx + Def_xx ** 2 + Def_xy ** 2) *self.DIC_calibration_factor
                self.Exy[self.Actual_Image] =  1. / 2 * (Def_xy + Def_yx + Def_xx * Def_xy + Def_yx * Def_yy) *self.DIC_calibration_factor
                self.Eyy[self.Actual_Image] =  1. / 2 * (2 * Def_yy + Def_xy ** 2 + Def_yy ** 2) *self.DIC_calibration_factor

        if Smooth:
            self.Exx[self.Actual_Image] = self.SmoothT(self.Exx[self.Actual_Image], Filter, Value)
            self.Eyy[self.Actual_Image] = self.SmoothT(self.Eyy[self.Actual_Image], Filter, Value)
            self.Exy[self.Actual_Image] = self.SmoothT(self.Exy[self.Actual_Image], Filter, Value)


        if self.CB_gaussianCorrection.IsChecked():
            #sigma = self.All_parameters[4]
            #sigma[np.isnan(sigma)] = 0

            #Max_X = np.max(self.Exx[self.Actual_Image][~np.isnan(self.Exx[self.Actual_Image])])
            #Max_Y =np.max(self.Eyy[self.Actual_Image][~np.isnan(self.Exx[self.Actual_Image])])

            #X = np.hstack( (self.Exx[self.Actual_Image].reshape(-1,1)/np.max(self.Exx[self.Actual_Image][~np.isnan(self.Exx[self.Actual_Image])]),
             #                                self.Eyy[self.Actual_Image].reshape(-1,1)/np.max(self.Eyy[self.Actual_Image][~np.isnan(self.Exx[self.Actual_Image])]),
              #                               self.disX_smooth[self.Actual_Image].reshape(-1, 1)/np.max(self.disX_smooth[self.Actual_Image][~np.isnan(self.Exx[self.Actual_Image])]),
              #                               self.disY_smooth[self.Actual_Image].reshape(-1, 1)/np.max(self.disY_smooth[self.Actual_Image][~np.isnan(self.Exx[self.Actual_Image])]),
              #                               sigma.reshape(-1, 1)/np.max(sigma)) )
            X = np.hstack( ( self.All_parameters[0].reshape(-1,1),self.All_parameters[1].reshape(-1,1),
                           self.All_parameters[2].reshape(-1, 1),self.All_parameters[3].reshape(-1,1),
                           self.All_parameters[4].reshape(-1, 1),self.All_parameters[5].reshape(-1,1)))


            Indexes = ~np.isnan(X[:,0])
            X = X[Indexes,:]#.reshape(-1,5)
            self.Exx_correction = self.GP_x.predict(X)
            self.Eyy_correction = self.GP_y.predict(X)
            #self.Exy_correction = self.GP_xy.predict(X)

            #pdb.set_trace()
            Dumb = np.zeros((self.Exx[self.Actual_Image].shape[0]*self.Exx[self.Actual_Image].shape[1],1))

            Dumb[Indexes] += self.Exx_correction.reshape(-1,1)
            #Dumb = Dumb*Max_X#self.SmoothT(Dumb, Filter, Value)
            self.Exx_correction = Dumb*self.DIC_calibration_factor_ML

            if self.ET_usChooseStrain.IsChecked():
                self.Exx_correction = self.SmoothT(self.Exx_correction, Filter, Filter_size)

            self.Exx[self.Actual_Image] += Dumb.reshape((self.Exx[self.Actual_Image].shape[0],self.Exx[self.Actual_Image].shape[1]))

            Dumb = np.zeros((self.Exx[self.Actual_Image].shape[0] * self.Exx[self.Actual_Image].shape[1], 1))
            Dumb[Indexes] += self.Eyy_correction.reshape(-1,1)
            #Dumb = Dumb*Max_Y#self.SmoothT(Dumb, Filter, Value)*Max_Y

            self.Eyy_correction = Dumb*self.DIC_calibration_factor_ML

            if self.ET_usChooseStrain.IsChecked():
                self.Eyy_correction = self.SmoothT(self.Eyy_correction, Filter, Filter_size)

            self.Eyy[self.Actual_Image] += Dumb.reshape((self.Exx[self.Actual_Image].shape[0],self.Exx[self.Actual_Image].shape[1]))

            #Dumb[~np.isnan(X)] += self.Exy_correction

            self.Exy[self.Actual_Image] += (1/2*(self.Eyy_correction+self.Exx_correction)).reshape((self.Exx[self.Actual_Image].shape[0],self.Exx[self.Actual_Image].shape[1]))

        if False:#self.ET_usChooseStrain.IsChecked():
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

        #dlg = wx.MessageDialog(self, 'Calculation Done.', 'Done',
         #                          wx.OK | wx.ICON_INFORMATION)
        #wx.FutureCall(5000, dlg.Destroy)
        #dlg.ShowModal()
        #dlg.Destroy()

    def Plane_fit(self,I,Radius):

        Def_y = np.zeros((I.shape[0],I.shape[1]))
        Def_x = np.zeros((I.shape[0],I.shape[1]))

        I_i_padded = np.lib.pad(I.astype(float), (Radius, Radius), 'edge')

        #import pdb
        #pdb.set_trace()
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


                 C = np.matmul(linalg.pinv(A),Bx.flatten())#linalg.lstsq(A, Bx.flatten())
                 #C = C[0]

                 Def_x[idxI,idxJ] = C[1]
                 Def_y[idxI,idxJ] = C[2]

                 #print 'Dx' + str(C[1])
                 #print 'Dy' + str(C[2])
                 print(float(t)/(I.shape[0]*I.shape[1]))
                 t += 1

        return Def_y,Def_x

    def sgolay2d(self,z, window_size, order, derivative=None):
        """
        """
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

            #AAA = convolve_fft(Z, -r,crop='off')
            #AAA = AAA[(len(r) - 1) / 2:-(len(r) - 1) / 2, (len(r) - 1) / 2:-(len(r) - 1) / 2]

            #BBB = convolve_fft(Z, -c, crop='off')
            #BBB = BBB[(len(r) - 1) / 2:-(len(r) - 1) / 2, (len(r) - 1) / 2:-(len(r) - 1) / 2]

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

    ######################## MAIN ######################
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
            Values = it.Interp_spline(Points, self.InterpI['coeff'])  # USE INTERP S
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
                    #pdb.set_trace()
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

                    #elif idxImages > 0: # If second image, use guess from previous one
                        #P = P_temp[idxX][idxY]
                    # Gauss

                    # If use the original image to calculate or the interpolated one
                    if Original_I == True:
                        from scipy.optimize import minimize
                        #import pdb
                        #pdb.set_trace()
                        #res = minimize(GN_opt.DIC_direct,P,args=(self.I_f_interp[self.Actual_Image], self.I_ref_interp, Template, np.array([j , i]), self.InterpS[self.Actual_Image], Algo), method='Nelder-Mead')
                        #C = 1-0.5*res.fun
                        #P = res.x
                        #C_max = -1
                        #Pos_max = [0,0]
                        #for idxX in np.arange(-12,12):
                            #for idxYY in np.arange(-12,12):
                        P, C = GN_opt.GN_opt(self.I_f_interp[self.Actual_Image], self.I_ref_interp, Template, P,
                                                    np.array([j , i , self.Ryi, self.Rxi]),
                                                     self.InterpS[self.Actual_Image],
                                                     self.InterpI, Algo,GaussianWindow,Template_mask)
                    else:
                        #import pdb
                        #pdb.set_trace()
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

                        #if C < 0.6 and idxImages > 0: # If still low correlation, stay the same as before
                         #   P = P_temp[idxX][idxY]

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

            #self.disX[self.Actual_Image] = disX
            #self.disY[self.Actual_Image] = disY
            #self.disS[self.Actual_Image] = np.sqrt(disX**2+disY**2)

            # Interpolate
            self.X,self.Y = np.meshgrid([range(int(self.Ryi+w),int(self.Ryf-w))][0],[range(int(self.Rxi+w),int(self.Rxf-w))][0])
            self.X = np.transpose(self.X)
            self.Y = np.transpose(self.Y)

            #points = np.transpose(np.vstack((MeshLocX.flatten('F'),MeshLocY.flatten('F'))))

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
            #self.InterpS[self.Actual_Image] = []

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
        #dlg = wx.MessageDialog(self, 'Calculation Completed.', 'Done',
        #                       wx.OK | wx.ICON_INFORMATION)
        #dlg.ShowModal()
        #dlg.Destroy()

        self.StatusBar.SetStatusText('Displacement Calculated!')

    def TemplateMatch(self,I_f, Template, Method, Center):

        res = cv2.matchTemplate(I_f, Template,
                                Method)  # ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        disX = max_loc[0] + len(Template) / 2 - Center[0]
        disY = max_loc[1] + len(Template) / 2 - Center[1]
        C = max_val

        return disX, disY, C


####################################### DISPLACEMENT #############################
    def PB_filterDisp_Callback(self,e):

        # Parameter Inputs
        Filter_length = float(self.ET_filterSizeDisp.GetLineText(0))
        Std = float(self.ET_stdGausDisp.GetLineText(0))
        Filter = self.CB_filtersDisp.GetSelection()
        Let_us = self.ET_usChooseDisp.IsChecked()
        All_images = self.CB_allImages.IsChecked()

        if Filter == 1:
            Value = Filter_length
        elif Filter == 0:
            Value = Std

        Length = 1
        if All_images == True:
            Length = self.Len_images
        ######### Filter Displacements########
        for idx in range(Length):

            if All_images == True: # If not all, just pass the actual index
                self.Actual_Image = idx

            self.disY_smooth[self.Actual_Image] = self.SmoothT(self.disY[self.Actual_Image],Filter,Value)#UU/VV#ndimage.gaussian_filter(self.disY, sigma=(Std, Std, 0), order=0)

            self.disX_smooth[self.Actual_Image] = self.SmoothT(self.disX[self.Actual_Image],Filter,Value)#UU / VV  # ndimage.gaussian_filter(self.disY, sigma=(Std, Std, 0), order=0)

        masked_array = np.ma.array(self.disY_smooth[self.Actual_Image], mask=self.MaskY)
        self.plot1Disp.draw(self.I_all[self.Actual_Image],self.Y+ self.disX_smooth[self.Actual_Image], self.X+ self.disY_smooth[self.Actual_Image], masked_array, 'V Displacement Smooth', 'X', 'Y')

        self.plot1Data = masked_array
        self.plot1Disp.canvas.mpl_connect('motion_notify_event', self.VisuDisp1)
        #dlg = wx.MessageDialog(self, 'Done Filtering.', 'Finish',  # Second argument is title
         #                      wx.OK | wx.ICON_INFORMATION)
        #dlg.ShowModal()
        #dlg.Destroy()

    ####################### RESULT ###############################
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

            #self.StrainGates_xx = self.Tree_results.AppendItem(self.StrainGates_X, 'Exx: '+str(Exx*(10**6)))
            #self.StrainGates_yy = self.Tree_results.AppendItem(self.StrainGates_X, 'Eyy: ' + str(Eyy * (10 ** 6)))
            #self.StrainGates_xy = self.Tree_results.AppendItem(self.StrainGates_X, 'Exy: ' + str(Exy * (10 ** 6)))


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
        #import pdb
        #pdb.set_trace()


##################################### CALIBRATE #########################################

    def PB_lineCalibrate_Callback(self,e):

        self.Calibration = 0
        self.plot1Res.canvas.mpl_connect('button_press_event', self.onClickCalibration)

        # Update status button
        self.StatusBar.SetLabel('Select First point (In X)')

    def onClickCalibration(self, e):

        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (e.button, e.x, e.y, e.xdata, e.ydata))

        self.Calibration += 1
        if self.Calibration == 1:
            self.Cal_one_Y = e.ydata
            self.Cal_one_X = e.xdata
            self.StatusBar.SetLabel('Select Second point(In X)')
        if self.Calibration == 2:
            self.Cal_two_Y = e.ydata
            self.Cal_two_X = e.xdata

            Real_value = float(self.ET_distCalib.GetLineText(0))
            Pixel_Value = abs(self.Cal_one_X-self.Cal_two_X)

            self.Calibration_factor = Real_value/Pixel_Value

            self.StatusBar.SetLabel('Calibration Factor Calculated. '+str(self.Calibration_factor)+' mm per pixels')

    def PB_showCalibrate_Callback(self,e):

        Property = self.CB_propertyCalibration.GetSelection()

        if Property == 0:
            Plot = self.disX_smooth[self.Actual_Image]
            title = 'U[mm]'
        else:
            Plot = self.disY_smooth[self.Actual_Image]
            title = 'V[mm]'

        masked_array = np.ma.array(Plot*self.Calibration_factor, mask=self.MaskY)
        self.plot1Res.draw(self.I_all[self.Actual_Image],  self.Y+self.disX_smooth[self.Actual_Image],
                                self.X+self.disY_smooth[self.Actual_Image], masked_array, title, 'X', 'Y')



    ################### GRID ANALYSIS ################################
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
                self.Toolbar_cplus_Callback(e,w[idxW],steps[idxS])#Toolbar_calculate_Callback(e,w[idxW],steps[idxS])
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
            with FB_FD_OPEN(f) as fa:
                X = pickle.load(fa)

        dlg = wx.FileDialog(self, "Choose the Y data:", 'C:', "", "*.*", wx.FD_OPEN)

        if dlg.ShowModal() == wx.ID_OK:
            # Update Statusbar
            self.StatusBar.SetStatusText('Loading Project...')

            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            f = os.path.join(self.dirname, self.filename)
            # Load project and data
            with FD_OPEN(f) as fa:
                Y = pickle.load(fa)

        dlg = wx.FileDialog(self, "Choose the Eyy data:", 'C:', "", "*.*", wx.FD_OPEN)

        if dlg.ShowModal() == wx.ID_OK:
            # Update Statusbar
            self.StatusBar.SetStatusText('Loading Project...')

            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            f = os.path.join(self.dirname, self.filename)
            # Load project and data
            with FD_OPEN(f) as fa:
                Eyy_Ansys = pickle.load(fa)


        if True:

            ################################################

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

                    self.Toolbar_cplus_Callback(e, w[idxW], steps[idxS])#Toolbar_calculate_Callback(e, w[idxW], steps[idxS])
                    self.PB_filterDisp_Callback(e)

                    Points_X = np.arange(self.Rxi, self.Rxf +1).astype(int)#Points_X = np.arange(self.Rxi + w[idxW], self.Rxf - w[idxW]).astype(int)
                    Points_Y = np.arange(self.Ryi , self.Ryf +1).astype(int)


                    # X = np.transpose(X)
                    # Y = np.transpose(Y)
                    X = -X_ori[np.ix_(Points_Y, Points_X)]
                    Y = -Y_ori[np.ix_(Points_Y, Points_X)]
                    Eyy_Ansys = Eyy_Ansys[np.ix_(Points_Y, Points_X)]

                   # X = self.SmoothT(X*self.MaskY,0,15.)
                    #Y = self.SmoothT(Y*self.MaskY, 0, 15.)
                    Mask_pre = self.MaskY.copy()
                    Mask_pre[Mask_pre == 0] = np.nan

                    X = self.SmoothT(X*Mask_pre,0,3.)#*Mask_pre
                    Y = self.SmoothT(Y*Mask_pre, 0,3.)#*Mask_pre
                    #Eyy_Ansys = self.SmoothT(Eyy_Ansys*Mask_pre, 0,3.)

                    #X = np.ma.array(X, mask=self.MaskY)
                    #Y = np.ma.array(Y, mask=self.MaskY)


                    #Def_xy, Def_xx = np.gradient(X)
                    #Def_yy, Def_yx = np.gradient(Y)

                    #Exx_b = 1. / 2 * (2 * Def_xx + Def_xx ** 2 + Def_xy ** 2) * 10 ** 5
                    #Exy_b = 1. / 2 * (Def_xy + Def_yx + Def_xx * Def_xy + Def_yx * Def_yy) * 10 ** 5
                    #Eyy_b = 1. / 2 * (2 * Def_yy + Def_xy ** 2 + Def_yy ** 2) * 10 ** 5

                    ################ Plane Fit###########
                    Filter_size = 21
                    U = X*Mask_pre#self.disX_smooth[self.Actual_Image].copy()
                    #U[U != (X*Mask_pre).copy()] = 0
                    Def_xxU, Def_xyU = self.sgolay2d(U, Filter_size, order=2,
                                                     derivative='both')  # self.Plane_fit(U,Filter_size)

                    Def_xx = Def_xxU   #/Def_xxV
                    Def_xy = Def_xyU   #/ Def_xyV

                    # Y
                    U = Y* Mask_pre  # self.disX_smooth[self.Actual_Image].copy()
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
                    ######################################################################
                    #Exx_b = self.SmoothT(Exx_b,0,5.)
                    #Exy_b = self.SmoothT(Exy_b, 0, 5.)
                    #Eyy_b = self.SmoothT(Eyy_b, 0, 5.)

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
                    #U[U != self.disY_smooth[self.Actual_Image].copy()] = 0
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


                    #import pdb
                    #pdb.set_trace()
                    ######################################################################



                    #Def_xy, Def_xx = np.gradient(disX)
                    #Def_yy, Def_yx = np.gradient(disY)

                    #Exx = 1. / 2 * (2 * Def_xx + Def_xx ** 2 + Def_xy ** 2) * 10 ** 5
                    #Exy = 1. / 2 * (Def_xy + Def_yx + Def_xx * Def_xy + Def_yx * Def_yy) * 10 ** 5
                    #Eyy = 1. / 2 * (2 * Def_yy + Def_xy ** 2 + Def_yy ** 2) * 10 ** 5


                    #DiffX = (disX - np.mean(disX)) - (X - np.mean(X))
                    #DiffY = (disY - np.mean(disY)) - (Y - np.mean(Y))
                    DiffX = Exx_b - Exx
                    DiffY = Eyy_b - Eyy
                    DiffXY = Exy_b - Exy

                    # Difference
                    #plt.figure(0)
                    #plt.subplot(1, 2, 1)
                    Displacement = False

                    if Displacement == True:


                        plt.imshow(DiffY)
                        plt.title('Difference[rms]: ' + str(
                            np.round(1000 * np.sqrt(np.mean(DiffY ** 2))) / 1000)+'. w:'+str(w[idxW])+'. s:'+str(steps[idxS]))
                        plt.colorbar()

                        plt.savefig(Path+'/Error_Y_Ansys_benchmark_w'+str(w[idxW])+'_s'+str(steps[idxS])+'.png', bbox_inches='tight',dpi=600)

                        #plt.subplot(1, 2, 2)
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


                    #self.PB_filterStrain_Callback(e)

                    # Deformations
                    #masked_array = np.ma.array(self.Exx[self.Actual_Image], mask=self.MaskY)
                    #Exx = masked_array[Pos + Comp_vec[idxW], :]

                    #masked_array = np.ma.array(self.Eyy[self.Actual_Image], mask=self.MaskY)
                    #Eyy = masked_array[Pos + Comp_vec[idxW], :]

                    #masked_array = np.ma.array(self.Exy[self.Actual_Image], mask=self.MaskY)
                    #Exy = masked_array[Pos + Comp_vec[idxW], :]

                    #plt.subplot(3, 1, 1)
                    #plt.plot(Exx)
                    #plt.subplot(3, 1, 2)
                    #plt.plot(Eyy)
                    #plt.subplot(3, 1, 3)
                    #plt.plot(Exy)
                    #plt.show()

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
            #X = np.transpose(X)
            #Y = np.transpose(Y)
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

            import pdb
            #pdb.set_trace()


    def Menu_loadROI_Callback(self,e):
        dlg = wx.FileDialog(self, "Choose the mask data:", 'C:', "", "*.*", wx.FD_OPEN)

        if dlg.ShowModal() == wx.ID_OK:
            # Update Statusbar
            self.StatusBar.SetStatusText('Loading Project...')

            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            f = os.path.join(self.dirname, self.filename)
            # Load project and data
            with FD_OPEN(f) as fa:
                mask = pickle.load(fa)

        if self.mask == None:
            self.mask = mask
        else:
            self.mask = np.logical_and(self.mask,~mask)

        self.mask[self.mask==True] = 1
        self.mask[self.mask == False] = 0
        self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y')
        self.ST_GUIStatus.SetLabel('Done!')

    def Menu_saveMask_Callback(self,e):


        # Create dictionary with everything
        Project = { 'Rxi': self.Rxi, 'Ryi': self.Ryi,
                   'Rxf': self.Rxf, 'Ryf': self.Ryf,
                   'mask': self.mask}  # ,'InterpS':self.InterpS,'InterpI':self.InterpI,'I_f_interp':self.I_f_interp}

        # FD_OPEN save prompt
        dlg = wx.FileDialog(self, "Save project as:", 'C:', "", "*.*", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()  # GetPath
            f = os.path.join(self.dirname, self.filename)
            # Save into a file
            with FD_OPEN(f, 'w') as fa:
                pickle.dump(Project, fa)
            fa.close()

        dlg.Destroy()

        dlg = wx.MessageDialog(self, 'Done!', 'Saved',
                               wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()
    def Menu_loadMask_Callbak(self,e):
        dlg = wx.FileDialog(self, "Choose the project to load:", 'C:', "", "*.*", wx.FD_OPEN)

        if dlg.ShowModal() == wx.ID_OK:
            # Update Statusbar
            self.StatusBar.SetStatusText('Loading Project...')

            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            f = os.path.join(self.dirname, self.filename)
            # Load project and data
            with FD_OPEN(f) as fa:
                Project = pickle.load(fa)

            self.Rxi = Project['Rxi']
            self.Ryi = Project['Ryi']
            self.Rxf = Project['Rxf']
            self.Ryf = Project['Ryf']
            self.mask = Project['mask']

            self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y')
            self.ST_GUIStatus.SetLabel('Done!')



    ########################### HELP FUNCTIONS ####################
    def SmoothT(self,I,Filter,Value):

        if self.CB_cluster.IsChecked() == True:
            from sklearn import mixture
            from sklearn.cluster import KMeans
            n_components = int(self.ET_clusterNumber.GetLineText(0))
            gmm = KMeans(n_clusters=n_components, random_state=0)#mixture.GaussianMixture(n_components=n_components)
            X = np.squeeze(np.array([self.X.flatten().reshape((1,-1)),self.Y.flatten().reshape((1,-1)),I.flatten().reshape((1,-1))]))
            X = X.reshape((-1,3))

            for idx in range(3):
                X[np.isnan(X)[:,idx],:] = 0

            # Limit X
            Variables = int(self.ET_trainingVector.GetLineText(0))

            if Variables == 1:
                pass
            elif Variables == 2:
                X = X[:,2].reshape((-1,1))
            elif Variables == 3:
                X = X[:,[0,1]]
            #eval('X = X[:,'+Variables+']')



            gmm.fit(X)
            Index = gmm.predict(X)

            I_all = []
            for idx in range(n_components):
                I_temp = I.flatten()
                I_temp[~(Index==idx)] = np.nan

                I_all.append(I_temp.reshape(I.shape))


        if Filter == 0:


            if self.CB_cluster.IsChecked() == True:

                Smooth = np.zeros_like(I)
                for idx in range(n_components):

                    U = I_all[idx].copy()
                    U[U != I_all[idx].copy()] = 0
                    UU = ndimage.gaussian_filter(U, sigma=(Value, Value), order=0)  #

                    V = I_all[idx].copy() * 0 + 1
                    V[U != I_all[idx].copy()] = 0
                    VV = ndimage.gaussian_filter(V, sigma=(Value, Value), order=0)  #

                    Smooth_temp = UU / VV
                    Smooth_temp[np.isnan(Smooth_temp)] = 0
                    Smooth += Smooth_temp


            else:
            #Smooth =  ndimage.gaussian_filter(I, sigma=(Value, Value), order=0)
                U = I.copy()
                U[U != I.copy()] = 0
                UU = ndimage.gaussian_filter(U, sigma=(Value, Value), order=0)#

                V = I.copy() * 0 + 1
                V[U != I.copy()] = 0
                VV = ndimage.gaussian_filter(V, sigma=(Value, Value), order=0)#

                Smooth = UU / VV

                Filter_length = float(self.ET_filterSizeDisp.GetLineText(0))

                if Filter_length>0: # In case use both filters
                    U = Smooth.copy()
                    U[U != Smooth.copy()] = 0
                    UU = cv2.bilateralFilter(U.astype(np.float32), int(Filter_length), int(Filter_length / 2),
                                             int(Filter_length / 2))  # ndimage.filters.median_filter(U, size=int(Value))

                    V = Smooth.copy() * 0 + 1
                    V[U != Smooth.copy()] = 0
                    VV = cv2.bilateralFilter(V.astype(np.float32), int(Filter_length), int(Filter_length / 2),
                                             int(Filter_length / 2))  ## ndimage.filters.median_filter(V, size=int(Value))

                    Smooth = UU / VV


        if Filter == 1:

            #Smooth = ndimage.gaussian_filter(I, sigma=(Value, Value), order=0)
            U = I.copy()
            U[U != I.copy()] = 0
            UU = cv2.bilateralFilter(U.astype(np.float32),int(Value),int(Value/2),int(Value/2))#ndimage.filters.median_filter(U, size=int(Value))

            V = I.copy() * 0 + 1
            V[U != I.copy()] = 0
            VV = cv2.bilateralFilter(V.astype(np.float32),int(Value),int(Value/2),int(Value/2))## ndimage.filters.median_filter(V, size=int(Value))

            Smooth = UU / VV

            Std = float(self.ET_stdGausDisp.GetLineText(0))

            if Std > 0:  # In case use both filters

                U = Smooth.copy()
                U[U != Smooth.copy()] = 0
                UU = ndimage.gaussian_filter(U, sigma=(Std, Std), order=0)  #

                V = Smooth.copy() * 0 + 1
                V[U != Smooth.copy()] = 0
                VV = ndimage.gaussian_filter(V, sigma=(Std, Std), order=0)  #

                Smooth = UU / VV

        return Smooth


if __name__ == '__main__':
    app = wx.App(False)
    frame = CalcFrame(None)
    frame.Show(True)
    app.MainLoop()


