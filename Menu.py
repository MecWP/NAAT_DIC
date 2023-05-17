import wx #check
import cv2 #check
import os #check
import numpy as np #check
from PIL import Image

import wx.xrc #checar necessidade
import wx.grid #checar necessidade


class MenuMixin:
    
    def MenuFolder_Callback(self,e):
        Treshold_size = int(self.ET_treshImage.GetLineText(0) )#1500
        Ratio = float(self.ET_treshRate.GetLineText(0) )
        self.StatusBar.SetStatusText('Loading Images...')
        
        # Load Reference Image
        dlg = wx.FileDialog(self, "Choose The First Image", "D:", "", "*.*", wx.FD_OPEN) #https://pythonspot.com/wxpython-file-dialog/
        if dlg.ShowModal() == wx.ID_OK:
            print(dlg.GetPath()) #Test Print
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            self.I_ref = cv2.imread(os.path.join(self.dirname, self.filename), 0)  # Read grayscale

            # Bilateral Filter (NOT IN USE)
            diag = np.sqrt(self.I_ref.shape[0] ** 2 + self.I_ref.shape[1] ** 2)
            sigmaSpace = 0.08 * diag
            sigmaColor = 75
            # Bilateral Filter (NOT IN USE)

            if np.max(self.I_ref.shape) > Treshold_size:
                heigth = len(self.I_ref)
                width = len(self.I_ref[0])
                self.I_ref = np.array(Image.fromarray(obj=self.I_ref, mode=None).resize(size=(int(width*Ratio), int(heigth*Ratio)), resample=Image.BICUBIC))


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
        Files = os.listdir(dlg.GetPath()) #List of images
        self.I_all = []
        for image in Files:
            temp = cv2.imread(os.path.join(dlg.GetPath(), image), 0)

            # Bilateral Filter (NOT IN USE)
            diag = np.sqrt(temp.shape[0] ** 2 + temp.shape[1] ** 2)
            sigmaSpace = 0.08 * diag
            sigmaColor = 75
            # Bilateral Filter (NOT IN USE)

            if np.max(temp.shape) > Treshold_size:
                heigth = len(temp)
                width = len(temp[0])
                temp = np.array(Image.fromarray(obj=temp, mode=None).resize(size=(int(width*Ratio), int(heigth*Ratio)), resample=Image.BICUBIC))
                
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

        # Load Reference Image
        dlg = wx.FileDialog(self, "Choose The First Image", "D:", "", "*.*", wx.FD_OPEN)

        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()

            self.I_ref = cv2.imread(os.path.join(self.dirname, self.filename),0)  # Read grayscale
            
            # Bilateral Filter
            diag = np.sqrt(self.I_ref.shape[0] ** 2 + self.I_ref.shape[1] ** 2)
            sigmaSpace = 0.02 * diag
            sigmaColor = 20
            #self.I_ref = cv2.bilateralFilter(self.I_ref, -1, sigmaColor, sigmaSpace)
            self.I_ref = cv2.GaussianBlur(self.I_ref, (3, 3), 0)
            # Bilateral Filter (NOT IN USE)

            if np.max(self.I_ref.shape) > Treshold_size:
                heigth = len(self.I_ref)
                width = len(self.I_ref[0])
                self.I_ref = np.array(Image.fromarray(obj=self.I_ref, mode=None).resize(size=(int(width*Ratio), int(heigth*Ratio)), resample=Image.BICUBIC))
                
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
                heigth = len(temp)
                width = len(temp[0])
                temp = np.array(Image.fromarray(obj=temp, mode=None).resize(size=(int(width*Ratio), int(heigth*Ratio)), resample=Image.BICUBIC))
                
            # Bilateral Filter (NOT IN USE)
            diag = np.sqrt(temp.shape[0] ** 2 + temp.shape[1] ** 2)
            sigmaSpace = 0.02 * diag
            sigmaColor = 20
            #temp = cv2.bilateralFilter(temp, -1, sigmaColor, sigmaSpace)
            temp = cv2.GaussianBlur(temp, (3, 3), 0)
            # Bilateral Filter (NOT IN USE)

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