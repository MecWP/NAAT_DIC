import wx #check
import os #check
import numpy as np #check
import pickle #check

import wx.xrc #checar necessidade
import wx.grid #checar necessidade

class SaveLoadMixin:
    
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

        dlg = wx.FileDialog(self, "Save project as:", 'C:', "", "*.*", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory() #GetPath
            f = os.path.join(self.dirname, self.filename)

            # Save into a file
            file = open(f, 'wb')
            pickle.dump(Project, file)
            file.close()

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
            
            # Load a file
            file = open(f, 'rb')
            Project = pickle.load(file)
            file.close()

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
                self.StdX_fit = []
                self.StdY_fit = []
                self.StdXY_fit = []
                
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
                a = 1
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
    
    def Menu_loadROI_Callback(self,e):
        dlg = wx.FileDialog(self, "Choose the mask data:", 'C:', "", "*.*", wx.FD_OPEN)

        if dlg.ShowModal() == wx.ID_OK:
            # Update Statusbar
            self.StatusBar.SetStatusText('Loading Project...')

            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            f = os.path.join(self.dirname, self.filename)
            # Load project and data
            file = open(f, 'rb')
            mask = pickle.load(file)
            file.close()

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
            file = open(f, 'wb')
            pickle.dump(Project, file)
            file.close()

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
            file = open(f, 'rb')
            Project = pickle.load(file)
            file.close()
            

            self.Rxi = Project['Rxi']
            self.Ryi = Project['Ryi']
            self.Rxf = Project['Rxf']
            self.Ryf = Project['Ryf']
            self.mask = Project['mask']

            self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y')
            self.ST_GUIStatus.SetLabel('Done!')