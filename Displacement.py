import numpy as np #check

class DisplacementMixin:
    
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
