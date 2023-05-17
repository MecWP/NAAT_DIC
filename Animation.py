import numpy as np #check
import matplotlib.pyplot as plt #check

class AnimationMixin:
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