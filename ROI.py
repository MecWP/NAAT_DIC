import numpy as np #check

class ROIMixin:
    
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

            y, x = np.ogrid[0:nx, 0:ny ]
            x = x - self.EcY
            y = y - self.EcX

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

    # Easy buttons Ellipse
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

            y, x = np.ogrid[0:nx, 0:ny ]
            x = x - self.EcY
            y = y - self.EcX
            
            self.mask_ell = self.mask.copy()
            self.ell_x = x
            self.ell_y = y
            self.ell_rx = rx
            self.ell_ry = ry
            
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