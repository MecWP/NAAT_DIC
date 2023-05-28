import numpy as np #check

class ROIMixin:
    
    # Clear all selected ROIs (called when button is pressed)
    def PB_clearAll_Callback(self,e):
        self.rec_counter = 0 #Sets the record counter to zero
        self.mask[:] = 0 #Sets all values in the mask array to zero
        self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y') #Redraws the mask on the UI in the ROI area (AREA 2)
        self.plot1ROI.draw(self.I_ref, 'Select ROI', 'X', 'Y') #Reset selection screen (AREA 1))

        # Update status button
        self.ST_GUIStatus.SetLabel('ROI Cleared')

    # Add a quadrilateral region of interest
    def onclickRec(self,event):
        # Display the coordinates where the mouse click occurred
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))

        # Ignore all mouse clicks if rec_counter is set to -1, indicating that ROI selection has been disabled.
        if self.rec_counter == -1:
            return

        # Increase the click counter
        self.rec_counter += 1

        # First click: the x and y coordinates are saved as the upper left corner of the rectangle.
        if self.rec_counter == 1:
            self.Ryi = int(np.round(event.ydata))
            self.Rxi = int(np.round(event.xdata))
            self.ST_GUIStatus.SetLabel('Select Down Right Corner Point')
        # Second click: the x and y coordinates are saved as the down right corner of the rectangle.
        if self.rec_counter == 2:
            self.Ryf = int(np.round(event.ydata))
            self.Rxf = int(np.round(event.xdata))

            # The region within these coordinates in the mask matrix is set to 1, indicating the presence of an ROI.
            self.mask[self.Ryi:self.Ryf, self.Rxi:self.Rxf] = 1
            self.plot1ROI.draw(self.I_ref, 'Select ROI', 'X', 'Y') #The original image is redrawn with the new ROI. (AREA 1)

            # The ROI mask is superimposed on the original image with a transparency of 30%. (AREA 1)
            self.plot1ROI.axes.imshow(self.mask,cmap = 'gray',alpha=0.3)
            self.plot1ROI.canvas.draw()

            # The mask image is redrawn to show the new ROI. (AREA 2)
            self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y')
            self.rec_counter = 0
            self.ST_GUIStatus.SetLabel('Done!')

            #the counter is set to -1, indicating that ROI selection is disabled until user decides to add another one.
            self.rec_counter = -1

    # Initiate quadrilateral ROI selection (called when button is pressed)
    # Event handling method
    def PB_addRec_Callback(self,e):
        # Reset record counter to zero
        self.rec_counter = 0
        #Connect the onclickRec function to the mouse button press event on the canvas self.plot1ROI.canvas
        self.plot1ROI.canvas.mpl_connect('button_press_event', self.onclickRec)

        # Update status button
        self.ST_GUIStatus.SetLabel('Select Up Left Corner Point')

    # Remove a quadrilateral region of the ROI
    def onclickRecremove(self,event):
        # Display the coordinates where the mouse click occurred:
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))

        # Ignore all mouse clicks if rec_counter_remv is set to -1, indicating that ROI selection has been disabled.
        if self.rec_counter_remv == -1:
            return

        # Increase the remove click counter
        self.rec_counter_remv += 1

        # First click: the x and y coordinates are saved as the upper left corner of the rectangle.
        if self.rec_counter_remv == 1:
            self.Ryri = int(np.round(event.ydata))
            self.Rxri = int(np.round(event.xdata))
            self.ST_GUIStatus.SetLabel('Select Down Right Corner Point')
        # Second click: the x and y coordinates are saved as the down right corner of the rectangle.
        if self.rec_counter_remv == 2:
            self.Ryrf = int(np.round(event.ydata))
            self.Rxrf = int(np.round(event.xdata))

            # The region within these coordinates in the mask matrix is set to 0, indicating that the region
            # has no ROI.
            self.mask[self.Ryri:self.Ryrf, self.Rxri:self.Rxrf] = 0
            # The mask image is redrawn to show the new ROI. (AREA 2)
            self.plot2ROI.draw(self.mask, 'Mask', 'X', 'Y') 
            #The original image is redrawn with the new ROI. (AREA 1)
            self.plot1ROI.draw(self.I_ref, 'Select ROI', 'X', 'Y')

            # The ROI mask is superimposed on the original image with a transparency of 30%. (AREA 1)
            self.plot1ROI.axes.imshow(self.mask, cmap='gray', alpha=0.3)
            self.plot1ROI.canvas.draw()

            self.rec_counter = 0
            self.ST_GUIStatus.SetLabel('Done!')
            # the counter is set to -1, indicating that remove ROI selection is disabled until user decides to remove another one.
            self.rec_counter = -1

    # Initiate quadrilateral remove region of ROI selection (called when button is pressed)
    # Event handling method
    def PB_removeRec_Callback(self,e):
        # Reset record remove counter to zero
        self.rec_counter_remv = 0
        #Connect the onclickRec function to the mouse button press event on the canvas self.plot1ROI.canvas
        self.plot1ROI.canvas.mpl_connect('button_press_event', self.onclickRecremove)

        # Update status button
        self.ST_GUIStatus.SetLabel('Select Up Left Corner Point')

    # Remove a ellipse region of the ROI
    def onclickRecEllRemove(self,event):
        # Display the coordinates where the mouse click occurred:
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))

        # Ignore all mouse clicks if ell_counter is set to -1, indicating that removing ellipse region of ROI has been disabled.
        if self.ell_counter == -1:
            return

        # Increase the remove click counter
        self.ell_counter += 1

        # First click: the x and y coordinates are saved as the center of the ellipse.
        if self.ell_counter == 1:
            self.EcY = np.round(event.ydata)
            self.EcX = np.round(event.xdata)
            self.ST_GUIStatus.SetLabel('Select Vertical Radius')
        # Second click: the x and y coordinates are saved as the vertical radius of the ellipse
        if self.ell_counter == 2:
            self.Vyy = np.round(event.ydata)
            self.Vyx = np.round(event.xdata)
            self.ST_GUIStatus.SetLabel('Select Horizontal Radius')
        # Thrid click: the x and y coordinates are saved as the horizontal radius of the ellipse
        if self.ell_counter == 3:

            self.Vxy = np.round(event.ydata)
            self.Vxx = np.round(event.xdata)
            
            # Calculate the two radii of the ellipse
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