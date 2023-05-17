class CalibrateMixin:
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
    