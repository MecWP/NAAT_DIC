class ImagesTreeMixin:
    def Tree_photosMain_Callback(self,e):
        try:
            temp = self.Tree_photosMain.GetFocusedItem()
            Index = int(self.Tree_photosMain.GetItemText(temp)) #Tree_photosMain
            self.Actual_Image = Index-1
            self.plot2Image.draw(self.I_all[self.Actual_Image], 'Image '+str(Index), 'X', 'Y')
        except:
            return