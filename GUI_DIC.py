import matplotlib #check
matplotlib.use('WXAgg')

import gui
import matplotlib.pyplot as plt #check
plt.style.use('ggplot')
import MatplotlibWxClasses as pltWx

import Menu
import ROI
import Displacement as disp
import Strain
import Result
import Calibrate
import SaveLoad as SL
import CplusFunction as cplus
import UnknownMethods as UM
import Animation
import ImagesTree as ITree
import Calculation
import Smooth

class CalcFrame(Menu.MenuMixin, ROI.ROIMixin, disp.DisplacementMixin, Strain.StrainMixin, Result.ResultMixin, Calibrate.CalibrateMixin, SL.SaveLoadMixin, cplus.CplusMixin, UM.UnkownMethodsMixin, Animation.AnimationMixin, ITree.ImagesTreeMixin, Calculation.CalculationMixin, Smooth.SmoothMixin, gui.MainFrame):

    # constructor
    def __init__(self, parent):

        gui.MainFrame.__init__(self, parent)

        self.frame = gui.MainFrame

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