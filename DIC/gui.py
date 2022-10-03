# -*- coding: utf-8 -*- 

###########################################################################
## Python code generated with wxFormBuilder (version Mar 29 2017)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc

###########################################################################
## Class MainFrame
###########################################################################

class MainFrame ( wx.Frame ):
	
	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"DIC Thiago Lobato V0.2. thiagohgl@hotmail.com", pos = wx.DefaultPosition, size = wx.Size( 1300,790 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
		bSizer1 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_splitter1 = wx.SplitterWindow( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SP_LIVE_UPDATE|wx.SIMPLE_BORDER )
		self.m_splitter1.Bind( wx.EVT_IDLE, self.m_splitter1OnIdle )
		
		self.m_splitter1.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
		
		self.m_panel1 = wx.Panel( self.m_splitter1, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		self.m_panel1.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
		
		bSizer3 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_splitter2 = wx.SplitterWindow( self.m_panel1, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SP_LIVE_UPDATE )
		self.m_splitter2.Bind( wx.EVT_IDLE, self.m_splitter2OnIdle )
		
		self.m_splitter2.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_BTNFACE ) )
		
		self.m_panel4 = wx.Panel( self.m_splitter2, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		self.m_panel4.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_BTNFACE ) )
		
		bSizer5 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_notebook2 = wx.Notebook( self.m_panel4, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.TAB_Main = wx.Panel( self.m_notebook2, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		self.TAB_Main.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
		
		gSizer3 = wx.GridSizer( 0, 2, 0, 0 )
		
		self.m_staticText1 = wx.StaticText( self.TAB_Main, wx.ID_ANY, u"Windows size:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText1.Wrap( -1 )
		gSizer3.Add( self.m_staticText1, 0, wx.ALL, 5 )
		
		self.ET_sizeWindow = wx.TextCtrl( self.TAB_Main, wx.ID_ANY, u"15", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer3.Add( self.ET_sizeWindow, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText2 = wx.StaticText( self.TAB_Main, wx.ID_ANY, u"Step:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText2.Wrap( -1 )
		gSizer3.Add( self.m_staticText2, 0, wx.ALL, 5 )
		
		self.ET_step = wx.TextCtrl( self.TAB_Main, wx.ID_ANY, u"[10,10]", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer3.Add( self.ET_step, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText48 = wx.StaticText( self.TAB_Main, wx.ID_ANY, u"Automatic Refinement", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText48.Wrap( -1 )
		gSizer3.Add( self.m_staticText48, 0, wx.ALL, 5 )
		
		self.CB_stepAutomatic = wx.CheckBox( self.TAB_Main, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer3.Add( self.CB_stepAutomatic, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText3 = wx.StaticText( self.TAB_Main, wx.ID_ANY, u"Algorithm:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText3.Wrap( -1 )
		gSizer3.Add( self.m_staticText3, 0, wx.ALL, 5 )
		
		CB_algChoiceChoices = [ u"ZNSSD 1st Order", u"ZNSSD 2nd Order" ]
		self.CB_algChoice = wx.Choice( self.TAB_Main, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, CB_algChoiceChoices, 0 )
		self.CB_algChoice.SetSelection( 0 )
		gSizer3.Add( self.CB_algChoice, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText4 = wx.StaticText( self.TAB_Main, wx.ID_ANY, u"Calculate for all images?", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText4.Wrap( -1 )
		gSizer3.Add( self.m_staticText4, 0, wx.ALL, 5 )
		
		self.CB_allImages = wx.CheckBox( self.TAB_Main, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer3.Add( self.CB_allImages, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText411 = wx.StaticText( self.TAB_Main, wx.ID_ANY, u"Gauss weighted?", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText411.Wrap( -1 )
		gSizer3.Add( self.m_staticText411, 0, wx.ALL, 5 )
		
		self.CB_gaussCentered = wx.CheckBox( self.TAB_Main, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer3.Add( self.CB_gaussCentered, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText42 = wx.StaticText( self.TAB_Main, wx.ID_ANY, u"Correlation Threshold", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText42.Wrap( -1 )
		gSizer3.Add( self.m_staticText42, 0, wx.ALL, 5 )
		
		self.ET_tresholdMain = wx.TextCtrl( self.TAB_Main, wx.ID_ANY, u"0.9", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer3.Add( self.ET_tresholdMain, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText47 = wx.StaticText( self.TAB_Main, wx.ID_ANY, u"Number of Threads:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText47.Wrap( -1 )
		gSizer3.Add( self.m_staticText47, 0, wx.ALL, 5 )
		
		self.ET_parallelCores = wx.TextCtrl( self.TAB_Main, wx.ID_ANY, u"3", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer3.Add( self.ET_parallelCores, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText41 = wx.StaticText( self.TAB_Main, wx.ID_ANY, u"Status:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText41.Wrap( -1 )
		gSizer3.Add( self.m_staticText41, 0, wx.ALL, 5 )
		
		self.m_staticText361 = wx.StaticText( self.TAB_Main, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText361.Wrap( -1 )
		gSizer3.Add( self.m_staticText361, 0, wx.ALL, 5 )
		
		self.ST_GUIStatus = wx.StaticText( self.TAB_Main, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.ST_GUIStatus.Wrap( -1 )
		gSizer3.Add( self.ST_GUIStatus, 0, wx.ALL, 5 )
		
		self.m_staticText401 = wx.StaticText( self.TAB_Main, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText401.Wrap( -1 )
		gSizer3.Add( self.m_staticText401, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText51 = wx.StaticText( self.TAB_Main, wx.ID_ANY, u"Image Treshold size [Pixels]:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText51.Wrap( -1 )
		gSizer3.Add( self.m_staticText51, 0, wx.ALL, 5 )
		
		self.ET_treshImage = wx.TextCtrl( self.TAB_Main, wx.ID_ANY, u"1500", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer3.Add( self.ET_treshImage, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText52 = wx.StaticText( self.TAB_Main, wx.ID_ANY, u"Reduction Rate:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText52.Wrap( -1 )
		gSizer3.Add( self.m_staticText52, 0, wx.ALL, 5 )
		
		self.ET_treshRate = wx.TextCtrl( self.TAB_Main, wx.ID_ANY, u"0.2", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer3.Add( self.ET_treshRate, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		self.TAB_Main.SetSizer( gSizer3 )
		self.TAB_Main.Layout()
		gSizer3.Fit( self.TAB_Main )
		self.m_notebook2.AddPage( self.TAB_Main, u"DIC Main", True )
		self.TAB_Animation = wx.Panel( self.m_notebook2, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		self.TAB_Animation.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
		
		gSizer13 = wx.GridSizer( 0, 2, 0, 0 )
		
		self.m_staticText36 = wx.StaticText( self.TAB_Animation, wx.ID_ANY, u"Property: ", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText36.Wrap( -1 )
		gSizer13.Add( self.m_staticText36, 0, wx.ALL, 5 )
		
		CB_propertyAnimationChoices = [ u"Exx", u"Eyy", u"Exy", u"U", u"V", u"ExxFit", u"EyyFit", u"ExyFit" ]
		self.CB_propertyAnimation = wx.Choice( self.TAB_Animation, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, CB_propertyAnimationChoices, 0 )
		self.CB_propertyAnimation.SetSelection( 0 )
		gSizer13.Add( self.CB_propertyAnimation, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText37 = wx.StaticText( self.TAB_Animation, wx.ID_ANY, u"First Image:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText37.Wrap( -1 )
		gSizer13.Add( self.m_staticText37, 0, wx.ALL, 5 )
		
		self.ET_firstAnim = wx.TextCtrl( self.TAB_Animation, wx.ID_ANY, u"1", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer13.Add( self.ET_firstAnim, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText38 = wx.StaticText( self.TAB_Animation, wx.ID_ANY, u"Last Image (0 = last):", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText38.Wrap( -1 )
		gSizer13.Add( self.m_staticText38, 0, wx.ALL, 5 )
		
		self.ET_lastAnim = wx.TextCtrl( self.TAB_Animation, wx.ID_ANY, u"0", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer13.Add( self.ET_lastAnim, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText39 = wx.StaticText( self.TAB_Animation, wx.ID_ANY, u"Time between images[ms]:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText39.Wrap( -1 )
		gSizer13.Add( self.m_staticText39, 0, wx.ALL, 5 )
		
		self.ET_timeAnim = wx.TextCtrl( self.TAB_Animation, wx.ID_ANY, u"20", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer13.Add( self.ET_timeAnim, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText40 = wx.StaticText( self.TAB_Animation, wx.ID_ANY, u"Adaptative axis Limit?", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText40.Wrap( -1 )
		gSizer13.Add( self.m_staticText40, 0, wx.ALL, 5 )
		
		self.CB_adapTreshAnim = wx.CheckBox( self.TAB_Animation, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer13.Add( self.CB_adapTreshAnim, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		self.TAB_Animation.SetSizer( gSizer13 )
		self.TAB_Animation.Layout()
		gSizer13.Fit( self.TAB_Animation )
		self.m_notebook2.AddPage( self.TAB_Animation, u"Animation", False )
		self.TAB_Calibrate = wx.Panel( self.m_notebook2, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		self.TAB_Calibrate.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_MENU ) )
		
		gSizer14 = wx.GridSizer( 0, 2, 0, 0 )
		
		self.m_staticText46 = wx.StaticText( self.TAB_Calibrate, wx.ID_ANY, u"Give a distance in mm:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText46.Wrap( -1 )
		gSizer14.Add( self.m_staticText46, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.ET_distCalib = wx.TextCtrl( self.TAB_Calibrate, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer14.Add( self.ET_distCalib, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.ST_calibrated = wx.StaticText( self.TAB_Calibrate, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.ST_calibrated.Wrap( -1 )
		gSizer14.Add( self.ST_calibrated, 0, wx.ALL, 5 )
		
		self.PB_lineCalibrate = wx.Button( self.TAB_Calibrate, wx.ID_ANY, u"Select line distance", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer14.Add( self.PB_lineCalibrate, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText382 = wx.StaticText( self.TAB_Calibrate, wx.ID_ANY, u"Property:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText382.Wrap( -1 )
		gSizer14.Add( self.m_staticText382, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		CB_propertyCalibrationChoices = [ u"U", u"V" ]
		self.CB_propertyCalibration = wx.Choice( self.TAB_Calibrate, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, CB_propertyCalibrationChoices, 0 )
		self.CB_propertyCalibration.SetSelection( 0 )
		gSizer14.Add( self.CB_propertyCalibration, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText391 = wx.StaticText( self.TAB_Calibrate, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText391.Wrap( -1 )
		gSizer14.Add( self.m_staticText391, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.PB_showCalibrate = wx.Button( self.TAB_Calibrate, wx.ID_ANY, u"Show", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer14.Add( self.PB_showCalibrate, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		self.TAB_Calibrate.SetSizer( gSizer14 )
		self.TAB_Calibrate.Layout()
		gSizer14.Fit( self.TAB_Calibrate )
		self.m_notebook2.AddPage( self.TAB_Calibrate, u"Calibrate", False )
		
		bSizer5.Add( self.m_notebook2, 1, wx.EXPAND |wx.ALL, 5 )
		
		
		self.m_panel4.SetSizer( bSizer5 )
		self.m_panel4.Layout()
		bSizer5.Fit( self.m_panel4 )
		self.m_panel5 = wx.Panel( self.m_splitter2, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer4 = wx.BoxSizer( wx.VERTICAL )
		
		self.Tree_photosMain = wx.TreeCtrl( self.m_panel5, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TR_DEFAULT_STYLE )
		bSizer4.Add( self.Tree_photosMain, 1, wx.ALL|wx.EXPAND, 5 )
		
		
		self.m_panel5.SetSizer( bSizer4 )
		self.m_panel5.Layout()
		bSizer4.Fit( self.m_panel5 )
		self.m_splitter2.SplitHorizontally( self.m_panel4, self.m_panel5, 480 )
		bSizer3.Add( self.m_splitter2, 100, wx.EXPAND, 5 )
		
		
		self.m_panel1.SetSizer( bSizer3 )
		self.m_panel1.Layout()
		bSizer3.Fit( self.m_panel1 )
		self.m_panel2 = wx.Panel( self.m_splitter1, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		self.m_panel2.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
		
		bSizer6 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_notebook3 = wx.Notebook( self.m_panel2, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_notebook3.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
		
		self.TAB_images = wx.Panel( self.m_notebook3, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		self.TAB_images.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
		
		bSizer7 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.Panel_plot1 = wx.Panel( self.TAB_images, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		bSizer7.Add( self.Panel_plot1, 1, wx.EXPAND |wx.ALL, 5 )
		
		self.Panel_plot2 = wx.Panel( self.TAB_images, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		bSizer7.Add( self.Panel_plot2, 1, wx.EXPAND |wx.ALL, 5 )
		
		
		self.TAB_images.SetSizer( bSizer7 )
		self.TAB_images.Layout()
		bSizer7.Fit( self.TAB_images )
		self.m_notebook3.AddPage( self.TAB_images, u"Images", False )
		self.Tab_ROI = wx.Panel( self.m_notebook3, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		self.Tab_ROI.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
		
		bSizer8 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.Panel_plot1ROI = wx.Panel( self.Tab_ROI, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		bSizer8.Add( self.Panel_plot1ROI, 3, wx.EXPAND |wx.ALL, 5 )
		
		bSizer9 = wx.BoxSizer( wx.VERTICAL )
		
		self.Panel_addremROI = wx.Panel( self.Tab_ROI, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		self.Panel_addremROI.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_MENU ) )
		
		bSizer171 = wx.BoxSizer( wx.VERTICAL )
		
		self.PB_addRec = wx.Button( self.Panel_addremROI, wx.ID_ANY, u"Add Rectangle", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer171.Add( self.PB_addRec, 1, wx.ALL|wx.EXPAND, 10 )
		
		self.PB_addElli = wx.Button( self.Panel_addremROI, wx.ID_ANY, u"Add Ellipse", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer171.Add( self.PB_addElli, 1, wx.ALL|wx.EXPAND, 10 )
		
		self.PB_remPoly = wx.Button( self.Panel_addremROI, wx.ID_ANY, u"Remove Polygon", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer171.Add( self.PB_remPoly, 1, wx.ALL|wx.EXPAND, 10 )
		
		self.PB_removeRec = wx.Button( self.Panel_addremROI, wx.ID_ANY, u"Remove Rectangle", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer171.Add( self.PB_removeRec, 1, wx.ALL|wx.EXPAND, 10 )
		
		self.PB_remElli = wx.Button( self.Panel_addremROI, wx.ID_ANY, u"Remove Ellipse", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer171.Add( self.PB_remElli, 1, wx.ALL|wx.EXPAND, 10 )
		
		self.PB_clearAll = wx.Button( self.Panel_addremROI, wx.ID_ANY, u"Clear all", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer171.Add( self.PB_clearAll, 1, wx.ALL|wx.EXPAND, 10 )
		
		
		self.Panel_addremROI.SetSizer( bSizer171 )
		self.Panel_addremROI.Layout()
		bSizer171.Fit( self.Panel_addremROI )
		bSizer9.Add( self.Panel_addremROI, 1, wx.EXPAND |wx.ALL, 5 )
		
		bSizer161 = wx.BoxSizer( wx.VERTICAL )
		
		
		bSizer9.Add( bSizer161, 1, wx.EXPAND, 5 )
		
		self.Panel_Grid_Ellipse = wx.Panel( self.Tab_ROI, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		gSizer9 = wx.GridSizer( 0, 6, 0, 0 )
		
		self.m_staticText431 = wx.StaticText( self.Panel_Grid_Ellipse, wx.ID_ANY, u"X:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText431.Wrap( -1 )
		gSizer9.Add( self.m_staticText431, 0, wx.ALL, 5 )
		
		self.PB_Xplus = wx.Button( self.Panel_Grid_Ellipse, wx.ID_ANY, u"+", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer9.Add( self.PB_Xplus, 0, wx.ALL, 5 )
		
		self.PB_Xminus = wx.Button( self.Panel_Grid_Ellipse, wx.ID_ANY, u"-", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer9.Add( self.PB_Xminus, 0, wx.ALL, 5 )
		
		self.m_staticText441 = wx.StaticText( self.Panel_Grid_Ellipse, wx.ID_ANY, u"Y:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText441.Wrap( -1 )
		gSizer9.Add( self.m_staticText441, 0, wx.ALL, 5 )
		
		self.PB_Yplus = wx.Button( self.Panel_Grid_Ellipse, wx.ID_ANY, u"+", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer9.Add( self.PB_Yplus, 0, wx.ALL, 5 )
		
		self.PB_Yminus = wx.Button( self.Panel_Grid_Ellipse, wx.ID_ANY, u"-", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer9.Add( self.PB_Yminus, 0, wx.ALL, 5 )
		
		self.m_staticText451 = wx.StaticText( self.Panel_Grid_Ellipse, wx.ID_ANY, u"Rx:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText451.Wrap( -1 )
		gSizer9.Add( self.m_staticText451, 0, wx.ALL, 5 )
		
		self.PB_RxPlus = wx.Button( self.Panel_Grid_Ellipse, wx.ID_ANY, u"+", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer9.Add( self.PB_RxPlus, 0, wx.ALL, 5 )
		
		self.PB_RxMinus = wx.Button( self.Panel_Grid_Ellipse, wx.ID_ANY, u"-", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer9.Add( self.PB_RxMinus, 0, wx.ALL, 5 )
		
		self.m_staticText461 = wx.StaticText( self.Panel_Grid_Ellipse, wx.ID_ANY, u"Ry:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText461.Wrap( -1 )
		gSizer9.Add( self.m_staticText461, 0, wx.ALL, 5 )
		
		self.PB_RyPlus = wx.Button( self.Panel_Grid_Ellipse, wx.ID_ANY, u"+", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer9.Add( self.PB_RyPlus, 0, wx.ALL, 5 )
		
		self.PB_RyMinus = wx.Button( self.Panel_Grid_Ellipse, wx.ID_ANY, u"-", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer9.Add( self.PB_RyMinus, 0, wx.ALL, 5 )
		
		
		self.Panel_Grid_Ellipse.SetSizer( gSizer9 )
		self.Panel_Grid_Ellipse.Layout()
		gSizer9.Fit( self.Panel_Grid_Ellipse )
		bSizer9.Add( self.Panel_Grid_Ellipse, 1, wx.EXPAND |wx.ALL, 5 )
		
		self.Panel_plot2ROI = wx.Panel( self.Tab_ROI, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		bSizer9.Add( self.Panel_plot2ROI, 3, wx.EXPAND |wx.ALL, 5 )
		
		
		bSizer8.Add( bSizer9, 1, wx.EXPAND, 5 )
		
		
		self.Tab_ROI.SetSizer( bSizer8 )
		self.Tab_ROI.Layout()
		bSizer8.Fit( self.Tab_ROI )
		self.m_notebook3.AddPage( self.Tab_ROI, u"Select Roi", False )
		self.TAB_displacement = wx.Panel( self.m_notebook3, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		self.TAB_displacement.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
		
		bSizer10 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.Panel_plot1Disp = wx.Panel( self.TAB_displacement, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		bSizer10.Add( self.Panel_plot1Disp, 3, wx.EXPAND|wx.ALL, 5 )
		
		self.m_panel15 = wx.Panel( self.TAB_displacement, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		self.m_panel15.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
		
		bSizer11 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_panel151 = wx.Panel( self.m_panel15, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		self.m_panel151.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_BTNFACE ) )
		
		gSizer5 = wx.GridSizer( 0, 2, 0, 0 )
		
		self.m_staticText6 = wx.StaticText( self.m_panel151, wx.ID_ANY, u"Filter:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText6.Wrap( -1 )
		gSizer5.Add( self.m_staticText6, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		CB_filtersDispChoices = [ u"Gauss", u"Mean Filter" ]
		self.CB_filtersDisp = wx.Choice( self.m_panel151, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, CB_filtersDispChoices, 0 )
		self.CB_filtersDisp.SetSelection( 0 )
		gSizer5.Add( self.CB_filtersDisp, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText7 = wx.StaticText( self.m_panel151, wx.ID_ANY, u"Size:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText7.Wrap( -1 )
		gSizer5.Add( self.m_staticText7, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.ET_filterSizeDisp = wx.TextCtrl( self.m_panel151, wx.ID_ANY, u"51", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer5.Add( self.ET_filterSizeDisp, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText8 = wx.StaticText( self.m_panel151, wx.ID_ANY, u"Std(Gauss):", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText8.Wrap( -1 )
		gSizer5.Add( self.m_staticText8, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.ET_stdGausDisp = wx.TextCtrl( self.m_panel151, wx.ID_ANY, u"10", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer5.Add( self.ET_stdGausDisp, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText481 = wx.StaticText( self.m_panel151, wx.ID_ANY, u"Use Cluster Smooth:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText481.Wrap( -1 )
		gSizer5.Add( self.m_staticText481, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.CB_cluster = wx.CheckBox( self.m_panel151, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer5.Add( self.CB_cluster, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText49 = wx.StaticText( self.m_panel151, wx.ID_ANY, u"Number of Clusters:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText49.Wrap( -1 )
		gSizer5.Add( self.m_staticText49, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.ET_clusterNumber = wx.TextCtrl( self.m_panel151, wx.ID_ANY, u"3", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer5.Add( self.ET_clusterNumber, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText50 = wx.StaticText( self.m_panel151, wx.ID_ANY, u"Training Vector[1,2,3]:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText50.Wrap( -1 )
		gSizer5.Add( self.m_staticText50, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.ET_trainingVector = wx.TextCtrl( self.m_panel151, wx.ID_ANY, u"1", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer5.Add( self.ET_trainingVector, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText9 = wx.StaticText( self.m_panel151, wx.ID_ANY, u"Let Us Choose:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText9.Wrap( -1 )
		gSizer5.Add( self.m_staticText9, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.ET_usChooseDisp = wx.CheckBox( self.m_panel151, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer5.Add( self.ET_usChooseDisp, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText10 = wx.StaticText( self.m_panel151, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText10.Wrap( -1 )
		gSizer5.Add( self.m_staticText10, 0, wx.ALL, 5 )
		
		self.PB_filterDisp = wx.Button( self.m_panel151, wx.ID_ANY, u"Filter", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer5.Add( self.PB_filterDisp, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		self.m_panel151.SetSizer( gSizer5 )
		self.m_panel151.Layout()
		gSizer5.Fit( self.m_panel151 )
		bSizer11.Add( self.m_panel151, 2, wx.EXPAND |wx.ALL, 5 )
		
		self.m_panel16 = wx.Panel( self.m_panel15, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		self.m_panel16.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_BTNFACE ) )
		
		gSizer51 = wx.GridSizer( 0, 2, 0, 0 )
		
		self.m_staticText101 = wx.StaticText( self.m_panel16, wx.ID_ANY, u"Select Component:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText101.Wrap( -1 )
		gSizer51.Add( self.m_staticText101, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		CB_dispShowChoiceChoices = [ u"U", u"V", u"U+V" ]
		self.CB_dispShowChoice = wx.Choice( self.m_panel16, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, CB_dispShowChoiceChoices, 0 )
		self.CB_dispShowChoice.SetSelection( 0 )
		gSizer51.Add( self.CB_dispShowChoice, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText11 = wx.StaticText( self.m_panel16, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText11.Wrap( -1 )
		gSizer51.Add( self.m_staticText11, 0, wx.ALL, 5 )
		
		self.PB_visuDisp = wx.Button( self.m_panel16, wx.ID_ANY, u"Visualize", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer51.Add( self.PB_visuDisp, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		self.m_panel16.SetSizer( gSizer51 )
		self.m_panel16.Layout()
		gSizer51.Fit( self.m_panel16 )
		bSizer11.Add( self.m_panel16, 1, wx.EXPAND |wx.ALL, 5 )
		
		
		self.m_panel15.SetSizer( bSizer11 )
		self.m_panel15.Layout()
		bSizer11.Fit( self.m_panel15 )
		bSizer10.Add( self.m_panel15, 1, wx.EXPAND |wx.ALL, 5 )
		
		
		self.TAB_displacement.SetSizer( bSizer10 )
		self.TAB_displacement.Layout()
		bSizer10.Fit( self.TAB_displacement )
		self.m_notebook3.AddPage( self.TAB_displacement, u"Displacement ", True )
		self.TAB_strain = wx.Panel( self.m_notebook3, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		self.TAB_strain.SetBackgroundColour( wx.Colour( 247, 249, 248 ) )
		
		bSizer101 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.Panel_plot1Strain = wx.Panel( self.TAB_strain, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		bSizer101.Add( self.Panel_plot1Strain, 3, wx.EXPAND|wx.ALL, 5 )
		
		self.m_panel152 = wx.Panel( self.TAB_strain, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		self.m_panel152.SetBackgroundColour( wx.Colour( 232, 232, 242 ) )
		
		bSizer111 = wx.BoxSizer( wx.VERTICAL )
		
		self.RB_deformationType = wx.Panel( self.m_panel152, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		self.RB_deformationType.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_MENU ) )
		
		gSizer52 = wx.GridSizer( 0, 2, 0, 0 )
		
		self.m_staticText61 = wx.StaticText( self.RB_deformationType, wx.ID_ANY, u"Algorithm:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText61.Wrap( -1 )
		gSizer52.Add( self.m_staticText61, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		CB_filtersDisp1Choices = [ u"Direct Gradient", u"Strain Plane Fit" ]
		self.CB_filtersDisp1 = wx.Choice( self.RB_deformationType, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, CB_filtersDisp1Choices, 0 )
		self.CB_filtersDisp1.SetSelection( 0 )
		gSizer52.Add( self.CB_filtersDisp1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText71 = wx.StaticText( self.RB_deformationType, wx.ID_ANY, u"Size (Plane):", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText71.Wrap( -1 )
		gSizer52.Add( self.m_staticText71, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.ET_filterSizeStrain = wx.TextCtrl( self.RB_deformationType, wx.ID_ANY, u"21", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer52.Add( self.ET_filterSizeStrain, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText91 = wx.StaticText( self.RB_deformationType, wx.ID_ANY, u"Let us Choose:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText91.Wrap( -1 )
		gSizer52.Add( self.m_staticText91, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.ET_usChooseStrain = wx.CheckBox( self.RB_deformationType, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer52.Add( self.ET_usChooseStrain, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.CB_smoothStrain = wx.CheckBox( self.RB_deformationType, wx.ID_ANY, u"Smooth?", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.CB_smoothStrain.SetValue(True) 
		gSizer52.Add( self.CB_smoothStrain, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.CB_gaussianCorrection = wx.CheckBox( self.RB_deformationType, wx.ID_ANY, u"GP correction", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer52.Add( self.CB_gaussianCorrection, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText362 = wx.StaticText( self.RB_deformationType, wx.ID_ANY, u"Filter:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText362.Wrap( -1 )
		gSizer52.Add( self.m_staticText362, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		CB_filtersStrainChoices = [ u"Gauss", u"Median Filter", u"Mean Filter" ]
		self.CB_filtersStrain = wx.Choice( self.RB_deformationType, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, CB_filtersStrainChoices, 0 )
		self.CB_filtersStrain.SetSelection( 0 )
		gSizer52.Add( self.CB_filtersStrain, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText371 = wx.StaticText( self.RB_deformationType, wx.ID_ANY, u"Filter Value:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText371.Wrap( -1 )
		gSizer52.Add( self.m_staticText371, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.ET_filterValueStrain = wx.TextCtrl( self.RB_deformationType, wx.ID_ANY, u"5", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer52.Add( self.ET_filterValueStrain, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText351 = wx.StaticText( self.RB_deformationType, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText351.Wrap( -1 )
		gSizer52.Add( self.m_staticText351, 0, wx.ALL, 5 )
		
		self.PB_filterStrain = wx.Button( self.RB_deformationType, wx.ID_ANY, u"Calculate Strain", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer52.Add( self.PB_filterStrain, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		self.RB_deformationType.SetSizer( gSizer52 )
		self.RB_deformationType.Layout()
		gSizer52.Fit( self.RB_deformationType )
		bSizer111.Add( self.RB_deformationType, 2, wx.EXPAND |wx.ALL, 5 )
		
		self.m_panel161 = wx.Panel( self.m_panel152, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		self.m_panel161.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_MENU ) )
		
		gSizer511 = wx.GridSizer( 0, 2, 0, 0 )
		
		self.m_staticText1011 = wx.StaticText( self.m_panel161, wx.ID_ANY, u"Select Component:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText1011.Wrap( -1 )
		gSizer511.Add( self.m_staticText1011, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		CB_strainShowChoiceChoices = [ u"Exx", u"Eyy", u"Exy", u"ExxFit", u"EyyFit", u"ExyFit", u"Exx Uncertainty", u"Eyy Uncertainty", u"Exy Uncertainty" ]
		self.CB_strainShowChoice = wx.Choice( self.m_panel161, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, CB_strainShowChoiceChoices, 0 )
		self.CB_strainShowChoice.SetSelection( 0 )
		gSizer511.Add( self.CB_strainShowChoice, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText26 = wx.StaticText( self.m_panel161, wx.ID_ANY, u"Deformation Type:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText26.Wrap( -1 )
		gSizer511.Add( self.m_staticText26, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		m_radioBox1Choices = [ u"Lagrangian", u"Euclidian" ]
		self.m_radioBox1 = wx.RadioBox( self.m_panel161, wx.ID_ANY, u"Deformation Type", wx.DefaultPosition, wx.DefaultSize, m_radioBox1Choices, 1, wx.RA_SPECIFY_COLS )
		self.m_radioBox1.SetSelection( 1 )
		gSizer511.Add( self.m_radioBox1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText111 = wx.StaticText( self.m_panel161, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText111.Wrap( -1 )
		gSizer511.Add( self.m_staticText111, 0, wx.ALL, 5 )
		
		self.PB_visuStrain = wx.Button( self.m_panel161, wx.ID_ANY, u"Visualize", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer511.Add( self.PB_visuStrain, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		self.m_panel161.SetSizer( gSizer511 )
		self.m_panel161.Layout()
		gSizer511.Fit( self.m_panel161 )
		bSizer111.Add( self.m_panel161, 1, wx.EXPAND |wx.ALL, 5 )
		
		
		self.m_panel152.SetSizer( bSizer111 )
		self.m_panel152.Layout()
		bSizer111.Fit( self.m_panel152 )
		bSizer101.Add( self.m_panel152, 1, wx.EXPAND |wx.ALL, 5 )
		
		
		self.TAB_strain.SetSizer( bSizer101 )
		self.TAB_strain.Layout()
		bSizer101.Fit( self.TAB_strain )
		self.m_notebook3.AddPage( self.TAB_strain, u"Strain", False )
		self.TAB_result = wx.Panel( self.m_notebook3, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		self.TAB_result.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
		
		bSizer16 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_panel23 = wx.Panel( self.TAB_result, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		bSizer17 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.Panel_plot1Res = wx.Panel( self.m_panel23, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer17.Add( self.Panel_plot1Res, 1, wx.EXPAND |wx.ALL, 5 )
		
		self.Panel_plot2Res = wx.Panel( self.m_panel23, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer17.Add( self.Panel_plot2Res, 1, wx.EXPAND |wx.ALL, 5 )
		
		
		self.m_panel23.SetSizer( bSizer17 )
		self.m_panel23.Layout()
		bSizer17.Fit( self.m_panel23 )
		bSizer16.Add( self.m_panel23, 3, wx.EXPAND |wx.ALL, 5 )
		
		self.m_panel24 = wx.Panel( self.TAB_result, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		bSizer18 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_panel25 = wx.Panel( self.m_panel24, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SIMPLE_BORDER|wx.TAB_TRAVERSAL )
		self.m_panel25.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_MENU ) )
		
		gSizer11 = wx.GridSizer( 0, 2, 0, 0 )
		
		self.PB_virtualStrainGate = wx.Button( self.m_panel25, wx.ID_ANY, u"Insert Virtual Strain Gauge", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer11.Add( self.PB_virtualStrainGate, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.PB_lineDeformation = wx.Button( self.m_panel25, wx.ID_ANY, u"Calculate Line Deformation", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer11.Add( self.PB_lineDeformation, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText30 = wx.StaticText( self.m_panel25, wx.ID_ANY, u"Show Uncertainty?", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText30.Wrap( -1 )
		gSizer11.Add( self.m_staticText30, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.CB_uncertainty = wx.CheckBox( self.m_panel25, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer11.Add( self.CB_uncertainty, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText34 = wx.StaticText( self.m_panel25, wx.ID_ANY, u"Confidence:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText34.Wrap( -1 )
		gSizer11.Add( self.m_staticText34, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.ET_confidence = wx.TextCtrl( self.m_panel25, wx.ID_ANY, u"0.95", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer11.Add( self.ET_confidence, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText44 = wx.StaticText( self.m_panel25, wx.ID_ANY, u"Force Step/Vector [N]:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText44.Wrap( -1 )
		gSizer11.Add( self.m_staticText44, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.ET_forceStep = wx.TextCtrl( self.m_panel25, wx.ID_ANY, u"20", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer11.Add( self.ET_forceStep, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText381 = wx.StaticText( self.m_panel25, wx.ID_ANY, u"Transversal Area", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText381.Wrap( -1 )
		gSizer11.Add( self.m_staticText381, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.ET_areaResult = wx.TextCtrl( self.m_panel25, wx.ID_ANY, u"1", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer11.Add( self.ET_areaResult, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText45 = wx.StaticText( self.m_panel25, wx.ID_ANY, u"Images to Analyse:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText45.Wrap( -1 )
		gSizer11.Add( self.m_staticText45, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		CB_photoResultChoices = [ u"All" ]
		self.CB_photoResult = wx.Choice( self.m_panel25, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, CB_photoResultChoices, 0 )
		self.CB_photoResult.SetSelection( 0 )
		gSizer11.Add( self.CB_photoResult, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_staticText35 = wx.StaticText( self.m_panel25, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText35.Wrap( -1 )
		gSizer11.Add( self.m_staticText35, 0, wx.ALL, 5 )
		
		self.PB_SinglePointResult = wx.Button( self.m_panel25, wx.ID_ANY, u"Single Point", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer11.Add( self.PB_SinglePointResult, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.CB_resultsFit = wx.CheckBox( self.m_panel25, wx.ID_ANY, u"Fit Result", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer11.Add( self.CB_resultsFit, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.PB_fitAll = wx.Button( self.m_panel25, wx.ID_ANY, u"Calculate Fit for all points", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer11.Add( self.PB_fitAll, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
		self.m_panel25.SetSizer( gSizer11 )
		self.m_panel25.Layout()
		gSizer11.Fit( self.m_panel25 )
		bSizer18.Add( self.m_panel25, 2, wx.EXPAND |wx.ALL, 5 )
		
		self.Tree_results = wx.TreeCtrl( self.m_panel24, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TR_DEFAULT_STYLE )
		bSizer18.Add( self.Tree_results, 1, wx.ALL|wx.EXPAND, 5 )
		
		
		self.m_panel24.SetSizer( bSizer18 )
		self.m_panel24.Layout()
		bSizer18.Fit( self.m_panel24 )
		bSizer16.Add( self.m_panel24, 1, wx.EXPAND |wx.ALL, 5 )
		
		
		self.TAB_result.SetSizer( bSizer16 )
		self.TAB_result.Layout()
		bSizer16.Fit( self.TAB_result )
		self.m_notebook3.AddPage( self.TAB_result, u"Result Visualization", False )
		
		bSizer6.Add( self.m_notebook3, 1, wx.EXPAND |wx.ALL, 5 )
		
		
		self.m_panel2.SetSizer( bSizer6 )
		self.m_panel2.Layout()
		bSizer6.Fit( self.m_panel2 )
		self.m_splitter1.SplitVertically( self.m_panel1, self.m_panel2, 379 )
		bSizer1.Add( self.m_splitter1, 1, wx.EXPAND, 5 )
		
		
		self.SetSizer( bSizer1 )
		self.Layout()
		self.m_toolBar1 = self.CreateToolBar( wx.TB_DOCKABLE|wx.TB_HORIZONTAL|wx.SIMPLE_BORDER, wx.ID_ANY ) 
		self.m_toolBar1.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_SCROLLBAR ) )
		
		self.Toolbar_calculate = self.m_toolBar1.AddLabelTool( wx.ID_ANY, u"tool", wx.Bitmap( u"greenarrowicon.gif", wx.BITMAP_TYPE_ANY ), wx.NullBitmap, wx.ITEM_NORMAL, u"Calculate DIC", u"Calculate DIC", None ) 
		
		self.Toolbar_animate = self.m_toolBar1.AddLabelTool( wx.ID_ANY, u"tool", wx.Bitmap( u"16x16panel.png", wx.BITMAP_TYPE_ANY ), wx.NullBitmap, wx.ITEM_NORMAL, u"Animate Deformation", u"Animate Result", None ) 
		
		self.Toolbar_cplus = self.m_toolBar1.AddLabelTool( wx.ID_ANY, u"tool", wx.Bitmap( u"calculator.png", wx.BITMAP_TYPE_ANY ), wx.NullBitmap, wx.ITEM_NORMAL, u"C++ Calculation", u"C++ Calculation", None ) 
		
		self.m_toolBar1.Realize() 
		
		self.m_statusBar1 = self.CreateStatusBar( 2 )
		self.m_menubar1 = wx.MenuBar( 0 )
		self.File_menu = wx.Menu()
		self.Menu_TwoImages = wx.MenuItem( self.File_menu, wx.ID_ANY, u"Two images pair", wx.EmptyString, wx.ITEM_NORMAL )
		self.File_menu.AppendItem( self.Menu_TwoImages )
		
		self.MenuFolder = wx.MenuItem( self.File_menu, wx.ID_ANY, u"Folder", wx.EmptyString, wx.ITEM_NORMAL )
		self.File_menu.AppendItem( self.MenuFolder )
		
		self.File_menu.AppendSeparator()
		
		self.Menu_save = wx.MenuItem( self.File_menu, wx.ID_ANY, u"Save Results", wx.EmptyString, wx.ITEM_NORMAL )
		self.File_menu.AppendItem( self.Menu_save )
		
		self.Menu_saveProject = wx.MenuItem( self.File_menu, wx.ID_ANY, u"Save Project", wx.EmptyString, wx.ITEM_NORMAL )
		self.File_menu.AppendItem( self.Menu_saveProject )
		
		self.Menu_loadProject = wx.MenuItem( self.File_menu, wx.ID_ANY, u"Load Project", wx.EmptyString, wx.ITEM_NORMAL )
		self.File_menu.AppendItem( self.Menu_loadProject )
		
		self.File_menu.AppendSeparator()
		
		self.Menu_loadMask = wx.MenuItem( self.File_menu, wx.ID_ANY, u"Load Mask", wx.EmptyString, wx.ITEM_NORMAL )
		self.File_menu.AppendItem( self.Menu_loadMask )
		
		self.Menu_saveMask = wx.MenuItem( self.File_menu, wx.ID_ANY, u"Save Mask", wx.EmptyString, wx.ITEM_NORMAL )
		self.File_menu.AppendItem( self.Menu_saveMask )
		
		self.m_menubar1.Append( self.File_menu, u"File" ) 
		
		self.View_menu = wx.Menu()
		self.Menu_GridAnalysis = wx.MenuItem( self.View_menu, wx.ID_ANY, u"Parameter Grid Analysis", wx.EmptyString, wx.ITEM_NORMAL )
		self.View_menu.AppendItem( self.Menu_GridAnalysis )
		
		self.Menu_CompareResult = wx.MenuItem( self.View_menu, wx.ID_ANY, u"Compare Result", wx.EmptyString, wx.ITEM_NORMAL )
		self.View_menu.AppendItem( self.Menu_CompareResult )
		
		self.Menu_loadROI = wx.MenuItem( self.View_menu, wx.ID_ANY, u"Load ROI", wx.EmptyString, wx.ITEM_NORMAL )
		self.View_menu.AppendItem( self.Menu_loadROI )
		
		self.m_menubar1.Append( self.View_menu, u"Analysis" ) 
		
		self.About_menu = wx.Menu()
		self.Menu_aboutme = wx.MenuItem( self.About_menu, wx.ID_ANY, u"About the Creator", wx.EmptyString, wx.ITEM_NORMAL )
		self.About_menu.AppendItem( self.Menu_aboutme )
		
		self.m_menubar1.Append( self.About_menu, u"About" ) 
		
		self.SetMenuBar( self.m_menubar1 )
		
		
		self.Centre( wx.BOTH )
		
		# Connect Events
		self.PB_lineCalibrate.Bind( wx.EVT_BUTTON, self.PB_lineCalibrate_Callback )
		self.PB_showCalibrate.Bind( wx.EVT_BUTTON, self.PB_showCalibrate_Callback )
		self.Tree_photosMain.Bind( wx.EVT_TREE_SEL_CHANGED, self.Tree_photosMain_Callback )
		self.PB_addRec.Bind( wx.EVT_BUTTON, self.PB_addRec_Callback )
		self.PB_addElli.Bind( wx.EVT_BUTTON, self.PB_addElli_Callback )
		self.PB_removeRec.Bind( wx.EVT_BUTTON, self.PB_removeRec_Callback )
		self.PB_remElli.Bind( wx.EVT_BUTTON, self.PB_remElli_Callback )
		self.PB_clearAll.Bind( wx.EVT_BUTTON, self.PB_clearAll_Callback )
		self.PB_Xplus.Bind( wx.EVT_BUTTON, self.PB_Xplus_Callback )
		self.PB_Xminus.Bind( wx.EVT_BUTTON, self.PB_Xminus_Callback )
		self.PB_Yplus.Bind( wx.EVT_BUTTON, self.PB_Yplus_Callback )
		self.PB_Yminus.Bind( wx.EVT_BUTTON, self.PB_Yminus_Callback )
		self.PB_RxPlus.Bind( wx.EVT_BUTTON, self.PB_RxPlus_Callback )
		self.PB_RxMinus.Bind( wx.EVT_BUTTON, self.PB_RxMinus_Callback )
		self.PB_RyPlus.Bind( wx.EVT_BUTTON, self.PB_RyPlus_Callback )
		self.PB_RyMinus.Bind( wx.EVT_BUTTON, self.PB_RyMinus_Callback )
		self.PB_filterDisp.Bind( wx.EVT_BUTTON, self.PB_filterDisp_Callback )
		self.PB_visuDisp.Bind( wx.EVT_BUTTON, self.PB_visuDisp_Callback )
		self.CB_gaussianCorrection.Bind( wx.EVT_CHECKBOX, self.CB_gaussianCorrection_Callback )
		self.PB_filterStrain.Bind( wx.EVT_BUTTON, self.PB_filterStrain_Callback )
		self.PB_visuStrain.Bind( wx.EVT_BUTTON, self.PB_visuStrain_Callback )
		self.PB_virtualStrainGate.Bind( wx.EVT_BUTTON, self.PB_virtualStrainGate_Callback )
		self.PB_lineDeformation.Bind( wx.EVT_BUTTON, self.PB_lineDeformation_Callback )
		self.PB_SinglePointResult.Bind( wx.EVT_BUTTON, self.PB_SinglePointResult_Callback )
		self.PB_fitAll.Bind( wx.EVT_BUTTON, self.PB_fitAll_Callback )
		self.Bind( wx.EVT_TOOL, self.Toolbar_calculate_Callback, id = self.Toolbar_calculate.GetId() )
		self.Bind( wx.EVT_TOOL, self.Toolbar_animate_Callback, id = self.Toolbar_animate.GetId() )
		self.Bind( wx.EVT_TOOL, self.Toolbar_cplus_Callback, id = self.Toolbar_cplus.GetId() )
		self.Bind( wx.EVT_MENU, self.Menu_TwoImages_Callback, id = self.Menu_TwoImages.GetId() )
		self.Bind( wx.EVT_MENU, self.MenuFolder_Callback, id = self.MenuFolder.GetId() )
		self.Bind( wx.EVT_MENU, self.Menu_saveProject_Callback, id = self.Menu_saveProject.GetId() )
		self.Bind( wx.EVT_MENU, self.Menu_loadProject_Callback, id = self.Menu_loadProject.GetId() )
		self.Bind( wx.EVT_MENU, self.Menu_loadMask_Callbak, id = self.Menu_loadMask.GetId() )
		self.Bind( wx.EVT_MENU, self.Menu_saveMask_Callback, id = self.Menu_saveMask.GetId() )
		self.Bind( wx.EVT_MENU, self.Menu_GridAnalysis_Callback, id = self.Menu_GridAnalysis.GetId() )
		self.Bind( wx.EVT_MENU, self.Menu_CompareResult_Callback, id = self.Menu_CompareResult.GetId() )
		self.Bind( wx.EVT_MENU, self.Menu_loadROI_Callback, id = self.Menu_loadROI.GetId() )
	
	def __del__( self ):
		pass
	
	
	# Virtual event handlers, overide them in your derived class
	def PB_lineCalibrate_Callback( self, event ):
		event.Skip()
	
	def PB_showCalibrate_Callback( self, event ):
		event.Skip()
	
	def Tree_photosMain_Callback( self, event ):
		event.Skip()
	
	def PB_addRec_Callback( self, event ):
		event.Skip()
	
	def PB_addElli_Callback( self, event ):
		event.Skip()
	
	def PB_removeRec_Callback( self, event ):
		event.Skip()
	
	def PB_remElli_Callback( self, event ):
		event.Skip()
	
	def PB_clearAll_Callback( self, event ):
		event.Skip()
	
	def PB_Xplus_Callback( self, event ):
		event.Skip()
	
	def PB_Xminus_Callback( self, event ):
		event.Skip()
	
	def PB_Yplus_Callback( self, event ):
		event.Skip()
	
	def PB_Yminus_Callback( self, event ):
		event.Skip()
	
	def PB_RxPlus_Callback( self, event ):
		event.Skip()
	
	def PB_RxMinus_Callback( self, event ):
		event.Skip()
	
	def PB_RyPlus_Callback( self, event ):
		event.Skip()
	
	def PB_RyMinus_Callback( self, event ):
		event.Skip()
	
	def PB_filterDisp_Callback( self, event ):
		event.Skip()
	
	def PB_visuDisp_Callback( self, event ):
		event.Skip()
	
	def CB_gaussianCorrection_Callback( self, event ):
		event.Skip()
	
	def PB_filterStrain_Callback( self, event ):
		event.Skip()
	
	def PB_visuStrain_Callback( self, event ):
		event.Skip()
	
	def PB_virtualStrainGate_Callback( self, event ):
		event.Skip()
	
	def PB_lineDeformation_Callback( self, event ):
		event.Skip()
	
	def PB_SinglePointResult_Callback( self, event ):
		event.Skip()
	
	def PB_fitAll_Callback( self, event ):
		event.Skip()
	
	def Toolbar_calculate_Callback( self, event ):
		event.Skip()
	
	def Toolbar_animate_Callback( self, event ):
		event.Skip()
	
	def Toolbar_cplus_Callback( self, event ):
		event.Skip()
	
	def Menu_TwoImages_Callback( self, event ):
		event.Skip()
	
	def MenuFolder_Callback( self, event ):
		event.Skip()
	
	def Menu_saveProject_Callback( self, event ):
		event.Skip()
	
	def Menu_loadProject_Callback( self, event ):
		event.Skip()
	
	def Menu_loadMask_Callbak( self, event ):
		event.Skip()
	
	def Menu_saveMask_Callback( self, event ):
		event.Skip()
	
	def Menu_GridAnalysis_Callback( self, event ):
		event.Skip()
	
	def Menu_CompareResult_Callback( self, event ):
		event.Skip()
	
	def Menu_loadROI_Callback( self, event ):
		event.Skip()
	
	def m_splitter1OnIdle( self, event ):
		self.m_splitter1.SetSashPosition( 379 )
		self.m_splitter1.Unbind( wx.EVT_IDLE )
	
	def m_splitter2OnIdle( self, event ):
		self.m_splitter2.SetSashPosition( 480 )
		self.m_splitter2.Unbind( wx.EVT_IDLE )
	

