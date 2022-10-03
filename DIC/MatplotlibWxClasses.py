import matplotlib
matplotlib.use('WXAgg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import math
import wx

########################## 2D #####################
class CanvasPanel2D(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent,wx.EXPAND |wx.ALL)

        self.parent = parent
        self.figure = Figure()
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.axes = self.figure.add_subplot(311)
        self.axes2 = self.figure.add_subplot(312)
        self.axes3 = self.figure.add_subplot(313)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.EXPAND |wx.ALL)#wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

        # Handle size
        self._resizeflag = False

        self.Bind(wx.EVT_IDLE, self._onIdle)
        self.Bind(wx.EVT_SIZE, self._onSize)

        self._SetSize()

    def draw(self,x,y,Title,Xlabel,Ylabel,Legends,Override):


        # Labels
        if Override == 1:
            self.axes.cla() # Clean axes before plotting
            self.axes2.cla()
            self.axes3.cla()
        self.axes.set_title(Title)
        self.axes.set_xlabel(Xlabel)
        self.axes.set_ylabel(Ylabel)

        self.axes.plot(x, y[0])
        self.axes.legend(Legends[0],loc=0)

        self.axes2.plot(x, y[1])
        self.axes2.legend(Legends[1],loc=0)

        self.axes3.plot(x, y[2])
        self.axes3.legend(Legends[2],loc=0)

        self.canvas.draw()


    def _onSize(self, event):
        self._resizeflag = True


    def _onIdle(self, evt):
        if True:#self._resizeflag:
            self._resizeflag = False
            self._SetSize()


    def _SetSize(self):
        pixels = tuple(self.parent.GetClientSize())
        self.SetSize(pixels)
        self.canvas.SetSize(pixels)
        self.figure.set_size_inches(float(pixels[0]) / self.figure.get_dpi(),
                                    float(pixels[1]) / self.figure.get_dpi())

################################ 2D LOG ############
class CanvasPanel2D_log(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent,wx.EXPAND |wx.ALL)

        self.parent = parent
        self.figure = Figure()
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.axes = self.figure.add_subplot(111)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.EXPAND |wx.ALL)#wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

        # Handle size
        self._resizeflag = False

        self.Bind(wx.EVT_IDLE, self._onIdle)
        self.Bind(wx.EVT_SIZE, self._onSize)

        self._SetSize()

    def draw(self,x,y,Title,Xlabel,Ylabel):


        # Labels
        self.axes.cla() # Clean axes before plotting
        self.axes.set_title(Title)
        self.axes.set_xlabel(Xlabel)
        self.axes.set_ylabel(Ylabel)

        self.axes.semilogx(x, y)
        self.canvas.draw()


    def _onSize(self, event):
        self._resizeflag = True


    def _onIdle(self, evt):
        if True:#self._resizeflag:
            self._resizeflag = False
            self._SetSize()


    def _SetSize(self):
        pixels = tuple(self.parent.GetClientSize())
        self.SetSize(pixels)
        self.canvas.SetSize(pixels)
        self.figure.set_size_inches(float(pixels[0]) / self.figure.get_dpi(),
                                    float(pixels[1]) / self.figure.get_dpi())



# 3D
class CanvasPanel3D(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent,wx.EXPAND |wx.ALL)

        self.parent = parent
        self.figure = Figure()
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.axes = self.figure.add_subplot(111, projection='3d')
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        #Set up Matplotlib Toolbar
        #self.chart_toolbar = NavigationToolbar2Wx(self.canvas)
        #tw,th = self.chart_toolbar.GetSizeTuple()
        #fw,fh = self.canvas.GetSizeTuple()
        #self.chart_toolbar.SetSize(wx.Size(fw, th))
        #self.chart_toolbar.Realize()


        self._resizeflag = False

        self.Bind(wx.EVT_IDLE, self._onIdle)
        self.Bind(wx.EVT_SIZE, self._onSize)

        # Add figure and toolbar
        self.sizer.Add(self.canvas, 1, wx.EXPAND |wx.ALL)
        #self.sizer.Add(self.chart_toolbar, 1, flag=wx.ALIGN_CENTER, border=5) #Toolbar

        self.SetSizer(self.sizer)
        self.Fit()
        self._SetSize()


    def draw(self,x,y,z,Title):

        # Set labels
        self.axes.cla() # Clean axes before plotting
        self.axes.set_title(Title)
        self.axes.set_xlabel('X')
        self.axes.set_ylabel('Y')
        self.axes.set_zlabel('Z')
        self.axes.plot(x, y, z, '*')
        self._SetSize()
        self.canvas.draw()


    def _onSize(self, event):
        self._resizeflag = True


    def _onIdle(self, evt):
        if True:#self._resizeflag:
            self._resizeflag = False
            self._SetSize()


    def _SetSize(self):
        pixels = tuple(self.parent.GetClientSize())
        self.SetSize(pixels)
        self.canvas.SetSize(pixels)
        self.figure.set_size_inches(float(pixels[0]) / self.figure.get_dpi(),
                                    float(pixels[1]) / self.figure.get_dpi())


######### Image plot #########################
class CanvasPanel2D_Image(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent,wx.EXPAND |wx.ALL)

        self.parent = parent
        self.figure = Figure()
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.axes = self.figure.add_subplot(111)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.EXPAND |wx.ALL)#wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

        # Handle size
        self._resizeflag = False

        self.Bind(wx.EVT_IDLE, self._onIdle)
        self.Bind(wx.EVT_SIZE, self._onSize)

        self._SetSize()

    def draw(self,img,Title,Xlabel,Ylabel):


        # Labels

        self.axes.cla() # Clean axes before plotting
        self.axes.set_title(Title)
        self.axes.set_xlabel(Xlabel)
        self.axes.set_ylabel(Ylabel)

        self.axes.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        self.canvas.draw()


    def _onSize(self, event):
        self._resizeflag = True


    def _onIdle(self, evt):
        if True:#self._resizeflag:
            self._resizeflag = False
            self._SetSize()


    def _SetSize(self):
        pixels = tuple(self.parent.GetClientSize())
        self.SetSize(pixels)
        self.canvas.SetSize(pixels)
        self.figure.set_size_inches(float(pixels[0]) / self.figure.get_dpi(),
                                    float(pixels[1]) / self.figure.get_dpi())

######### Contourn plus image plot #########################
class CanvasPanel3D_Image(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent,wx.EXPAND |wx.ALL)

        self.parent = parent
        self.figure = Figure()
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.axes = self.figure.add_subplot(111)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1,wx.EXPAND |wx.ALL)#wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

        # Handle size
        self._resizeflag = False

        self.Bind(wx.EVT_IDLE, self._onIdle)
        self.Bind(wx.EVT_SIZE, self._onSize)

        self._SetSize()

    def draw(self,img,x,y,z,Title,Xlabel,Ylabel,Zmin = None,Zmax = None):


        # Labels
        self.axes.cla() # Clean axes before plotting

        self.axes.set_title(Title)
        self.axes.set_xlabel(Xlabel)
        self.axes.set_ylabel(Ylabel)

        self.axes.imshow(img, cmap = 'gray', interpolation = 'bicubic')

        if Zmin is None:
            self.p = self.axes.pcolor(x, y, z,alpha=0.4, cmap='jet')#, vmin=z_min, vmax=z_max) jet magma inferno gist_rainbow
        else:
            self.p = self.axes.pcolor(x, y, z, alpha=0.4, cmap='jet', vmin=Zmin, vmax=Zmax)
                            #pcolorfast
        if len(self.figure.axes) <= 1:
            self.colorBar =  self.figure.colorbar(self.p, ax= self.axes)
        else:
            self.colorBar.remove()
            self.colorBar = self.figure.colorbar(self.p, ax=self.axes)


        self.canvas.draw()


    def _onSize(self, event):
        self._resizeflag = True


    def _onIdle(self, evt):
        if True:#self._resizeflag:
            self._resizeflag = False
            self._SetSize()


    def _SetSize(self):
        pixels = tuple(self.parent.GetClientSize())
        self.SetSize(pixels)
        self.canvas.SetSize(pixels)
        self.figure.set_size_inches(float(pixels[0]) / self.figure.get_dpi(),
                                    float(pixels[1]) / self.figure.get_dpi())