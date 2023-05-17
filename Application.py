import wx #check
from GUI_DIC import CalcFrame

if __name__ == '__main__':
    app = wx.App(False)
    frame = CalcFrame(None)
    frame.Show(True)
    app.MainLoop()