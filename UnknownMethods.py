import wx #check

import wx.xrc #checar necessidade
import wx.grid #checar necessidade
class UnkownMethodsMixin:
    def Toolbar_Calculate_Callback(self, e):

        self.StatusBar.SetStatusText('Done Calculation.')

        # Mesage box
        dlg = wx.MessageDialog(self, 'Done Calculation (Yeah!)', 'Finish',  # Second argument is title
                               wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

    ################## Context Menu Callbacks ###################
    def Tree_control_Callback(self, e):
        Sel_Index = self.Tree_control.GetFocusedItem()
        Sel_Index = self.Tree_control.GetItemText(Sel_Index)

        ### Launcher creates wxMenu. ###
        menu = wx.Menu()
        menu.Append(0, 'Displacement constrain')
        menu.Append(1, 'Impedance Constrain')
        wx.EVT_MENU(menu, 0, self.Context_menu_B_fixed)

        ### 5. Launcher displays menu with call to PopupMenu, invoked on the source component, passing event's GetPoint. ###
        self.PopupMenu(menu, e.GetPoint())
        menu.Destroy()  # destroy to avoid mem leak

    def Context_menu_B_fixed(self, e):
        a = 1
        print(5)