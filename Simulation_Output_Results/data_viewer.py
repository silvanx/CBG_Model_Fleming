from pathlib import Path
import sys, os
import matplotlib
from PyQt5 import QtWidgets, QtGui, QtCore
import plot_utils as u
import pandas as pd

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,\
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

matplotlib.use("Qt5Agg")


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class DetailedViewWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(DetailedViewWindow, self).__init__(parent)

        self.setWindowTitle("Detailed View Window")
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.canvas.axes.remove()
        first_ax = self.canvas.fig.add_subplot(3, 1, 1)
        self.axs = [
            first_ax,
            self.canvas.fig.add_subplot(3, 1, 2, sharex=first_ax),
            self.canvas.fig.add_subplot(3, 1, 3, sharex=first_ax)
        ]
        toolbar = NavigationToolbar(self.canvas, self)

        main_widget = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout()
        central_layout.addWidget(toolbar)
        central_layout.addWidget(self.canvas)
        main_widget.setLayout(central_layout)

        self.setCentralWidget(main_widget)

    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        if ev.key() == QtCore.Qt.Key.Key_Escape:
            self.close()


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self) -> None:
        super().__init__()

        hover_label_style = (
            "QLabel::hover"
            "{"
            "color : #8e8e8e"
            "}")

        self.fitness_dir = (
            "Simulation_Output_Results/PI_grid_search_12"
            )
        self.results_dir = (
            "Simulation_Output_Results/ift"
            )
        self.file_list = []
        self.current_file = None
        self.last_arrows = None
        self.zlim_exponent_high = 1
        self.zlim_exponent_low = -9
        self.df = pd.read_excel(
            "Simulation_Output_Results/output.xlsx"
            ).dropna(subset=['Simulation dir'])

        self.parameter_plot = MplCanvas(self, width=12, height=7, dpi=100)
        parameter_toolbar = NavigationToolbar(self.parameter_plot, self)

        self.description = QtWidgets.QLabel()
        self.description.setText("Description")
        self.description.setFont(QtGui.QFont("Roboto", 15))
        self.description.setFixedHeight(120)

        # self.zlim_slider_top = QtWidgets.QSlider(QtCore.Qt.Vertical)
        # self.zlim_slider_top.setMinimum(-9)
        # self.zlim_slider_top.setMaximum(1)
        # self.zlim_slider_top.setValue(self.zlim_slider_top.maximum())
        # self.zlim_slider_top.setTickPosition(QtWidgets.QSlider.TicksRight)
        # self.zlim_slider_top.setTickInterval(1)
        self.zlim_slider = RangeSlider(QtCore.Qt.Vertical)
        self.zlim_slider.setMinimum(-9)
        self.zlim_slider.setMaximum(1)
        self.zlim_slider.setHigh(self.zlim_slider.maximum())
        self.zlim_slider.setLow(self.zlim_slider.minimum())
        self.zlim_slider.setTickPosition(QtWidgets.QSlider.TicksRight)
        self.zlim_slider.setTickInterval(1)
        self.zlim_slider.sliderMoved.connect(self.update_plot_bounds)

        layout_plot = QtWidgets.QVBoxLayout()
        layout_plot.addWidget(parameter_toolbar)
        layout_plot.addWidget(self.parameter_plot)
        self.parameter_plot.setMinimumWidth(200)
        layout_plot_outer = QtWidgets.QHBoxLayout()
        layout_plot_outer.addLayout(layout_plot)
        layout_plot_outer.addWidget(self.zlim_slider)

        recalculate_fitness_button = QtWidgets.QPushButton()
        recalculate_fitness_button.setText("Reload fitness (R)")
        recalculate_fitness_button.clicked.connect(
            self.recalculate_fitness
        )

        open_detailed_view_button = QtWidgets.QPushButton()
        open_detailed_view_button.setText("Open detailed view (D)")
        self.detailed_view_window = DetailedViewWindow(self)
        open_detailed_view_button.clicked.connect(
            self.open_detailed_view_window)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(recalculate_fitness_button)
        buttons.addWidget(open_detailed_view_button)

        # Left side
        layout_left = QtWidgets.QVBoxLayout()

        layout_left.addWidget(self.description)
        layout_left.addLayout(layout_plot_outer)
        layout_left.addLayout(buttons)

        # Right side
        layout_right = QtWidgets.QVBoxLayout()

        self.fitness_directory_label = QtWidgets.QLabel()
        self.fitness_directory_label.setStyleSheet(hover_label_style)
        if self.results_dir is not None:
            self.fitness_directory_label.setText(self.fitness_dir)
        else:
            self.fitness_directory_label.setText('None')

        self.directory_label = QtWidgets.QLabel()
        self.directory_label.setStyleSheet(hover_label_style)
        if self.results_dir is not None:
            self.directory_label.setText(self.results_dir)
        else:
            self.directory_label.setText('None')

        self.file_list_widget = QtWidgets.QListWidget()
        self.file_list_widget.itemSelectionChanged.connect(
            self.plot_clicked_dir_arrows)
        self.file_list_widget.installEventFilter(self)

        self.directory_label.mousePressEvent = self.change_file_dir
        self.fitness_directory_label.mousePressEvent = self.change_fitness_dir

        layout_right.addWidget(QtWidgets.QLabel('Fitness directory:'))
        layout_right.addWidget(self.fitness_directory_label)
        layout_right.addWidget(QtWidgets.QLabel('Result directory:'))
        layout_right.addWidget(self.directory_label)
        layout_right.addWidget(self.file_list_widget)

        main_splitter = QtWidgets.QSplitter()

        widget_left = QtWidgets.QWidget()
        widget_left.setLayout(layout_left)

        widget_right = QtWidgets.QWidget()
        widget_right.setLayout(layout_right)

        main_splitter.addWidget(widget_left)
        main_splitter.addWidget(widget_right)

        layout_main = QtWidgets.QHBoxLayout()
        layout_main.addWidget(main_splitter)

        widget = QtWidgets.QWidget()
        # Main layout
        widget.setLayout(layout_main)
        self.setCentralWidget(widget)

        self.installEventFilter(self)

        u.plot_pi_fitness_function(Path(self.fitness_dir),
                                   self.parameter_plot.fig,
                                   self.parameter_plot.axes)
        self.last_lambda = 1
        self.file_list = self.populate_file_list()
        self.refresh_file_list_display()

        self.setWindowTitle("Closed-loop parameters")
        self.show()

    def eventFilter(self,
                    object: QtCore.QObject,
                    event: QtCore.QEvent) -> bool:
        if event.type() != QtCore.QEvent.Type.KeyPress:
            return False

        if event.key() == QtCore.Qt.Key.Key_D:
            self.open_detailed_view_window()
            return True
        elif event.key() == QtCore.Qt.Key.Key_R:
            self.recalculate_fitness()
            return True

        return False

    def open_detailed_view_window(self) -> None:
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        for ax in self.detailed_view_window.axs:
            ax.cla()
        if self.current_file is not None:
            filename = (Path(self.results_dir) /
                        self.file_list[self.current_file])
            u.plot_ift_signals(
                filename,
                self.detailed_view_window.axs
            )
        self.detailed_view_window.canvas.draw()
        self.detailed_view_window.canvas.flush_events()
        self.detailed_view_window.show()
        QtWidgets.QApplication.restoreOverrideCursor()

    def recalculate_fitness(self) -> None:
        fitness_file = Path(self.fitness_dir) / 'output.npy'
        fitness_file.unlink()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            cax = self.parameter_plot.fig.axes[-1]
        except IndexError:
            cax = None
        self.parameter_plot.axes.cla()
        u.plot_pi_fitness_function(Path(self.fitness_dir),
                                   self.parameter_plot.fig,
                                   self.parameter_plot.axes,
                                   cax=cax,
                                   zlim_exponent_high=self.zlim_exponent_high)
        self.parameter_plot.draw()
        QtWidgets.QApplication.restoreOverrideCursor()

    def change_file_dir(self, ev):
        newdir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            '',
            self.results_dir
            )
        if newdir:
            self.results_dir = newdir
            self.directory_label.setText(newdir)
            self.current_file = None
            self.file_list = self.populate_file_list()
            self.refresh_file_list_display()

    def update_plot_bounds(self):
        self.zlim_exponent_high = self.zlim_slider.high()
        self.zlim_exponent_low = self.zlim_slider.low()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            cax = self.parameter_plot.fig.axes[-1]
        except IndexError:
            cax = None
        self.parameter_plot.axes.cla()
        u.plot_pi_fitness_function(Path(self.fitness_dir),
                                   self.parameter_plot.fig,
                                   self.parameter_plot.axes,
                                   cax=cax,
                                   zlim_exponent_high=self.zlim_exponent_high,
                                   zlim_exponent_low=self.zlim_exponent_low)
        if self.last_arrows is not None:
            for arrow in self.last_arrows:
                self.parameter_plot.axes.add_patch(arrow)
        self.parameter_plot.draw()
        QtWidgets.QApplication.restoreOverrideCursor()

    def change_fitness_dir(self, ev):
        newdir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            '',
            self.fitness_dir
            )
        if newdir:
            self.fitness_dir = newdir
            self.fitness_directory_label.setText(newdir)
            self.current_file = None
        try:
            cax = self.parameter_plot.fig.axes[-1]
        except IndexError:
            cax = None
        self.parameter_plot.axes.cla()
        u.plot_pi_fitness_function(Path(self.fitness_dir),
                                   self.parameter_plot.fig,
                                   self.parameter_plot.axes,
                                   cax=cax)
        self.parameter_plot.draw()

    def populate_file_list(self):
        dir = Path(self.results_dir)
        file_list = [file.name for file in dir.iterdir() if file.is_dir()]
        return file_list

    def refresh_file_list_display(self):
        self.file_list_widget.clearSelection()
        self.file_list_widget.clear()
        for file in self.file_list:
            currentItem = QtWidgets.QListWidgetItem(self.file_list_widget)
            currentItem.setText(file)

    def plot_clicked_dir_arrows(self):
        if len(self.file_list_widget.selectedItems()) > 0:
            if self.last_arrows is not None:
                for a in self.last_arrows:
                    a.remove()
                self.last_arrows = None
            item = self.file_list_widget.selectedItems()[0]
            text = item.text()
            self.current_file = self.file_list.index(text)
            f = Path(self.results_dir) / text

            row = self.df[self.df["Simulation dir"].str.contains(text)]
            outfiles = list(f.glob("*.out"))
            if len(outfiles) == 1:
                config = u.read_config_from_output_file(outfiles[0])
                description = (
                    f"{f.stem}\t"
                    f"{config['RunTime']} ms\t"
                    f"controller: {config['Controller']} "
                    f"({config['stage_length']} s)\n"
                    f"Kp init: {config['kp']}, "
                    f"Ti init: {config['ti']}\n"
                    f"gamma: {config['gamma']}, "
                    f"lambda: {config['lam']}\n"
                    f"min_kp,min_ti: {config['min_kp'], config['min_ti']}"
                    )
                self.description.setText(description) 
            elif not row.empty:
                description = (
                    f"ID: {row.iloc[0]['Simulation number']}\t"
                    f"{row.iloc[0]['Sim duration [ms]']} ms\t"
                    f"controller: {row.iloc[0]['controller']} "
                    f"({row.iloc[0]['IFT experiment length [s]']} s)\n"
                    f"Kp init: {row.iloc[0]['Kp']}, "
                    f"Ti init: {row.iloc[0]['Ti']}\n"
                    f"gamma: {row.iloc[0]['gamma']}, "
                    f"lambda: {row.iloc[0]['lambda']}\n"
                    f"min_kp,min_ti: {row.iloc[0]['min_kp,min_ti']}"
                    )
                self.description.setText(description)
                if self.last_lambda != row.iloc[0]['lambda']:
                    try:
                        cax = self.parameter_plot.fig.axes[-1]
                    except IndexError:
                        cax = None
                    self.parameter_plot.axes.cla()
                    u.plot_pi_fitness_function(Path(self.fitness_dir),
                                               self.parameter_plot.fig,
                                               self.parameter_plot.axes,
                                               cax=cax,
                                               lam=row.iloc[0]['lambda'],
                                               zlim_exponent_high=self.zlim_exponent_high,
                                               zlim_exponent_low=self.zlim_exponent_low)
                    self.parameter_plot.draw()
                    self.last_lambda = row.iloc[0]['lambda']
            else:
                self.description.setText("No description found in database")

            _, _, _, _, _, params, _ = u.read_ift_results(f)
            self.last_arrows = u.add_arrows_to_plot(
                self.parameter_plot.axes,
                params
                )
            self.parameter_plot.draw()


class RangeSlider(QtWidgets.QSlider):
    sliderMoved = QtCore.pyqtSignal(int, int)

    """ A slider for ranges.

        This class provides a dual-slider for ranges, where there is a defined
        maximum and minimum, as is a normal slider, but instead of having a
        single slider value, there are 2 slider values.

        This class emits the same signals as the QSlider base class, with the 
        exception of valueChanged
    """
    def __init__(self, *args):
        super(RangeSlider, self).__init__(*args)

        self._low = self.minimum()
        self._high = self.maximum()

        self.pressed_control = QtWidgets.QStyle.SC_None
        self.tick_interval = 0
        self.tick_position = QtWidgets.QSlider.NoTicks
        self.hover_control = QtWidgets.QStyle.SC_None
        self.click_offset = 0

        # 0 for the low, 1 for the high, -1 for both
        self.active_slider = 0

    def low(self):
        return self._low

    def setLow(self, low:int):
        self._low = low
        self.update()

    def high(self):
        return self._high

    def setHigh(self, high):
        self._high = high
        self.update()

    def paintEvent(self, event):
        # based on http://qt.gitorious.org/qt/qt/blobs/master/src/gui/widgets/qslider.cpp

        painter = QtGui.QPainter(self)
        style = QtWidgets.QApplication.style()

        # draw groove
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        opt.siderValue = 0
        opt.sliderPosition = 0
        opt.subControls = QtWidgets.QStyle.SC_SliderGroove
        if self.tickPosition() != self.NoTicks:
            opt.subControls |= QtWidgets.QStyle.SC_SliderTickmarks
        style.drawComplexControl(QtWidgets.QStyle.CC_Slider, opt, painter, self)
        groove = style.subControlRect(QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderGroove, self)

        # drawSpan
        #opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        opt.subControls = QtWidgets.QStyle.SC_SliderGroove
        #if self.tickPosition() != self.NoTicks:
        #    opt.subControls |= QtWidgets.QStyle.SC_SliderTickmarks
        opt.siderValue = 0
        #print(self._low)
        opt.sliderPosition = self._low
        low_rect = style.subControlRect(QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderHandle, self)
        opt.sliderPosition = self._high
        high_rect = style.subControlRect(QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderHandle, self)

        #print(low_rect, high_rect)
        low_pos = self.__pick(low_rect.center())
        high_pos = self.__pick(high_rect.center())

        min_pos = min(low_pos, high_pos)
        max_pos = max(low_pos, high_pos)

        c = QtCore.QRect(low_rect.center(), high_rect.center()).center()
        #print(min_pos, max_pos, c)
        if opt.orientation == QtCore.Qt.Horizontal:
            span_rect = QtCore.QRect(QtCore.QPoint(min_pos, c.y()-2), QtCore.QPoint(max_pos, c.y()+1))
        else:
            span_rect = QtCore.QRect(QtCore.QPoint(c.x()-2, min_pos), QtCore.QPoint(c.x()+1, max_pos))

        #self.initStyleOption(opt)
        #print(groove.x(), groove.y(), groove.width(), groove.height())
        if opt.orientation == QtCore.Qt.Horizontal: groove.adjust(0, 0, -1, 0)
        else: groove.adjust(0, 0, 0, -1)

        if True: #self.isEnabled():
            highlight = self.palette().color(QtGui.QPalette.Highlight)
            painter.setBrush(QtGui.QBrush(highlight))
            painter.setPen(QtGui.QPen(highlight, 0))
            #painter.setPen(QtGui.QPen(self.palette().color(QtGui.QPalette.Dark), 0))
            '''
            if opt.orientation == QtCore.Qt.Horizontal:
                self.setupPainter(painter, opt.orientation, groove.center().x(), groove.top(), groove.center().x(), groove.bottom())
            else:
                self.setupPainter(painter, opt.orientation, groove.left(), groove.center().y(), groove.right(), groove.center().y())
            '''
            #spanRect = 
            painter.drawRect(span_rect.intersected(groove))
            #painter.drawRect(groove)

        for i, value in enumerate([self._low, self._high]):
            opt = QtWidgets.QStyleOptionSlider()
            self.initStyleOption(opt)

            # Only draw the groove for the first slider so it doesn't get drawn
            # on top of the existing ones every time
            if i == 0:
                opt.subControls = QtWidgets.QStyle.SC_SliderHandle# | QtWidgets.QStyle.SC_SliderGroove
            else:
                opt.subControls = QtWidgets.QStyle.SC_SliderHandle

            if self.tickPosition() != self.NoTicks:
                opt.subControls |= QtWidgets.QStyle.SC_SliderTickmarks

            if self.pressed_control:
                opt.activeSubControls = self.pressed_control
            else:
                opt.activeSubControls = self.hover_control

            opt.sliderPosition = value
            opt.sliderValue = value                                  
            style.drawComplexControl(QtWidgets.QStyle.CC_Slider, opt, painter, self)

    def mousePressEvent(self, event):
        event.accept()

        style = QtWidgets.QApplication.style()
        button = event.button()

        # In a normal slider control, when the user clicks on a point in the 
        # slider's total range, but not on the slider part of the control the
        # control would jump the slider value to where the user clicked.
        # For this control, clicks which are not direct hits will slide both
        # slider parts

        if button:
            opt = QtWidgets.QStyleOptionSlider()
            self.initStyleOption(opt)

            self.active_slider = -1

            for i, value in enumerate([self._low, self._high]):
                opt.sliderPosition = value                
                hit = style.hitTestComplexControl(style.CC_Slider, opt, event.pos(), self)
                if hit == style.SC_SliderHandle:
                    self.active_slider = i
                    self.pressed_control = hit

                    self.triggerAction(self.SliderMove)
                    self.setRepeatAction(self.SliderNoAction)
                    self.setSliderDown(True)
                    break

            if self.active_slider < 0:
                self.pressed_control = QtWidgets.QStyle.SC_SliderHandle
                self.click_offset = self.__pixelPosToRangeValue(self.__pick(event.pos()))
                self.triggerAction(self.SliderMove)
                self.setRepeatAction(self.SliderNoAction)
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self.pressed_control != QtWidgets.QStyle.SC_SliderHandle:
            event.ignore()
            return

        event.accept()
        new_pos = self.__pixelPosToRangeValue(self.__pick(event.pos()))
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)

        if self.active_slider < 0:
            offset = new_pos - self.click_offset
            self._high += offset
            self._low += offset
            if self._low < self.minimum():
                diff = self.minimum() - self._low
                self._low += diff
                self._high += diff
            if self._high > self.maximum():
                diff = self.maximum() - self._high
                self._low += diff
                self._high += diff            
        elif self.active_slider == 0:
            if new_pos >= self._high:
                new_pos = self._high - 1
            self._low = new_pos
        else:
            if new_pos <= self._low:
                new_pos = self._low + 1
            self._high = new_pos

        self.click_offset = new_pos

        self.update()

        #self.emit(QtCore.SIGNAL('sliderMoved(int)'), new_pos)
        self.sliderMoved.emit(self._low, self._high)

    def __pick(self, pt):
        if self.orientation() == QtCore.Qt.Horizontal:
            return pt.x()
        else:
            return pt.y()

    def __pixelPosToRangeValue(self, pos):
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        style = QtWidgets.QApplication.style()

        gr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderGroove, self)
        sr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderHandle, self)

        if self.orientation() == QtCore.Qt.Horizontal:
            slider_length = sr.width()
            slider_min = gr.x()
            slider_max = gr.right() - slider_length + 1
        else:
            slider_length = sr.height()
            slider_min = gr.y()
            slider_max = gr.bottom() - slider_length + 1

        return style.sliderValueFromPosition(self.minimum(), self.maximum(),
                                             pos-slider_min, slider_max-slider_min,
                                             opt.upsideDown)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()
