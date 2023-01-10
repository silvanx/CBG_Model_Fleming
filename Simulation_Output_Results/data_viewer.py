from pathlib import Path
import sys
import matplotlib
from PyQt5 import QtWidgets, QtGui
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
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas.axes.remove()
        self.canvas.axes = None
        self.axs = [
            self.canvas.fig.add_subplot(3, 1, 1),
            self.canvas.fig.add_subplot(3, 1, 2),
            self.canvas.fig.add_subplot(3, 1, 3)
        ]
        toolbar = NavigationToolbar(self.canvas, self)

        main_widget = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout()
        central_layout.addWidget(toolbar)
        central_layout.addWidget(self.canvas)
        main_widget.setLayout(central_layout)

        self.setCentralWidget(main_widget)


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
        self.df = pd.read_excel(
            "Simulation_Output_Results/output.xlsx"
            ).dropna(subset=['Simulation dir'])

        self.parameter_plot = MplCanvas(self, width=12, height=7, dpi=100)
        parameter_toolbar = NavigationToolbar(self.parameter_plot, self)

        self.description = QtWidgets.QLabel()
        self.description.setText("Description")
        self.description.setFont(QtGui.QFont("Roboto", 15))
        self.description.setFixedHeight(100)

        layout_plotting = QtWidgets.QVBoxLayout()
        layout_plotting.addWidget(parameter_toolbar)
        layout_plotting.addWidget(self.parameter_plot)
        self.parameter_plot.setMinimumWidth(200)

        open_detailed_view_button = QtWidgets.QPushButton()
        open_detailed_view_button.setText("Open detailed view (D)")
        self.detailed_view_window = DetailedViewWindow(self)
        open_detailed_view_button.clicked.connect(
            self.open_detailed_view_window)

        # Left side
        layout_left = QtWidgets.QVBoxLayout()

        layout_left.addWidget(self.description)
        layout_left.addLayout(layout_plotting)
        layout_left.addWidget(open_detailed_view_button)

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

        u.plot_pi_fitness_function(Path(self.fitness_dir),
                                   self.parameter_plot.fig,
                                   self.parameter_plot.axes)
        self.last_lambda = 1
        self.file_list = self.populate_file_list()
        self.refresh_file_list_display()

        self.setWindowTitle("Closed-loop parameters")
        self.show()

    def open_detailed_view_window(self):
        if self.current_file is not None:
            filename = Path(self.results_dir) / self.file_list[self.current_file]
            u.plot_ift_signals(
                filename,
                self.detailed_view_window.axs
            )
        self.detailed_view_window.canvas.draw()
        self.detailed_view_window.canvas.flush_events()
        self.detailed_view_window.show()

    def change_file_dir(self, event):
        newdir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            '',
            self.results_dir
            )
        if newdir:
            self.file_dir = newdir
            self.directory_label.setText(newdir)
            self.current_file = None
            self.file_list = self.populate_file_list()
            self.refresh_file_list_display()

    def change_fitness_dir(self, event):
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
                    self.parameter_plot.axes.patches.remove(a)
                self.last_arrows = None
            item = self.file_list_widget.selectedItems()[0]
            text = item.text()
            self.current_file = self.file_list.index(text)
            f = Path(self.results_dir) / text

            row = self.df[self.df["Simulation dir"].str.contains(text)]
            if not row.empty:
                description = (
                    f"ID: {row.iloc[0]['Simulation number']}\t"
                    f"{row.iloc[0]['Sim duration [ms]']} ms\t"
                    f"controller: {row.iloc[0]['controller']} "
                    f"({row.iloc[0]['IFT experiment length [s]']} s)\n"
                    f"Kp init: {row.iloc[0]['Kp']}, "
                    f"Ti init: {row.iloc[0]['Ti']}\n"
                    f"gamma: {row.iloc[0]['gamma']}, "
                    f"lambda: {row.iloc[0]['lambda']}\n"
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
                                               lam=row.iloc[0]['lambda'])
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


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()
