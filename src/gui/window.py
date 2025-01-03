# Library Imports
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QComboBox
)
from PyQt5 import QtWebEngineWidgets, QtCore, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox
from folium import plugins, IFrame
import qdarktheme
import folium as folium

# System Imports
import sys

# Project Imports
import algorithms.bfs as bfs
import algorithms.astar as astar
import algorithms.graph as graph_maker
import utilities.logger as logger
import predict as prediction_module
import main as main

from utilities.time import *

# Constants
WINDOW_TITLE = f"Traffic Prediction System - v{main.VERSION} - Developed by Daniel, Jeremy, Nicola & Minh"
WINDOW_SIZE = (1200, 500)
WINDOW_LOCATION = (160, 70)

# Global variables
graph = None
map_widget = None
selected_model = "lstm"  # Default model

def update_map(html):
    global map_widget

    map_widget.setHtml(html, QtCore.QUrl(""))

# Create PIN markers for the SCATs site on the map
def create_marker(scat, map_obj, color="green", size=30, tooltip=None, end=False):
    tip = str(scat)
    if tooltip:
        tip = tooltip

    html = f"""
        <div style="font-family: Arial; font-size: 11px; padding: 2px; text-align: center; min-width: 60px;">
        <b>Scat Number: {scat}</b>
        </div>
        """
    
    if end:
        html = f"""
        <div style="font-family: Arial; font-size: 11px; padding: 2px; text-align: center; min-width: 60px;">
        <b>End Scat: {scat}</b>
        </div>
        """

    popup = folium.Popup(html, max_width=200)

    custom_icon = folium.CustomIcon(
        icon_image="assets/pin.png",
        icon_size=(size, size),
        icon_anchor=(size // 2, size),
        popup_anchor=(0, -size)
    )

    folium.Marker(
        graph_maker.get_coords_by_scat(int(scat)),
        popup=popup,
        tooltip=tip,
        icon=custom_icon,
    ).add_to(map_obj)

# Create circle markers for the SCATs site on the map
def create_circle_marker(scat, map_obj, color="grey", size=2, tooltip=None, start=False):
    tip = str(scat)
    if tooltip:
        tip = tooltip
    
    html = f"""
        <div style="font-family: Arial; font-size: 11px; padding: 2px; text-align: center; min-width: 60px;">
        <b>Scat Number: {scat}</b>
        </div>
        """
    if start:
        html = f"""
        <div style="font-family: Arial; font-size: 11px; padding: 2px; text-align: center; min-width: 60px;">
        <b>Start Scat: {scat}</b>
        </div>
        """

    popup = folium.Popup(html, max_width=75)

    folium.CircleMarker(
        graph_maker.get_coords_by_scat(int(scat)),
        radius=size,
        color=color,
        fill=True,
        fill_color=color,
        popup=popup,
        tooltip=tip,
    ).add_to(map_obj)

def create_popup(index, time, distance):
    other = ""
    if index == 0:
        other = " - Fastest Path"

    html = f"""
     <div style="font-family: Arial; font-size: 10px; width: 110px;">
        <div style="padding: 4px; border-bottom: 1px solid #dee2e6;">
            <b>Path {index+1} {other}</b>
        </div>
        <table style="width: 100%; border-collapse: collapse; margin-top: 4px;">
            <tr style="background-color: #ffffff;">
                <td style="padding: 2px;"><b>Time:</b></td>
                <td style="padding: 2px;">{time} mins</td>
            </tr>
            <tr style="background-color: #ffffff;">
                <td style="padding: 2px;"><b>Distance:</b></td>
                <td style="padding: 2px;">{distance:.2f} km</td>
            </tr>
        </table>
    </div>
    """
    popup = folium.Popup(html, max_width=130)

    return popup

# Function to run the pathfinding algorithm

def get_threshold_color(flow):
    if flow <= 100:
        return "blue"
    elif flow > 100 and flow < 200:
        return "orange"
    elif flow >= 200 and flow <= 250:
        return "yellow"
    elif flow > 250:
        return "red"

def show_info_message(text, title):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText(text)
    msg.setWindowTitle(title)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.setWindowIcon(QIcon('assets/app_icon.png'))
    msg.exec_()

def run_pathfinding(start, end, date_time):
    global graph, menu_layout

    startCheck = graph_maker.does_scat_exist(start)
    endCheck = graph_maker.does_scat_exist(end)

    # Check if start and end scat numbers exist.
    if startCheck  == False and endCheck == False:
        show_info_message("Both SCAT numbers do not exist. Please enter a valid SCAT number.", "Invalid SCAT Number")
        return
    
    if startCheck == False:
        show_info_message(f"Start SCAT number {start} does not exist. Please enter a valid SCAT number.", "Invalid SCAT Number")
        return
    
    if endCheck == False:
        show_info_message("End SCAT number {end} does not exist. Please enter a valid SCAT number.", "Invalid SCAT Number")
        return

    # clear flow data
    astar.flow_dict = {}

    logger.log(f"Running pathfinding algorithm from {start} to {end}")

    graph = graph_maker.generate_graph()

    map_obj = folium.Map(
        location=(-37.820946, 145.060832), zoom_start=12, tiles="CartoDB Positron"
    )

    draw_all_scats(map_obj)

    logger.log(f"Using start and end node [{start}, {end}]")

    # format datetime
    datetime_split = date_time.split(" ")
    date = format_date_universal(datetime_split[0])
    time = round_to_nearest_15_minutes(datetime_split[1])
    formatted_datetime = f"{date} {time}"

    paths = astar.astar(graph, start, int(end), formatted_datetime, model = selected_model)

    if paths is None or len(paths) == 0:
        logger.log("No paths found.")
        return

    # Reverse the paths so the last path is drawn first
    reversed_paths = list(reversed(paths))
    display_index = len(reversed_paths) - 1
    is_main_path = True

    # Draw each path with a different color
    for path_index, path_info in enumerate(reversed_paths):
        is_main_path = False

        # if last path, make it blue
        if path_index == len(reversed_paths) - 1:
            is_main_path = True

        # print(path_info)
        # Draw the path segments
        for i in range(len(path_info['path']) - 1):
            current = path_info['path'][i]
            next_node = path_info['path'][i + 1]

            start_lat, start_long = graph_maker.get_coords_by_scat(current)
            end_lat, end_long = graph_maker.get_coords_by_scat(next_node)

            logger.log(f"Visited: {current} -> {next_node}")

            # If path is a main path
            if is_main_path:
                # Check thresholds for traffic flow, from dict.
                flow = astar.flow_dict.get(f"{next_node}")

                if not flow:
                    flow = 0

                color = get_threshold_color(flow)
            else:
                color = "#A0C8FF"

            # if the node is not the first or last node draw cirlce
            if i != 0 and i != len(path_info['path']) - 1:
                create_circle_marker(current, map_obj, color=color, size=2)

            folium.PolyLine(
                [(start_lat, start_long), (end_lat, end_long)],
                color=color,
                weight=2.5 if path_index == 0 else 2.0,
                opacity=1.0 if path_index == 0 else 0.8,
                popup=create_popup(display_index, path_info['time'], path_info['distance']),
                tooltip=f'Path {display_index + 1} - Segment: {current} → {next_node}', 
            ).add_to(map_obj)

        # Add a summary for this path
        logger.log(
            f"Path {display_index + 1} - {len(path_info['path'])} nodes, Color: {color}")
        display_index -= 1

     # add start and end markers on the map with the displayed scat number
    create_circle_marker(start, map_obj, color="lightgreen",
                         size=3, tooltip=f"Start - {start}", start=True)
    create_marker(end, map_obj, tooltip=f"End - {end}", end=True)

    update_map(map_obj._repr_html_())
    
    logger.log(f"Flow Dict -> {astar.flow_dict}")
    path_label_str = ""

    if len(paths) == 1:
        path_label_str = f"Pathfinding complete. {len(paths)} path found. \nDate: {get_day_of_week(date)} {format_date_to_words(date_time)} \nModel: {selected_model.upper()}"
    else:
        path_label_str = f"Pathfinding complete. {len(paths)} paths found. \nDate: {get_day_of_week(date)} {format_date_to_words(date_time)} \nModel: {selected_model.upper()}"
    
    if (menu_layout.parent().findChild(QLabel, "path_display") is not None):
        menu_layout.parent().findChild(QLabel, "path_display").setText(path_label_str)
    else:
        path_display = QLabel(path_label_str)
        path_display.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: white; background-color: #333; padding: 5px;"
        )
        path_display.setAlignment(QtCore.Qt.AlignCenter)
        path_display.setObjectName("path_display")
        menu_layout.addWidget(path_display)

    path_str = ""
    for index, path in enumerate(paths):
        node_string = " → ".join([str(node) for node in path['path']])
        time = f"{path['time']} minutes"
        if (path['time'] < 1):
            time = f"{round(path['time'] * 60, 2)} seconds"
        path_str += f"Path {index + 1}: {node_string} \nTime: {time} \nDistance: {path['distance']} km \n \n"

    if (menu_layout.parent().findChild(QPlainTextEdit, "path_text") is not None):
        menu_layout.parent().findChild(QPlainTextEdit, "path_text").setPlainText(path_str)
    else:
        path_text = QPlainTextEdit()
        path_text.setObjectName("path_text")
        path_text.setPlainText(path_str)
        path_text.setReadOnly(True)
        path_text.setStyleSheet(
            "font-size: 12px; color: white; background-color: #333; padding: 5px;"
        )
        # this will add a text box with the path information everytime you run the pathfinding algorithm
        menu_layout.addWidget(path_text)
   


# Create the menu widget for the GUI
def make_menu():
    global menu_layout
    logger.log("Creating menu...")

    # Create a widget for the menu
    menu_widget = QWidget()

    # Create a layout for the menu
    menu_layout = QVBoxLayout()
    menu_widget.setLayout(menu_layout)

    # Title at the top middle
    title = QLabel(f"Traffic Prediction System v{main.VERSION}")
    title.setStyleSheet(
        "font-size: 20px; font-weight: bold; color: white; background-color: #333; padding: 5px;"
    )
    title.setAlignment(QtCore.Qt.AlignCenter)
    menu_layout.addWidget(title)

    # Two textboxes, one for "Start Scats Number", one for "End Scats Number"
    start_scats = QtWidgets.QLineEdit()
    start_scats.setPlaceholderText("Start Scats Number")
    menu_layout.addWidget(start_scats)

    end_scats = QtWidgets.QLineEdit()
    end_scats.setPlaceholderText("End Scats Number")
    menu_layout.addWidget(end_scats)

    # Date and time input box
    datetime_select = QtWidgets.QDateTimeEdit()

    # Set time to 1st of October 2006 at 1:30AM
    datetime_select.setDateTime(QtCore.QDateTime.currentDateTime())
    datetime_select.setDate(QtCore.QDate(2006, 10, 1))
    datetime_select.setDisplayFormat("dd/MM/yyyy HH:mm")
    menu_layout.addWidget(datetime_select)

    # Dropdown for selecting the model
    model_dropdown = QComboBox()

    model_dropdown.addItem("SAEs")
    model_dropdown.addItem("CNN")
    model_dropdown.addItem("LSTM")
    model_dropdown.addItem("GRU")

    model_dropdown.setCurrentText("LSTM")  # Set default selection
    # model_dropdown.currentIndexChanged.connect(update_selected_model)
    model_dropdown.currentTextChanged.connect(
        lambda text: update_selected_model(text))
    menu_layout.addWidget(model_dropdown)

    # Button to run pathfinding algorithm
    run_button = QPushButton("Run Pathfinding")
    run_button.clicked.connect(
        lambda: run_pathfinding(
            start_scats.text(),
            end_scats.text(),
            datetime_select.text(),
        )
    )
    menu_layout.addWidget(run_button)

    # Add a stretcher to push buttons to the top
    menu_layout.addStretch()

    # Set the size and position of the menu
    menu_widget.setFixedWidth(int(WINDOW_SIZE[0] * 0.3))  # 20% of window width
    menu_widget.setFixedHeight(WINDOW_SIZE[1])  # 100% of window height

    return menu_widget

# Function to update the selected model for training


def update_selected_model(model):
    global selected_model
    model_map = {
        "SAEs": "saes",
        "CNN": "cnn",
        "LSTM": "lstm",
        "GRU": "gru"
    }
    selected_model = model_map[model]
    logger.log(f"Selected model: {selected_model}")


def create_map():
    global graph, map_widget

    logger.log("Creating map...")

    # create map
    map_obj = folium.Map(
        location=(-37.820946, 145.060832), zoom_start=12, tiles="CartoDB Positron"
    )
    map_widget = QtWebEngineWidgets.QWebEngineView()

    draw_all_scats(map_obj)

    return map_obj


def draw_all_scats(map_obj):
    # Get all scat numbers and long lats
    scats = graph_maker.get_all_scats()

    logger.log(f"Creating nodes...")
    for scat in scats:
        # create map markers for the scats
        create_circle_marker(scat, map_obj)


def make_window():
    global graph, map_widget

    logger.log("Creating window...")

    # Create main widget and layout
    main_widget = QWidget()
    main_layout = QHBoxLayout()
    main_widget.setLayout(main_layout)
    main_layout.setSpacing(0)  # Set spacing to zero

    update_map(create_map()._repr_html_())

    map_widget.page().setBackgroundColor(QtCore.Qt.transparent)

    # Add map and menu to layout
    main_layout.addWidget(make_menu())
    main_layout.addWidget(map_widget)

    return main_widget


def run():
    global app, graph

    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")

    window = QMainWindow()
    window.setWindowTitle(WINDOW_TITLE)
    window.setGeometry(
        WINDOW_LOCATION[0], WINDOW_LOCATION[1], WINDOW_SIZE[0], WINDOW_SIZE[1]
    )

    window.setWindowIcon(QIcon('assets/app_icon.png'))

    graph_maker.init()
    prediction_module.init()
    window.setCentralWidget(make_window())

    logger.log("Window created.")

    window.show()
    app.exec()
