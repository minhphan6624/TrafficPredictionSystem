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
WINDOW_TITLE = "TrafficPredictionSystem"
WINDOW_SIZE = (1200, 500)
WINDOW_LOCATION = (160, 70)

# Global variables
graph = None
map_widget = None
selected_model = "seas"  # Default model


def update_map(html):
    global map_widget

    map_widget.setHtml(html, QtCore.QUrl(""))

# Create PIN markers for the SCATs site on the map
def create_marker(scat, map_obj, color="green", size=30, tooltip=None):
    tip = "Scat " + str(scat)
    if tooltip:
        tip = tooltip

    html = f"""
        <h4>Scat Number: {scat}</h4>
        """
    iframe = folium.IFrame(html=html, width=150, height=100)
    popup = folium.Popup(iframe, max_width=200)

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
def create_circle_marker(scat, map_obj, color="grey", size=2, tooltip=None):
    tip = "Scat " + str(scat)
    if tooltip:
        tip = tooltip

    html = f"""
        <h4>Scat Number: {scat}</h4>
        """
    iframe = folium.IFrame(html=html, width=150, height=100)
    popup = folium.Popup(iframe, max_width=200)

    folium.CircleMarker(
        graph_maker.get_coords_by_scat(int(scat)),
        radius=size,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=popup,
        tooltip=tip,
    ).add_to(map_obj)

# Function to run the pathfinding algorithm
def run_pathfinding(start, end, datetime):
    global graph, menu_layout

    logger.log(f"Running pathfinding algorithm from {start} to {end}")

    graph = graph_maker.generate_graph()

    map_obj = folium.Map(
        location=(-37.820946, 145.060832), zoom_start=12, tiles="CartoDB Positron"
    )

    draw_all_scats(map_obj)

    logger.log(f"Using start and end node [{start}, {end}]")

    # format datetime
    datetime_split = datetime.split(" ")
    date = datetime_split[0]
    time = round_to_nearest_15_minutes(datetime_split[1])
    formatted_datetime = f"{date} {time}"

    paths = astar.astar(graph, start, int(end), formatted_datetime, model = selected_model)

    if paths is None or len(paths) == 0:
        logger.log("No paths found.")
        return

    # Reverse the paths so the last path is drawn first
    reversed_paths = list(reversed(paths))
    display_index = len(reversed_paths) - 1
    # Draw each path with a different color
    for path_index, path_info in enumerate(reversed_paths):
        color = "#A0C8FF"
        # if last path, make it blue
        if path_index == len(reversed_paths) - 1:
            color = "blue"

        # print(path_info)

        logger.log(f"\nDrawing Path {path_index + 1} in {color}")

        # Draw the path segments
        for i in range(len(path_info['path']) - 1):
            current = path_info['path'][i]
            next_node = path_info['path'][i + 1]

            start_lat, start_long = graph_maker.get_coords_by_scat(current)
            end_lat, end_long = graph_maker.get_coords_by_scat(next_node)

            logger.log(f"Visited: {current} -> {next_node}")

            # if the node is not the first or last node draw cirlce
            if i != 0 and i != len(path_info['path']) - 1:
                create_circle_marker(current, map_obj, color=color, size=2)

            # Create the path line with the current color
            folium.PolyLine(
                [(start_lat, start_long), (end_lat, end_long)],
                color=color,
                weight=2.5 if path_index == 0 else 2.0,
                opacity=1.0 if path_index == 0 else 0.8,
                popup=f'Path {display_index + 1}',
                tooltip=f'Path {display_index + 1} - Segment: {current} → {next_node}'
            ).add_to(map_obj)

        # Add a summary for this path
        logger.log(
            f"Path {display_index + 1} - {len(path_info['path'])} nodes, Color: {color}")
        display_index -= 1

     # add start and end markers on the map with the displayed scat number
    create_circle_marker(start, map_obj, color="red",
                         size=3, tooltip=f"Start - {start}")
    create_marker(end, map_obj, tooltip=f"End - {end}")

    update_map(map_obj._repr_html_())

    # should display the time as well
    path_display = QLabel(f"Pathfinding complete. {len(paths)} paths found.")
    path_display.setStyleSheet(
        "font-size: 16px; font-weight: bold; color: white; background-color: #333; padding: 5px;"
    )
    path_display.setAlignment(QtCore.Qt.AlignCenter)
    menu_layout.addWidget(path_display)

    path_str = ""
    for index, path in enumerate(paths):
        node_string = " -> ".join([str(node) for node in path['path']])
        time = f"{path['time']} minutes"
        if (path['time'] < 1):
            time = f"{round(path['time'] * 60, 2)} seconds"
        path_str += f"Path {index + 1}: {node_string} \n {time} - {path['distance']} km \n \n"

    path_text = QPlainTextEdit()
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
    datetime_select.setDateTime(QtCore.QDateTime.currentDateTime())
    menu_layout.addWidget(datetime_select)

    # Dropdown for selecting the model
    model_dropdown = QComboBox()

    model_dropdown.addItem("SAEs")
    model_dropdown.addItem("CNN")
    model_dropdown.addItem("LSTM")
    model_dropdown.addItem("GRU")

    model_dropdown.setCurrentText("SAEs")  # Set default selection
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

    # Button to reset the map (doesn't currently work as expected)
    reset_button = QPushButton("Reset")
    reset_button.clicked.connect(
        lambda: update_map(create_map()._repr_html_()))
    # updates the map but the visual doesn't update
    menu_layout.addWidget(reset_button)

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
    # qdarktheme.setup_theme("dark")

    window = QMainWindow()
    window.setWindowTitle(WINDOW_TITLE)
    window.setGeometry(
        WINDOW_LOCATION[0], WINDOW_LOCATION[1], WINDOW_SIZE[0], WINDOW_SIZE[1]
    )

    graph_maker.init()
    prediction_module.init()
    window.setCentralWidget(make_window())

    logger.log("Window created.")

    window.show()
    app.exec()
