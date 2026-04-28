DARK_THEME = """
QMainWindow, QWidget {
    background: #0B0D10;
    color: #E6EDF3;
    font-family: "Microsoft YaHei", "Segoe UI", Arial;
    font-size: 13px;
}
QFrame#Panel {
    background: #12161C;
    border: 1px solid #242A33;
    border-radius: 8px;
}
QLabel#Title {
    color: #E6EDF3;
    font-size: 18px;
    font-weight: 700;
}
QLabel#Subtle {
    color: #8B949E;
}
QLabel#StateIdle {
    color: #8B949E;
    font-size: 34px;
    font-weight: 800;
}
QLabel#StateWakeup {
    color: #3FB950;
    font-size: 34px;
    font-weight: 800;
}
QLabel#MetricName {
    color: #8B949E;
}
QLabel#MetricValue {
    color: #E6EDF3;
    font-weight: 600;
}
QProgressBar {
    background: #0B0D10;
    border: 1px solid #242A33;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}
QProgressBar::chunk {
    background: #2F81F7;
    border-radius: 4px;
}
QListWidget {
    background: #0B0D10;
    border: 1px solid #242A33;
    border-radius: 6px;
    color: #E6EDF3;
    padding: 6px;
}
QListWidget::item {
    padding: 4px;
}
QListWidget::item:selected {
    background: #1F6FEB;
}
QPushButton {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 6px;
    color: #E6EDF3;
    padding: 7px 12px;
}
QPushButton:hover {
    background: #1F2937;
}
QPushButton:checked {
    background: #1F6FEB;
    border-color: #2F81F7;
}
"""
