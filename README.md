# Python Project Group 8
This is our python project for CIS4930

## File Listing
```
PythonWeatherProject/
│
├── data/
│ └── sample.json
│
├── static/
│ └── styles.css
│
├── templates/
│ └── base.html
│ └── index.html
│ └── results.html
│
├── tests/
│ └── __init__.py
│ └── test_algorithms.py
│ └── test_app.py
│ └── test_data_processor.py
│
├── algorithms.py
├── app.py
├── config.py
├── data_processor.py
├── requirementx.txt
└── README.md
```
## How to Setup/Run
Setup virtual environment
Install requirements:
```bash
pip install -r requirements.txt
```
**To run the application:**
```bash
python app.py
```
view on local host: http://127.0.0.1:5000/

**To run tests:**
```bash
pytest tests/
```
**Using the application:**
- Type in a City into the search bar
- Click "Get Forecast"
- View forecast results
- To repeat, click "Search Again" at the bottom
