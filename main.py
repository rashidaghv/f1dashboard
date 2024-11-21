import fastf1
from fastf1 import plotting
import matplotlib.pyplot as plt
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Enable cache for faster data retrieval
fastf1.Cache.enable_cache('F:/ditt/f1_cache')

# Load race data (Example: 2023 Bahrain GP)
race = fastf1.get_event(2023, 'Bahrain Grand Prix')
session = race.get_session('Race')  # Get the race session
session.load()  # Load session data

# Map driver abbreviations to car numbers using session.results
results = session.results
driver_to_car = {row.Abbreviation: str(row.DriverNumber) for _, row in results.iterrows()}

# Prepare lap data
laps = session.laps
laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

# Ensure 'Stint' column exists
if 'Stint' not in laps.columns:
    # Create 'Stint' column based on pit stops
    laps['Stint'] = laps.groupby('Driver')['PitInLap'].cumsum().fillna(0).astype(int) + 1

# Get a list of unique drivers
unique_drivers = laps['Driver'].unique()

# Create lap time plot
fig_lap_times = px.line(
    laps,
    x='LapNumber',
    y='LapTimeSeconds',
    color='Driver',
    title='Lap Times for All Drivers',
    labels={'LapNumber': 'Lap Number', 'LapTimeSeconds': 'Lap Time (seconds)'}
)

# Create position changes plot
fig_positions = px.line(
    laps,
    x='LapNumber',
    y='Position',
    color='Driver',
    title='Position Changes During the Race',
    labels={'LapNumber': 'Lap Number', 'Position': 'Position'}
)
fig_positions.update_yaxes(autorange="reversed")  # Reverse position axis

# Highlight pit stops on lap time graph
pit_stops = laps[laps['PitInTime'].notnull()]
for _, pit_stop in pit_stops.iterrows():
    fig_lap_times.add_vline(
        x=pit_stop['LapNumber'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Pit Stop: {pit_stop['Driver']}",
        annotation_position="top left"
    )

# Create Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("F1 Race Strategy Dashboard"),
    html.Div(id='pit-stop-prediction'),  # Placeholder for pit stop prediction
    html.Div([
        html.H3("Race Information"),
        html.P(f"Event: {race.EventName}"),
        html.P(f"Location: {race.Location}"),
        html.P(f"Date: {race.EventDate}")
    ]),
    html.Label("Select Driver:"),
    dcc.Dropdown(
        id='driver-dropdown',
        options=[{'label': driver, 'value': driver} for driver in unique_drivers],
        value=unique_drivers[0],  # Default value
        clearable=False
    ),
    dcc.Checklist(
        id='pitstop-filter',
        options=[{'label': 'Highlight Pit Stops', 'value': 'show_pitstops'}],
        value=[]
    ),
    dcc.Graph(id='lap-time-graph'),
    dcc.Graph(figure=fig_positions),  # Position graph
    dcc.Graph(id='telemetry-graph'),  # Telemetry graph
    dcc.Graph(id='tire-degradation-graph')  # Tire degradation graph
])

# Callback to update lap time graph based on driver selection and pit stop filter
@app.callback(
    Output('lap-time-graph', 'figure'),
    [Input('driver-dropdown', 'value'), Input('pitstop-filter', 'value')]
)
def update_lap_time_graph(selected_driver, pitstop_filter):
    # Filter data by selected driver
    driver_laps = laps[laps['Driver'] == selected_driver]
    fig = px.line(
        driver_laps,
        x='LapNumber',
        y='LapTimeSeconds',
        title=f'Lap Times for {selected_driver}',
        labels={'LapNumber': 'Lap Number', 'LapTimeSeconds': 'Lap Time (seconds)'}
    )
    # Highlight pit stops if the filter is enabled
    if 'show_pitstops' in pitstop_filter:
        driver_pit_stops = driver_laps[driver_laps['PitInTime'].notnull()]
        fig.add_scatter(
            x=driver_pit_stops['LapNumber'],
            y=driver_pit_stops['LapTimeSeconds'],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Pit Stops'
        )
    return fig

# Callback to update telemetry graph
@app.callback(
    Output('telemetry-graph', 'figure'),
    [Input('driver-dropdown', 'value')]
)
def update_telemetry_graph(selected_driver):
    # Get the car number for the selected driver
    car_number = driver_to_car.get(selected_driver)

    if car_number not in session.car_data:
        return px.scatter(
            title=f"No telemetry data available for {selected_driver}",
            labels={'TimeSeconds': 'Time (seconds)', 'Speed': 'Speed (km/h)'}
        )

    # Fetch telemetry using car number
    telemetry = session.car_data[car_number]
    telemetry['TimeSeconds'] = telemetry['Time'].dt.total_seconds()

    # Create telemetry plot
    fig = px.line(
        telemetry,
        x='TimeSeconds',
        y='Speed',
        title=f'Telemetry Data (Speed) for {selected_driver}',
        labels={'TimeSeconds': 'Time (seconds)', 'Speed': 'Speed (km/h)'}
    )
    return fig

# Callback to update tire degradation graph
@app.callback(
    Output('tire-degradation-graph', 'figure'),
    [Input('driver-dropdown', 'value')]
)
def update_tire_degradation_graph(selected_driver):
    driver_laps = laps[laps['Driver'] == selected_driver].copy()
    driver_laps['TyreAge'] = driver_laps.groupby('Stint').cumcount() + 1
    fig = px.scatter(
        driver_laps,
        x='TyreAge',
        y='LapTimeSeconds',
        color='Compound',
        trendline='ols',
        title=f'Tire Degradation for {selected_driver}',
        labels={'TyreAge': 'Tire Age (laps)', 'LapTimeSeconds': 'Lap Time (seconds)'}
    )
    return fig

# Predict optimal pit stop lap using improved model
def predict_pit_stop_improved(driver):
    try:
        # Get lap data for the selected driver
        driver_laps = laps[laps['Driver'] == driver].copy()

        # Get the current stint number
        current_stint = driver_laps['Stint'].max()

        # Filter laps to only include the current stint
        driver_stint_laps = driver_laps[driver_laps['Stint'] == current_stint].copy()

        # Check if there are enough laps for prediction
        if driver_stint_laps.shape[0] < 5:
            print(f"Not enough laps for driver {driver} in current stint.")
            return None

        # Prepare features
        driver_stint_laps['TyreAge'] = driver_stint_laps.groupby('Stint').cumcount() + 1
        target = 'LapTimeSeconds'

        # Check if 'Compound' data is available and valid
        if 'Compound' in driver_stint_laps.columns and driver_stint_laps['Compound'].notnull().all():
            features = ['LapNumber', 'TyreAge', 'Compound']
            X = driver_stint_laps[features]
            # Preprocessing for categorical data
            preprocessor = ColumnTransformer(
                transformers=[
                    ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Compound'])
                ], remainder='passthrough'
            )
            X_processed = preprocessor.fit_transform(X)
        else:
            print(f"'Compound' data not available for driver {driver}. Adjusting model.")
            features = ['LapNumber', 'TyreAge']
            X = driver_stint_laps[features]
            X_processed = X.values  # No preprocessing needed

        y = driver_stint_laps[target]

        # Ensure there are no missing values
        valid_indices = (~np.isnan(X_processed).any(axis=1)) & (~y.isna())
        X_processed = X_processed[valid_indices]
        y = y[valid_indices]  # Use direct indexing instead of .iloc[]

        # Ensure feature and target lengths match
        if X_processed.shape[0] != y.shape[0]:
            print("Mismatch between features and target variable lengths.")
            return None

        # Train Gradient Boosting Regressor
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_processed, y)

        # Prepare future laps data
        future_lap_numbers = np.arange(driver_stint_laps['LapNumber'].max() + 1,
                                       driver_stint_laps['LapNumber'].max() + 20)
        future_tyre_age = future_lap_numbers - driver_stint_laps['LapNumber'].iloc[-1] + driver_stint_laps['TyreAge'].iloc[-1]

        future_data = pd.DataFrame({
            'LapNumber': future_lap_numbers,
            'TyreAge': future_tyre_age
        })

        if 'Compound' in features:
            future_compound = driver_stint_laps['Compound'].iloc[-1]
            future_data['Compound'] = future_compound
            future_X_processed = preprocessor.transform(future_data)
        else:
            future_X_processed = future_data.values

        # Predict future lap times
        future_times = model.predict(future_X_processed)

        # Determine threshold
        recent_lap_times = y.tail(3)
        threshold = recent_lap_times.mean() + 2 * recent_lap_times.std()

        # Predict the lap where lap time exceeds the threshold
        predicted_lap = None
        for lap_num, lap_time in zip(future_lap_numbers, future_times):
            if lap_time > threshold:
                predicted_lap = int(lap_num)
                break

        return predicted_lap
    except Exception as e:
        print(f"Error in predict_pit_stop_improved for driver {driver}: {e}")
        return None

@app.callback(
    Output('pit-stop-prediction', 'children'),
    [Input('driver-dropdown', 'value')]
)
def update_pit_stop_prediction(selected_driver):
    predicted_lap = predict_pit_stop_improved(selected_driver)
    if predicted_lap:
        return f"Predicted Pit Stop Lap for {selected_driver}: Lap {predicted_lap}"
    else:
        return f"Unable to predict pit stop lap for {selected_driver}."

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
