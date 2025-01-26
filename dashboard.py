import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
try:
    df = pd.read_csv('Mall_Customers.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Dataset file 'Mall_Customers.csv' not found. Please ensure it is in the same directory as 'dashboard.py'.")

# Data Cleaning: Rename columns for easier use
df.rename(columns={
    'Annual Income (k$)': 'AnnualIncome',
    'Spending Score (1-100)': 'SpendingScore'
}, inplace=True)

# Feature Engineering: Add a 'Recency' column (all customers are considered active)
df['Recency'] = 0

# Perform clustering (KMeans with 5 clusters)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(df[['Recency', 'AnnualIncome', 'SpendingScore']])
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Predefined Insights for each cluster
cluster_insights = {
    0: "High-income, high-spending customers. Focus on premium product campaigns.",
    1: "Moderate-income, moderate-spending customers. Encourage loyalty programs.",
    2: "Low-income, low-spending customers. Focus on affordable options.",
    3: "High-income, low-spending customers. Promote value-added services.",
    4: "Low-income, high-spending customers. Offer targeted discounts and promotions."
}

# Create the Dash app
app = dash.Dash(__name__)
app.title = "Customer Segmentation Dashboard"

# Expose the server for Gunicorn
server = app.server

# Layout of the dashboard
app.layout = html.Div(
    style={'backgroundColor': '#f9f9f9', 'fontFamily': 'Arial, sans-serif', 'padding': '20px'},
    children=[
        html.H1("Customer Segmentation Dashboard", 
                style={'textAlign': 'center', 'color': '#333', 'marginBottom': '20px'}),
        
        # Tabs for Navigation
        dcc.Tabs([
            # Tab 1: Project Overview
            dcc.Tab(label="Project Overview", children=[
                dcc.Markdown('''
                ## Project Overview
                This dashboard is designed to analyze customer behavior using KMeans clustering. 
                It segments customers based on income, spending score, and age to derive actionable insights.
                
                ### Objectives:
                - Identify high-value customers.
                - Optimize marketing strategies for different customer segments.
                - Increase revenue and customer satisfaction.

                ### Features:
                - Interactive cluster analysis.
                - Age demographics visualization.
                - Downloadable insights and datasets.
                ''', style={'margin': '20px', 'lineHeight': '1.8'})
            ]),

            # Tab 2: Cluster Analysis
            dcc.Tab(label="Cluster Analysis", children=[
                # Cluster Selection
                html.Div(
                    style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'},
                    children=[
                        dcc.Dropdown(
                            id='cluster-dropdown',
                            options=[{'label': f'Cluster {i}', 'value': i} for i in range(5)],
                            value=0,  # Default to Cluster 0
                            clearable=False,
                            style={'width': '50%', 'padding': '10px', 'borderRadius': '5px'}
                        )
                    ]
                ),
                # Scatter Plot
                html.Div(
                    style={'marginTop': '30px'},
                    children=[
                        dcc.Graph(id='scatter-plot', style={'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'})
                    ]
                ),
                # Bar Chart for Cluster Averages
                html.Div(
                    style={'marginTop': '30px'},
                    children=[
                        dcc.Graph(id='bar-chart', style={'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'})
                    ]
                )
            ]),

            # Tab 3: Demographics
            dcc.Tab(label="Demographics", children=[
                # Gender Filter
                html.Div(
                    style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'},
                    children=[
                        dcc.Dropdown(
                            id='gender-dropdown',
                            options=[{'label': gender, 'value': gender} for gender in df['Gender'].unique()],
                            value=None,
                            placeholder="Select Gender",
                            style={'width': '50%', 'padding': '10px', 'borderRadius': '5px'}
                        )
                    ]
                ),
                # Age Distribution Plot
                html.Div(
                    style={'marginTop': '30px'},
                    children=[
                        dcc.Graph(id='age-distribution', style={'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'})
                    ]
                )
            ]),

            # Tab 4: Insights and Export
            dcc.Tab(label="Insights & Export", children=[
                html.Div(id='cluster-summary', style={'width': '60%', 'margin': 'auto', 'marginTop': '20px'}),
                html.Div([
                    html.Button("Download Data", id="download-button", style={'marginTop': '20px'}),
                    dcc.Download(id="download-dataframe")
                ], style={'textAlign': 'center'})
            ])
        ])
    ]
)

# Callback for Scatter Plot, Bar Chart, and Cluster Summary
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('bar-chart', 'figure'),
     Output('cluster-summary', 'children')],
    [Input('cluster-dropdown', 'value')]
)
def update_cluster_analysis(selected_cluster):
    # Filter data by selected cluster
    filtered_df = df[df['Cluster'] == selected_cluster]
    
    # Scatter plot
    scatter_fig = px.scatter(
        filtered_df, x='AnnualIncome', y='SpendingScore', color='Gender',
        title=f"Customer Segments: Cluster {selected_cluster}",
        labels={'AnnualIncome': 'Annual Income (k$)', 'SpendingScore': 'Spending Score (1-100)'},
        color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    )

    # Bar chart for averages
    cluster_summary = filtered_df[['Age', 'AnnualIncome', 'SpendingScore']].mean().reset_index()
    cluster_summary.columns = ['Metric', 'Value']
    bar_fig = px.bar(
        cluster_summary, x='Metric', y='Value',
        title=f"Average Metrics for Cluster {selected_cluster}",
        labels={'Value': 'Average Value'},
        color_discrete_sequence=['#636EFA']
    )

    # Summary and Insights
    insight_text = cluster_insights.get(selected_cluster, "No insights available for this cluster.")
    summary_div = html.Div([
        html.H4(f"Insights for Cluster {selected_cluster}"),
        html.P(insight_text, style={'marginTop': '10px', 'fontStyle': 'italic'})
    ])

    return scatter_fig, bar_fig, summary_div

# Callback for Age Distribution
@app.callback(
    Output('age-distribution', 'figure'),
    [Input('gender-dropdown', 'value')]
)
def update_age_distribution(selected_gender):
    # Filter data by gender if selected
    filtered_df = df[df['Gender'] == selected_gender] if selected_gender else df
    
    # Age distribution plot
    age_fig = px.histogram(
        filtered_df, x='Age', nbins=10, title="Age Distribution",
        labels={'Age': 'Age'}, color_discrete_sequence=['#636EFA']
    )
    return age_fig

# Callback for Downloading Data
@app.callback(
    Output("download-dataframe", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)
def download_data(n_clicks):
    return dcc.send_data_frame(df.to_csv, "customer_segmentation_data.csv")

if __name__ == '__main__':
    app.run_server(debug=True)
