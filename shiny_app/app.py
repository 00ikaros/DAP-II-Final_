import pandas as pd
import plotly.express as px
from shiny import App, render, reactive, ui
import plotly.graph_objects as go


# Load datasets outside the server function
processed_reviews = pd.read_csv("D:/uchicago/24 fall/data/final/DAP-II-Final-Presentation/shiny_app/processed_reviews.csv", parse_dates=['Timestamp'])
topic_representations = pd.read_csv("D:/uchicago/24 fall/data/final/DAP-II-Final-Presentation/shiny_app/topic_rep.csv")
daily_sentiment = pd.read_csv("D:/uchicago/24 fall/data/final/DAP-II-Final-Presentation/shiny_app/daily_sentiment.csv", parse_dates=['Date'])
monthly_sentiment = pd.read_csv("D:/uchicago/24 fall/data/final/DAP-II-Final-Presentation/shiny_app/monthly_sentiment.csv", parse_dates=['YearMonth'])
topic_probabilities = pd.read_csv("D:/uchicago/24 fall/data/final/DAP-II-Final-Presentation/shiny_app/prob_sentiment.csv")


# Define the UI
app_ui = ui.page_fluid(
    ui.h2("Sentiment and Topic Analysis Dashboard"),
    ui.navset_bar(
        # Page 1: Sentiment Analysis
        ui.nav_panel(
            "Sentiment Analysis",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Filters"),
                    # Date Range Filter
                    ui.input_date_range(
                        "date_range",
                        "Select Date Range:",
                        start="2004-01-01",  # Adjusted start date based on your data
                        end="2024-12-31"
                    )
                    # Removed Topic Selection here
                ),
                ui.panel_absolute(
                    ui.navset_tab(
                        ui.nav_panel(
                            "Sentiment Trends",
                            ui.output_ui("sentiment_trends")
                        )
                    )
                )
            )
        ),
        # Page 2: Topic Modeling
        ui.nav_panel(
            "Topic Modeling",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Filters"),
                    # Date Range Filter (optional for topic modeling)
                    ui.input_date_range(
                        "topic_date_range",
                        "Select Date Range:",
                        start="2004-01-01",
                        end="2024-12-31"
                    )
                ),
                ui.panel_absolute(
                    ui.output_ui("topic_distribution"),
                    ui.output_text_verbatim("topic_details")
                )
            )
        ),
        # Page 3: Detailed Reviews
        ui.nav_panel(
            "Detailed Reviews",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Filters"),
                    # Topic Selection
                    ui.input_select(
                        "filter_topic",
                        "Filter by Topic:",
                        choices=[],
                        multiple=True,
                        selected=None
                    ),
                    # Search Box
                    ui.input_text(
                        "search_text",
                        "Search Reviews:",
                        placeholder="Enter keyword..."
                    )
                ),
                ui.panel_absolute(
                    ui.output_table("reviews_table")
                )
            )
        ),
        title="Dashboard"
    )
)

# Define the server logic
def server(input, output, session):
    # Update Topic Selection Inputs Dynamically
    @reactive.Effect
    def update_topic_choices():
        print("Updating topic choices...")
        if not topic_representations.empty:
            topics = topic_representations['Topic'].drop_duplicates().sort_values().tolist()
            topics = [int(topic) for topic in topics if pd.notnull(topic)]
            topic_labels = [f"topic_{topic}" for topic in topics]
            print("Topic labels:", topic_labels)
            # Update only 'filter_topic' dropdown
            session.send_input_message(
                "filter_topic",
                {"choices": topic_labels, "selected": None}
            )
        else:
            print("Topic representations are empty.")
            session.send_input_message("filter_topic", {"choices": [], "selected": None})
    
    # Page 1: Sentiment Trends Plot
    @output
    @render.ui
    def sentiment_trends():
        df_daily = daily_sentiment
        df_monthly = monthly_sentiment

        if df_daily is None or df_monthly is None or df_daily.empty or df_monthly.empty:
            return ui.HTML("<p>No data available for sentiment trends.</p>")

        # Apply date range filter
        start_date = pd.to_datetime(input.date_range()[0])
        end_date = pd.to_datetime(input.date_range()[1])
        mask_daily = (df_daily['Date'] >= start_date) & (df_daily['Date'] <= end_date)
        mask_monthly = (df_monthly['YearMonth'] >= start_date) & (df_monthly['YearMonth'] <= end_date)

        if df_daily[mask_daily].empty and df_monthly[mask_monthly].empty:
            return ui.HTML("<p>No data available for the selected date range.</p>")

        # Create daily sentiment plot
        fig_daily = px.scatter(
            df_daily[mask_daily],
            x='Date',
            y='Sentiment',
            title='Average Daily Sentiment',
            labels={'Sentiment': 'Average Sentiment Score'}
        )

        # Create monthly sentiment plot
        fig_monthly = px.scatter(
            df_monthly[mask_monthly],
            x='YearMonth',
            y='Sentiment',
            title='Average Monthly Sentiment',
            labels={'Sentiment': 'Average Sentiment Score'}
        )

        # Combine the two plots into subplots
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Daily Sentiment", "Monthly Sentiment"))

        # Add traces
        for trace in fig_daily['data']:
            fig.add_trace(trace, row=1, col=1)
        for trace in fig_monthly['data']:
            fig.add_trace(trace, row=2, col=1)

        fig.update_layout(height=800, width = 1200, showlegend=False)

        # Return the HTML representation
        return ui.HTML(fig.to_html(full_html=False))

    # Page 2: Topic Distribution Plot
    @output
    @render.ui
    def topic_distribution():
        if processed_reviews is None or processed_reviews.empty or topic_representations is None or topic_representations.empty:
            return ui.HTML("<p>No data available for topic distribution.</p>")

        # Apply date range filter
        start_date = pd.to_datetime(input.topic_date_range()[0])
        end_date = pd.to_datetime(input.topic_date_range()[1])
        mask = (processed_reviews['Timestamp'] >= start_date) & (processed_reviews['Timestamp'] <= end_date)
        filtered_reviews = processed_reviews[mask]

        if filtered_reviews.empty:
            return ui.HTML("<p>No data available for the selected date range.</p>")

        # Count reviews per topic
        topic_counts = filtered_reviews['Assigned_Topic'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']
        topic_counts = topic_counts.sort_values('Topic', ascending=True)

        # Merge with topic representations for top words
        topic_words = topic_representations.groupby('Topic')['Word'].apply(lambda x: ', '.join(x.head(5))).reset_index()
        topic_counts = topic_counts.merge(topic_words, on='Topic', how='left')

        # Plot
        fig = px.bar(
            topic_counts,
            x='Topic',
            y='Count',
            text='Count',
            title='Topic Distribution',
            labels={'Topic': 'Topic Number', 'Count': 'Number of Reviews'},
            hover_data=['Word']
        )

        fig.update_traces(textposition='auto')
        fig.update_layout(
            xaxis=dict(tickmode='linear'),
            height=500,
            width=1200
        )

        # Return the HTML representation
        return ui.HTML(fig.to_html(full_html=False))

    # Removed the 'topic_details' output
    # Page 3: Detailed Reviews Table
    @output
    @render.table
    def reviews_table():
        if processed_reviews is None or processed_reviews.empty:
            return pd.DataFrame({"Error": ["No data available."]})

        # Apply topic filter
        selected_topics = input.filter_topic()
        if selected_topics:
            topic_numbers = [int(topic.split("_")[1]) for topic in selected_topics]
            mask_topic = processed_reviews['Assigned_Topic'].isin(topic_numbers)
        else:
            mask_topic = pd.Series([True] * len(processed_reviews))

        # Apply search text filter
        search_text = input.search_text()
        if search_text:
            search_text = search_text.lower()
            mask_search = processed_reviews['Review'].str.lower().str.contains(search_text, na=False)
        else:
            mask_search = pd.Series([True] * len(processed_reviews))

        # Combine all filters
        combined_mask = mask_topic & mask_search
        filtered_reviews = processed_reviews[combined_mask]

        # Select columns to display
        display_columns = ['PLACEKEY', 'Review', 'Timestamp', 'Assigned_Topic']
        display_df = filtered_reviews[display_columns].rename(columns={
            'PLACEKEY': 'Place Key',
            'Review': 'Review Text',
            'Timestamp': 'Timestamp',
            'Assigned_Topic': 'Topic Number'
        })

        return display_df.reset_index(drop=True)

# Create the Shiny app object
app = App(app_ui, server)