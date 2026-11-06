import sys
from pathlib import Path

# Add the main project folder to Python's import path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.flood_map import load_flood_map_data, create_flood_map
import pandas as pd
import plotly.express as px
import streamlit as st

from backend.cyclone_predictor import (
    load_cyclone_data,
    load_cyclone_model,
    predict_next_wind,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

CYCLONE_FILE = DATA_DIR / "india_cyclones.csv"
FLOOD_FILE = OUTPUT_DIR / "flood_district_risk_ranking.csv"

st.set_page_config(
    page_title="Disaster Governance AI",
    layout="wide",
)

# --------------------------------------------------
# PROFESSIONAL GOVERNMENT-STYLE LIGHT THEME
# --------------------------------------------------

st.markdown("""
<style>
[data-testid="stMarkdownContainer"] h1 {
    color: #FFFFFF !important;
}
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: #102B50 !important;
    border-right: 4px solid #D6A52A;
}
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] [role="radio"] {
    padding: 8px 10px;
    border-radius: 6px;
}
[data-testid="stSidebar"] [role="radio"]:hover {
    background: rgba(214,165,42,0.18);
}
h1, h2, h3, h4 { color: #142D4E !important; }
p, li { color: #46566B; line-height: 1.65; }
[data-testid="stMetric"] {
    background: #F5F7FA !important;
    border: 1px solid #DFE5ED !important;
    border-left: 4px solid #D6A52A !important;
    border-radius: 9px !important;
    padding: 18px !important;
    box-shadow: 0 3px 10px rgba(16,43,80,0.05);
}
[data-testid="stMetricLabel"] { color: #64748B !important; }
[data-testid="stMetricValue"] { color: #142D4E !important; }
.stButton > button, [data-testid="stDownloadButton"] button {
    background: #102B50 !important;
    color: #FFFFFF !important;
    border: 1px solid #102B50 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
.stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
    background: #D6A52A !important;
    border-color: #D6A52A !important;
    color: #102B50 !important;
}
[data-testid="stExpander"] {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    background: #FFFFFF !important;
    border-radius: 8px;
}
hr { border-color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------

st.markdown("""<div style="background-color:#102B50;padding:30px 32px;border-radius:10px;border-bottom:5px solid #D6A52A;margin-bottom:28px;">
<div style="color:#E6BD63;font-size:12px;font-weight:700;letter-spacing:2px;margin-bottom:12px;">NATIONAL DISASTER INTELLIGENCE PLATFORM</div>
<h1 style="color:#FFFFFF !important;font-size:34px;margin:0;padding:0;font-weight:750;">Disaster Governance AI</h1>
<p style="color:#FFFFFF !important;font-size:15px;margin-top:12px;margin-bottom:0;">Cyclone Intelligence | Flood Hazard Mapping | Predictive Analytics | Governance Decision Support</p>
</div>""", unsafe_allow_html=True)

# --------------------------------------------------
# CACHED LOADERS
# --------------------------------------------------

@st.cache_data
def load_csv(path_string):
    return pd.read_csv(path_string, low_memory=False)


@st.cache_data
def get_cyclone_data():
    return load_cyclone_data()


@st.cache_resource
def get_cyclone_model():
    return load_cyclone_model()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Disaster Intelligence")
module = st.sidebar.radio(
    "Select Module",
   [
    "Overview",
    "Cyclone Intelligence",
    "Cyclone Prediction",
    "Flood Risk Intelligence",
    "Flood Hazard Map",
    "Governance Decision Support",
    "AI Model Explainability",
    "Model Validation",
    "River Network Intelligence",

],
)


# --------------------------------------------------
# LOAD AVAILABLE DATA
# --------------------------------------------------

cyclone_df = None
flood_df = None

if CYCLONE_FILE.exists():
    cyclone_df = load_csv(str(CYCLONE_FILE))

if FLOOD_FILE.exists():
    flood_df = load_csv(str(FLOOD_FILE))


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if module == "Overview":
    st.header("National Disaster Intelligence Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cyclone Intelligence")
        if cyclone_df is not None:
            st.metric("Track observations", f"{len(cyclone_df):,}")
            st.metric(
                "Unique storms",
                f"{cyclone_df['STORM_ID'].nunique():,}",
            )
        else:
            st.warning("Cyclone dataset not found.")

    with col2:
        st.subheader("Flood Intelligence")
        if flood_df is not None:
            st.metric("Districts analysed", f"{len(flood_df):,}")
            high_count = flood_df["Risk_Category"].isin(
                ["High", "Very High"]
            ).sum()
            st.metric("High and very high risk districts", int(high_count))
        else:
            st.warning("Flood risk ranking file not found.")

    st.divider()
    st.subheader("Project capabilities")
    st.markdown(
        """
        - Historical cyclone track analysis.
        - Experimental next-recorded-observation wind-speed estimation.
        - District-level historical flood risk ranking.
        - State-wise flood risk comparison.
        - Exploratory disaster preparedness and governance decision support.
        """
    )
    st.info(
        "This research prototype does not provide official disaster warnings, "
        "validated future cyclone forecasts, or evacuation guidance."
    )


# --------------------------------------------------
# CYCLONE INTELLIGENCE
# --------------------------------------------------

elif module == "Cyclone Intelligence":
    st.header(" Cyclone Intelligence")

    if cyclone_df is None:
        st.error("Cyclone dataset not found: data/processed/india_cyclones.csv")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Track observations", f"{len(cyclone_df):,}")
    c2.metric("Unique storms", f"{cyclone_df['STORM_ID'].nunique():,}")
    c3.metric("Subbasins", f"{cyclone_df['SUBBASIN'].nunique():,}")

    st.subheader("Historical cyclone observation locations")
    plot_df = cyclone_df.dropna(
        subset=["LATITUDE", "LONGITUDE_180"]
    )

    fig = px.scatter_geo(
        plot_df,
        lat="LATITUDE",
        lon="LONGITUDE_180",
        color="WIND_KNOTS",
        hover_name="NAME",
        hover_data=[
            "DATE_UTC",
            "SUBBASIN",
            "INTENSITY_CATEGORY",
        ],
        scope="asia",
        color_continuous_scale="Turbo",
        title="North Indian Ocean historical track observations",
    )
    fig.update_geos(
        lataxis_range=[-5, 35],
        lonaxis_range=[35, 110],
        showcountries=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cyclone intensity distribution")
    counts = (
        cyclone_df["INTENSITY_CATEGORY"]
        .value_counts()
        .rename_axis("Intensity")
        .reset_index(name="Observations")
    )
    st.bar_chart(counts, x="Intensity", y="Observations")

    st.subheader("Historical cyclone data")
    st.dataframe(cyclone_df.head(100), use_container_width=True)

    st.info(
        "IBTrACS contains historical observations, not live cyclone positions "
        "or official future forecasts."
    )


# --------------------------------------------------
# CYCLONE PREDICTION
# --------------------------------------------------

elif module == "Cyclone Prediction":
    st.header("AI-Based Cyclone Intensity Estimation")
    st.write(
        "Select a historical storm to estimate its next recorded wind-speed "
        "observation using your trained Random Forest sequence model."
    )

    try:
        prediction_df = get_cyclone_data()
        model, features = get_cyclone_model()

        required_columns = [
            "STORM_ID",
            "DATE_UTC",
            "WIND_KNOTS",
            "LATITUDE",
            "LONGITUDE_180",
        ]

        valid_df = prediction_df.dropna(
            subset=required_columns
        ).copy()

        if valid_df.empty:
            st.warning("No valid cyclone observations are available.")
            st.stop()

        storm_ids = valid_df["STORM_ID"].drop_duplicates().tolist()

        def storm_label(storm_id):
            storm = valid_df[
                valid_df["STORM_ID"] == storm_id
            ].sort_values("DATE_UTC")
            if "NAME" in storm.columns:
                names = storm["NAME"].dropna()
                if not names.empty:
                    return f"{names.iloc[0]} ({storm_id})"
            return str(storm_id)

        selected_storm = st.selectbox(
            "Select a historical cyclone",
            storm_ids,
            format_func=storm_label,
        )

        storm_history = valid_df[
            valid_df["STORM_ID"] == selected_storm
        ].sort_values("DATE_UTC")

        st.subheader("Latest historical observations")
        history_columns = [
            col for col in [
                "DATE_UTC",
                "NAME",
                "WIND_KNOTS",
                "LATITUDE",
                "LONGITUDE_180",
            ]
            if col in storm_history.columns
        ]
        st.dataframe(
            storm_history[history_columns].tail(10),
            use_container_width=True,
        )

        if len(storm_history) < 2:
            st.warning("This storm has fewer than two valid observations.")
        elif st.button(
            "Estimate next recorded wind speed",
            type="primary",
        ):
            try:
                result = predict_next_wind(
                    storm_history,
                    model,
                    features,
                )

                st.success("Estimation completed.")

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Latest recorded wind",
                    f"{result['previous_wind_knots']:.1f} knots",
                    f"{result['previous_wind_kmh']:.1f} km/h",
                )
                change = (
                    result["predicted_wind_knots"]
                    - result["previous_wind_knots"]
                )
                c2.metric(
                    "Estimated next wind",
                    f"{result['predicted_wind_knots']:.1f} knots",
                    f"{change:+.1f} knots",
                )
                c3.metric(
                    "Estimated next wind",
                    f"{result['predicted_wind_kmh']:.1f} km/h",
                )

                st.write(
                    "**Approximate next observation time (UTC):** "
                    + result["next_observation_time"].strftime(
                        "%Y-%m-%d %H:%M"
                    )
                )
                st.write(
                    "**Latest recorded location:** "
                    f"{result['latitude']:.2f}, "
                    f"{result['longitude']:.2f}"
                )

                chart_df = pd.DataFrame(
                    {
                        "Observation": [
                            "Latest recorded",
                            "Estimated next",
                        ],
                        "Wind speed (km/h)": [
                            result["previous_wind_kmh"],
                            result["predicted_wind_kmh"],
                        ],
                    }
                )
                fig = px.bar(
                    chart_df,
                    x="Observation",
                    y="Wind speed (km/h)",
                    color="Observation",
                    title="Latest recorded versus estimated wind speed",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.warning(
                    "Experimental estimate only. The model was trained to "
                    "estimate the next recorded observation. The future "
                    "location is unknown and is assumed to remain unchanged "
                    "from the latest recorded location. This is not a "
                    "validated 6-hour, 12-hour, or 24-hour forecast."
                )

            except ValueError as error:
                st.warning(str(error))
            except Exception as error:
                st.error(f"Prediction failed: {error}")

    except FileNotFoundError as error:
        st.error(str(error))
        st.info(
            "Check that the trained model exists at "
            "models/cyclone_sequence_model.pkl and the cyclone CSV exists "
            "at data/processed/india_cyclones.csv."
        )
    except Exception as error:
        st.error(f"Could not load the prediction module: {error}")

# --------------------------------------------------
# FLOOD RISK INTELLIGENCE
# --------------------------------------------------

elif module == "Flood Risk Intelligence":
    st.header("District-Level Flood Risk Intelligence")

    if flood_df is None:
        st.error(
            "Flood risk ranking file not found. Run "
            "python backend\\flood_risk_analysis.py first."
        )
        st.stop()

    states = sorted(flood_df["State_Name"].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox(
        "Filter by state",
        ["All States"] + states,
    )

    filtered_df = flood_df.copy()
    if selected_state != "All States":
        filtered_df = filtered_df[
            filtered_df["State_Name"] == selected_state
        ]

    if filtered_df.empty:
        st.warning("No district records found for this selection.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts analysed", f"{len(filtered_df):,}")
    c2.metric(
        "Average historical risk score",
        f"{filtered_df['Historical_Risk_Score'].mean():.2f}",
    )
    high_count = filtered_df["Risk_Category"].isin(
        ["High", "Very High"]
    ).sum()
    c3.metric("High risk districts", int(high_count))
    c4.metric(
        "Highest risk score",
        f"{filtered_df['Historical_Risk_Score'].max():.2f}",
    )

    st.divider()
    st.subheader("Top 10 historically high-risk districts")
    top10 = (
        filtered_df.sort_values(
            "Historical_Risk_Score",
            ascending=False,
        ).head(10)
    )

    fig_top = px.bar(
        top10.sort_values("Historical_Risk_Score"),
        x="Historical_Risk_Score",
        y="Dist_Name",
        color="Risk_Category",
        orientation="h",
        title="Top 10 districts by historical risk score",
        labels={
            "Historical_Risk_Score": "Historical Risk Score",
            "Dist_Name": "District",
        },
        color_discrete_map={
            "Low": "green",
            "Moderate": "yellow",
            "High": "orange",
            "Very High": "red",
        },
    )
    st.plotly_chart(fig_top, use_container_width=True)

    st.subheader("State-wise historical flood risk")
    state_risk = (
        filtered_df.groupby("State_Name", as_index=False)
        .agg(
            Average_Risk=("Historical_Risk_Score", "mean"),
            District_Count=("Dist_Name", "count"),
        )
        .sort_values("Average_Risk", ascending=False)
    )

    fig_state = px.bar(
        state_risk,
        x="State_Name",
        y="Average_Risk",
        title="Average historical risk score by state",
        labels={
            "State_Name": "State",
            "Average_Risk": "Average Historical Risk Score",
        },
    )
    st.plotly_chart(fig_state, use_container_width=True)

    st.subheader("District risk category distribution")
    category_counts = (
        filtered_df["Risk_Category"]
        .value_counts()
        .reindex(
            ["Low", "Moderate", "High", "Very High"],
            fill_value=0,
        )
        .rename_axis("Risk Category")
        .reset_index(name="Districts")
    )

    fig_category = px.pie(
        category_counts,
        names="Risk Category",
        values="Districts",
        title="Historical district risk distribution",
        color="Risk Category",
        color_discrete_map={
            "Low": "green",
            "Moderate": "yellow",
            "High": "orange",
            "Very High": "red",
        },
    )
    st.plotly_chart(fig_category, use_container_width=True)

    st.subheader("Flood indicators")
    fig_scatter = px.scatter(
        filtered_df,
        x="Corrected_Percent_Flooded_Area",
        y="DFSI",
        size="Human_fatality",
        color="Risk_Category",
        hover_name="Dist_Name",
        hover_data=[
            "State_Name",
            "Historical_Risk_Score",
            "Human_injured",
            "Mean_Flood_Duration",
        ],
        title="Flooded area versus district flood susceptibility index",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("District risk ranking table")
    display_df = filtered_df.sort_values(
        "Historical_Risk_Score",
        ascending=False,
    )

    display_columns = [
        "Dist_Name",
        "State_Name",
        "Historical_Risk_Score",
        "Risk_Category",
        "DFSI",
        "Corrected_Percent_Flooded_Area",
        "Human_fatality",
        "Human_injured",
        "Mean_Flood_Duration",
    ]
    st.dataframe(
        display_df[display_columns],
        use_container_width=True,
    )

    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download district risk analysis",
        data=csv_data,
        file_name="flood_district_risk_analysis.csv",
        mime="text/csv",
    )

    st.info(
        "Risk scores are weighted historical indicators created for this "
        "prototype. They are not calibrated probabilities of future flooding. "
        "Use them for exploratory governance prioritization, not official "
        "warnings or evacuation decisions."
    )

# --------------------------------------------------
# FLOOD HAZARD MAP
# --------------------------------------------------

elif module == "Flood Hazard Map":
    st.header("India District-Level Flood Hazard Map")

    st.write(
        "Explore district-level historical flood risk using geographic "
        "boundaries and the historical flood-risk indicators."
    )

    try:
        with st.spinner("Loading district boundaries and flood-risk data..."):
            map_df, matched, unmatched = load_flood_map_data()

        col1, col2, col3 = st.columns(3)
        col1.metric("Geographic districts", f"{len(map_df):,}")
        col2.metric("Matched districts", f"{matched:,}")
        col3.metric("Unmatched districts", f"{unmatched:,}")

        st.info(
            "This map displays historical risk scores, not live flood "
            "conditions or calibrated predictions of future flooding. "
            "Districts without a verified match are left unclassified."
        )

        state_options = ["All States"] + sorted(
            map_df["state_name"].dropna().unique().tolist()
        )

        selected_map_state = st.selectbox(
            "Filter map by state",
            state_options,
            key="flood_map_state",
        )

        map_fig = create_flood_map(map_df, selected_map_state)

        if map_fig is not None:
            st.plotly_chart(map_fig, use_container_width=True)
        else:
            st.warning(
                "No matched district risk data is available for this selection."
            )

        st.subheader("District-level map data")

        display_map_df = map_df[
            [
                "district",
                "state_name",
                "Historical_Risk_Score",
                "Risk_Category",
            ]
        ].copy()

        if selected_map_state != "All States":
            display_map_df = display_map_df[
                display_map_df["state_name"] == selected_map_state
            ]

        display_map_df = display_map_df.sort_values(
            "Historical_Risk_Score",
            ascending=False,
            na_position="last",
        )

        st.dataframe(display_map_df, use_container_width=True)

        map_csv = display_map_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download map district data",
            data=map_csv,
            file_name="india_district_flood_map_data.csv",
            mime="text/csv",
        )

    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Unable to load the flood hazard map: {error}")
# --------------------------------------------------
# GOVERNANCE DECISION SUPPORT
# --------------------------------------------------

elif module == "Governance Decision Support":

    st.header("Disaster Governance Decision Support")

    st.write(
        "Plan an indicative distribution of emergency resources "
        "using district-level historical flood-risk scores."
    )

    if flood_df is None:
        st.error("Flood risk ranking data is unavailable.")
        st.stop()

    st.warning(
        "This is an experimental planning tool. Allocations are based "
        "on historical risk indicators, not live flood conditions. "
        "They must not replace official disaster-management decisions."
    )

    st.subheader("1. Configure resource availability")

    col1, col2 = st.columns(2)

    with col1:
        total_resources = st.number_input(
            "Total available resource units",
            min_value=1,
            max_value=1000000,
            value=1000,
            step=100,
        )

    with col2:
        selected_state = st.selectbox(
            "Select state",
            ["All States"] + sorted(
                flood_df["State_Name"].dropna().unique().tolist()
            ),
            key="governance_state",
        )

    # Filter districts
    planning_df = flood_df.copy()

    if selected_state != "All States":
        planning_df = planning_df[
            planning_df["State_Name"] == selected_state
        ].copy()

    planning_df = planning_df.dropna(
        subset=["Historical_Risk_Score", "Dist_Name", "State_Name"]
    )

    if planning_df.empty:
        st.warning("No district risk data available for this selection.")
        st.stop()

    # Ensure scores are non-negative
    planning_df["Planning_Weight"] = (
        pd.to_numeric(
            planning_df["Historical_Risk_Score"],
            errors="coerce",
        ).fillna(0).clip(lower=0)
    )

    total_weight = planning_df["Planning_Weight"].sum()

    if total_weight <= 0:
        planning_df["Planning_Weight"] = 1
        total_weight = len(planning_df)

    # Calculate proportional allocation
    planning_df["Exact_Allocation"] = (
        planning_df["Planning_Weight"]
        / total_weight
        * int(total_resources)
    )

    planning_df["Recommended_Units"] = (
        planning_df["Exact_Allocation"].apply(int)
    )

    # Distribute remaining units using largest remainder method
    remaining_units = (
        int(total_resources)
        - int(planning_df["Recommended_Units"].sum())
    )

    planning_df["Remainder"] = (
        planning_df["Exact_Allocation"]
        - planning_df["Recommended_Units"]
    )

    remainder_indices = planning_df.sort_values(
        "Remainder",
        ascending=False,
    ).head(remaining_units).index

    planning_df.loc[
        remainder_indices,
        "Recommended_Units"
    ] += 1

    # Calculate allocation percentage
    planning_df["Allocation_Percentage"] = (
        planning_df["Recommended_Units"]
        / int(total_resources)
        * 100
    )

    planning_df = planning_df.sort_values(
        "Historical_Risk_Score",
        ascending=False,
    )

    # Summary metrics
    st.divider()
    st.subheader("2. Resource allocation summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Available resource units",
        f"{int(total_resources):,}",
    )

    col2.metric(
        "Districts considered",
        f"{len(planning_df):,}",
    )

    col3.metric(
        "Units allocated",
        f"{int(planning_df['Recommended_Units'].sum()):,}",
    )

    # Priority districts
    st.divider()
    st.subheader("3. Priority districts")

    top_districts = planning_df.head(10).copy()

    st.dataframe(
        top_districts[
            [
                "Dist_Name",
                "State_Name",
                "Historical_Risk_Score",
                "Risk_Category",
                "Recommended_Units",
                "Allocation_Percentage",
            ]
        ],
        use_container_width=True,
    )

    # Resource allocation chart
    st.divider()
    st.subheader("4. Indicative resource distribution")

    chart_df = planning_df.head(15).sort_values(
        "Recommended_Units",
        ascending=True,
    )

    fig = px.bar(
        chart_df,
        x="Recommended_Units",
        y="Dist_Name",
        orientation="h",
        color="Historical_Risk_Score",
        color_continuous_scale="YlOrRd",
        hover_data=[
            "State_Name",
            "Historical_Risk_Score",
            "Risk_Category",
            "Allocation_Percentage",
        ],
        title="Resource allocation for the 15 highest-risk districts",
        labels={
            "Recommended_Units": "Indicative resource units",
            "Dist_Name": "District",
            "Historical_Risk_Score": "Historical risk score",
        },
    )

    st.plotly_chart(fig, use_container_width=True)

    # Full allocation table
    st.divider()
    st.subheader("5. Complete district allocation plan")

    final_columns = [
        "Dist_Name",
        "State_Name",
        "Historical_Risk_Score",
        "Risk_Category",
        "Recommended_Units",
        "Allocation_Percentage",
    ]

    st.dataframe(
        planning_df[final_columns],
        use_container_width=True,
    )

    # Download allocation
    csv_data = planning_df[final_columns].to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download resource allocation plan",
        data=csv_data,
        file_name="disaster_resource_allocation.csv",
        mime="text/csv",
    )

    st.success(
        "Indicative allocation calculated successfully. "
        "The total allocated units match the available resource quantity."
    )

    st.caption(
        "Method: proportional allocation using historical district "
        "risk scores, with remaining units distributed using the "
        "largest-remainder method. This is a planning scenario, "
        "not an operational deployment recommendation."
    )
    
# --------------------------------------------------
# AI MODEL EXPLAINABILITY
# --------------------------------------------------

elif module == "AI Model Explainability":

    st.header("AI Model Explainability")

    st.write(
        "Understand which input features influence the cyclone "
        "intensity model and review its evaluation metrics."
    )

    st.info(
        "Feature importance represents overall model behaviour. "
        "It does not establish causation or explain an individual "
        "prediction."
    )

    # Load trained model
    try:
        model, features = get_cyclone_model()

        # Extract the underlying Random Forest estimator
        estimator = model

        if isinstance(model, dict):
            estimator = model["model"]

        if not hasattr(estimator, "feature_importances_"):
            st.error(
                "The loaded model does not support feature importance."
            )
            st.stop()

        importances = estimator.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances,
        })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False,
        )

        st.subheader("1. Feature importance")

        st.write(
            "The following chart shows the relative importance "
            "assigned to each input feature by the Random Forest."
        )

        fig = px.bar(
            importance_df.sort_values(
                "Importance",
                ascending=True,
            ),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            title="Cyclone model feature importance",
            labels={
                "Importance": "Relative importance",
                "Feature": "Input feature",
            },
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("2. Feature importance table")

        importance_display = importance_df.copy()
        importance_display["Importance (%)"] = (
            importance_display["Importance"] * 100
        ).round(2)

        st.dataframe(
            importance_display[
                ["Feature", "Importance (%)"]
            ],
            use_container_width=True,
        )

        st.subheader("3. Understanding the model inputs")

        feature_descriptions = {
            "PREV_WIND": "Wind speed in the previous observation.",
            "PREV2_WIND": "Wind speed two observations earlier.",
            "WIND_CHANGE": "Change in wind speed from the previous observation.",
            "LATITUDE": "Recorded cyclone latitude.",
            "LONGITUDE_180": "Recorded cyclone longitude.",
            "LAT_CHANGE": "Change in latitude between observations.",
            "LON_CHANGE": "Change in longitude between observations.",
            "HOURS_SINCE_PREV": "Time elapsed since the previous observation.",
            "MONTH_SIN": "Seasonal month representation using sine.",
            "MONTH_COS": "Seasonal month representation using cosine.",
            "YEAR": "Calendar year of the observation.",
        }

        for _, row in importance_df.iterrows():
            feature = row["Feature"]
            description = feature_descriptions.get(
                feature,
                "Model input feature.",
            )

            st.write(
                f"**{feature}** :- {description} "
                f"Importance: {row['Importance'] * 100:.2f}%"
            )

        st.divider()

        st.subheader("4. Model evaluation metrics")

        metrics_file = (
            ROOT / "outputs" / "cyclone_sequence_metrics.json"
        )

        if metrics_file.exists():
            import json

            with open(metrics_file, "r", encoding="utf-8") as file:
                metrics = json.load(file)

            st.json(metrics)

        else:
            st.warning(
                "Evaluation metrics file not found at "
                "outputs/cyclone_sequence_metrics.json"
            )

        st.subheader("5. Model limitations")

        st.markdown(
            """
            - The model uses historical cyclone observations.
            - Feature importance is a global explanation, not a
              guarantee of causation.
            - The model estimates the next recorded observation.
            - Future cyclone movement is not independently predicted.
            - Performance on historical test data does not guarantee
              accuracy during future cyclones.
            """
        )

    except FileNotFoundError as error:
        st.error(f"Model file not found: {error}")

    except Exception as error:
        st.error(f"Explainability module error: {error}")

# --------------------------------------------------
# CYCLONE MODEL VALIDATION
# --------------------------------------------------

elif module == "Model Validation":

    st.header("Cyclone Model Validation")

    st.write(
        "Evaluate the cyclone intensity model using unseen storms "
        "and compare its performance against a persistence baseline."
    )

    validation_file = (
        ROOT / "outputs" / "cyclone_validation_predictions.csv"
    )

    metrics_file = (
        ROOT / "outputs" / "cyclone_sequence_metrics.json"
    )

    if not validation_file.exists() or not metrics_file.exists():

        st.error(
            "Validation files not found. Run "
            "python backend/train_cyclone_sequence_model.py first."
        )

    else:

        import json

        validation = pd.read_csv(validation_file)

        with open(metrics_file, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        rf = metrics["random_forest"]
        baseline = metrics["persistence_baseline"]

        st.subheader("1. Validation dataset")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Testing observations",
            metrics["testing_observations"],
        )

        col2.metric(
            "Unseen testing storms",
            metrics["testing_storms"],
        )

        col3.metric(
            "Training/testing storm overlap",
            metrics["storm_overlap"],
        )

        st.info(
            "The test set contains storms excluded from model training. "
            "This helps evaluate performance on unseen storms."
        )

        st.divider()

        st.subheader("2. Random Forest vs persistence baseline")

        comparison = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "RÂ²"],
            "Random Forest": [
                rf["MAE"],
                rf["RMSE"],
                rf["R2"],
            ],
            "Persistence baseline": [
                baseline["MAE"],
                baseline["RMSE"],
                baseline["R2"],
            ],
        })

        st.dataframe(
            comparison.round(4),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Lower MAE and RMSE are better. Higher RÂ² is better."
        )

        metric_choice = st.selectbox(
            "Select evaluation metric",
            ["MAE", "RMSE", "RÂ²"],
        )

        metric_key = {
            "MAE": "MAE",
            "RMSE": "RMSE",
            "RÂ²": "R2",
        }[metric_choice]

        comparison_chart = pd.DataFrame({
            "Model": [
                "Random Forest",
                "Persistence baseline",
            ],
            "Score": [
                rf[metric_key],
                baseline[metric_key],
            ],
        })

        fig = px.bar(
            comparison_chart,
            x="Model",
            y="Score",
            color="Model",
            title=f"{metric_choice} comparison",
            text_auto=".3f",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.divider()

        st.subheader("3. Actual vs predicted wind speed")

        fig_actual = px.scatter(
            validation,
            x="ACTUAL_WIND_KNOTS",
            y="PREDICTED_WIND_KNOTS",
            hover_data=["STORM_ID", "DATE_UTC"],
            title="Actual vs Random Forest predicted wind speed",
            labels={
                "ACTUAL_WIND_KNOTS": "Actual wind (knots)",
                "PREDICTED_WIND_KNOTS": "Predicted wind (knots)",
            },
            opacity=0.65,
        )

        min_value = min(
            validation["ACTUAL_WIND_KNOTS"].min(),
            validation["PREDICTED_WIND_KNOTS"].min(),
        )

        max_value = max(
            validation["ACTUAL_WIND_KNOTS"].max(),
            validation["PREDICTED_WIND_KNOTS"].max(),
        )

        fig_actual.add_shape(
            type="line",
            x0=min_value,
            y0=min_value,
            x1=max_value,
            y1=max_value,
            line=dict(
                dash="dash",
                color="red",
            ),
        )

        st.plotly_chart(
            fig_actual,
            use_container_width=True,
        )

        st.caption(
            "Points closer to the diagonal line represent more accurate "
            "predictions."
        )

        st.divider()

        st.subheader("4. Prediction error distribution")

        fig_error = px.histogram(
            validation,
            x="MODEL_ERROR",
            nbins=35,
            title="Distribution of Random Forest prediction errors",
            labels={
                "MODEL_ERROR": "Actual minus predicted wind (knots)",
            },
        )

        st.plotly_chart(
            fig_error,
            use_container_width=True,
        )

        st.caption(
            "Positive errors mean the model underestimated wind speed. "
            "Negative errors mean it overestimated wind speed."
        )

        st.divider()

        st.subheader("5. Compare individual predictions")

        selected_storm = st.selectbox(
            "Select a test storm",
            sorted(
                validation["STORM_ID"].astype(str).unique()
            ),
        )

        storm_data = validation[
            validation["STORM_ID"].astype(str) == selected_storm
        ].copy()

        storm_data["DATE_UTC"] = pd.to_datetime(
            storm_data["DATE_UTC"],
            errors="coerce",
        )

        storm_data = storm_data.sort_values("DATE_UTC")

        fig_storm = px.line(
            storm_data,
            x="DATE_UTC",
            y=[
                "ACTUAL_WIND_KNOTS",
                "PREDICTED_WIND_KNOTS",
                "BASELINE_WIND_KNOTS",
            ],
            markers=True,
            title=f"Wind speed comparison :- {selected_storm}",
            labels={
                "value": "Wind speed (knots)",
                "DATE_UTC": "Observation date",
                "variable": "Series",
            },
        )

        st.plotly_chart(
            fig_storm,
            use_container_width=True,
        )

        st.divider()

        st.subheader("6. Validation interpretation")

        if rf["MAE"] < baseline["MAE"]:
            st.success(
                "Random Forest has lower MAE than the persistence baseline."
            )
        else:
            st.warning(
                "Persistence baseline has lower MAE. Random Forest "
                "does not outperform it on average absolute error."
            )

        if rf["RMSE"] < baseline["RMSE"]:
            st.success(
                "Random Forest has lower RMSE than the persistence baseline."
            )
        else:
            st.warning(
                "Persistence baseline has lower RMSE."
            )

        if rf["R2"] > baseline["R2"]:
            st.success(
                "Random Forest has higher RÂ² than the persistence baseline."
            )
        else:
            st.warning(
                "Persistence baseline has higher RÂ²."
            )

        st.markdown(
            """
            **Important limitations**

            - This is a historical holdout evaluation, not a live cyclone forecast.
            - The model predicts the next recorded observation.
            - Test storms were excluded from training, but the evaluation
              uses one random group split.
            - Historical data quality and reporting intervals affect results.
            - The model should not be used as a standalone emergency warning system.
            """
        )

        st.divider()

        st.subheader("7. Download validation results")

        st.download_button(
            "Download prediction results (CSV)",
            data=validation.to_csv(index=False),
            file_name="cyclone_validation_results.csv",
            mime="text/csv",
        )

        st.dataframe(
            validation.head(20),
            use_container_width=True,
        )


elif module == "River Network Intelligence":
    import json
    import networkx as nx

    st.header("River Network Intelligence")
    st.write(
        "Explore the downstream river connectivity between district features "
        "using the HydroRIVERS NEXT_DOWN network."
    )

    nodes_path = Path("outputs/final_river_network_nodes.csv")
    edges_path = Path("outputs/final_river_network_edges.csv")
    summary_path = Path("outputs/final_river_network_summary.json")

    if not all(x.exists() for x in [nodes_path, edges_path, summary_path]):
        st.error("River network output files are missing. Build the final river graph first.")
    else:
        nodes = pd.read_csv(nodes_path)
        edges = pd.read_csv(edges_path)

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        c1, c2, c3 = st.columns(3)
        c1.metric("District features", f"{summary.get('district_feature_nodes', len(nodes)):,}")
        c2.metric("River links", f"{summary.get('individual_river_links', 0):,}")
        c3.metric("Directed connections", f"{summary.get('unique_directed_district_connections', len(edges)):,}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Weak components", summary.get("weakly_connected_components", "N/A"))
        c5.metric("Strong components", summary.get("strongly_connected_components", "N/A"))
        c6.metric("Self-loops", summary.get("self_loops", "N/A"))

        st.divider()
        st.subheader("District connectivity")

        node_name = "district"
        if node_name not in nodes.columns:
            node_name = next((c for c in ["district_name", "name"] if c in nodes.columns), None)

        source_col = "upstream_node" if "upstream_node" in edges.columns else None
        target_col = "downstream_node" if "downstream_node" in edges.columns else None

        if source_col and target_col:
            edges["Upstream district"] = edges["upstream_district"].astype(str)
            edges["Downstream district"] = edges["downstream_district"].astype(str)

            district_options = sorted(set(edges["Upstream district"]) | set(edges["Downstream district"]))
            selected = st.selectbox("Select a district", district_options)

            upstream = edges.loc[edges["Downstream district"] == selected, "Upstream district"].drop_duplicates().tolist()
            downstream = edges.loc[edges["Upstream district"] == selected, "Downstream district"].drop_duplicates().tolist()

            left, right = st.columns(2)
            with left:
                st.markdown("**Upstream-connected districts**")
                st.write(upstream if upstream else "No upstream district connection recorded.")
            with right:
                st.markdown("**Downstream-connected districts**")
                st.write(downstream if downstream else "No downstream district connection recorded.")

            st.subheader("Directed river network")
            st.caption("Arrows indicate the direction represented by HydroRIVERS NEXT_DOWN.")
            graph = nx.DiGraph()
            for _, row in edges.iterrows():
                graph.add_edge(str(row[source_col]), str(row[target_col]))

            if graph.number_of_nodes() <= 700:
                import plotly.graph_objects as go
                pos = nx.spring_layout(graph, seed=42, k=0.35, iterations=30)
                edge_x, edge_y = [], []
                for u, v in graph.edges():
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y, mode="lines",
                    line=dict(width=0.6, color="#9CA3AF"),
                    hoverinfo="none"
                )
                node_x = [pos[n][0] for n in graph.nodes()]
                node_y = [pos[n][1] for n in graph.nodes()]
                node_trace = go.Scatter(
                    x=node_x, y=node_y, mode="markers",
                    marker=dict(size=6, color="#D4A72C"),
                    text=[str(n) for n in graph.nodes()],
                    hovertemplate="District feature: %{text}<extra></extra>"
                )
                fig = go.Figure(data=[edge_trace, node_trace])
                fig.update_layout(
                    height=650, showlegend=False,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis=dict(visible=False), yaxis=dict(visible=False)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Network is too large for this interactive overview.")

            st.subheader("Directed connection records")
            st.dataframe(
                edges[["Upstream district", "Downstream district"]].drop_duplicates(),
                use_container_width=True
            )
        else:
            st.error("Expected upstream/downstream district columns were not found.")

        st.divider()
        st.subheader("Method and limitations")
        st.markdown(
            """
            - Direction is derived from the HydroRIVERS `NEXT_DOWN` attribute.
            - The graph represents river connectivity between district features.
            - Connectivity does not prove that floodwater will reach a district.
            - This graph does not calculate flood arrival time or predict flood severity.
            - The network is based on the supplied district boundary dataset and extracted HydroRIVERS data.
            """
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "Disaster Governance AI | Cyclone and Flood Intelligence | "
    "Hackathon Research Prototype"
)
