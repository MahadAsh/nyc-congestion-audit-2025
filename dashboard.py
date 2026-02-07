import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

st.set_page_config(layout="wide", page_title="NYC Congestion Audit 2025")

# --- HEADER ---
st.title("2025 NYC Congestion Pricing Audit")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        leakage = pd.read_csv("./outputs/leakage_audit.csv")
        velocity = pd.read_csv("./outputs/velocity_heatmap.csv")
        weather = pd.read_csv("./outputs/weather_elasticity.csv")
        economics = pd.read_csv("./outputs/economics.csv")
        ghost = pd.read_csv("./outputs/ghost_audit.csv")
        return leakage, velocity, weather, economics, ghost
    except FileNotFoundError:
        return None, None, None, None, None

leakage_df, velocity_df, weather_df, economics_df, ghost_df = load_data()

if leakage_df is None:
    st.error("Data not found! Please run 'python pipeline.py' first.")
    st.stop()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["The Flow (Velocity)", "The Economics", "The Rain Tax", "Audit & Fraud"])

# --- TAB 1: VELOCITY HEATMAP ---
with tab1:
    st.header("Did the toll speed up traffic?")
    st.markdown("Average speed (MPH) inside the Congestion Zone by Day and Hour.")
    
    # Pivot data for heatmap
    heatmap_data = velocity_df.pivot(index='weekday', columns='hour', values='avg_speed')
    
    fig = px.imshow(
        heatmap_data, 
        labels=dict(x="Hour of Day", y="Day of Week", color="Speed (MPH)"),
        color_continuous_scale="RdYlGn", # Red = Slow, Green = Fast
        title="Congestion Velocity Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: ECONOMICS (TIPS vs SURCHARGE) ---
with tab2:
    st.header("Is it fair to drivers?")
    st.markdown("Hypothesis: Higher tolls reduce the disposable income passengers leave for drivers.")
    
    # Dual Axis Plot
    fig = go.Figure()
    
    # Bar: Surcharge
    fig.add_trace(go.Bar(
        x=economics_df['month'], 
        y=economics_df['avg_surcharge'], 
        name='Avg Surcharge ($)',
        marker_color='indianred'
    ))
    
    # Line: Tip Amount
    fig.add_trace(go.Scatter(
        x=economics_df['month'], 
        y=economics_df['avg_tip_amt'], 
        name='Avg Tip ($)',
        yaxis='y2',
        line=dict(color='royalblue', width=4)
    ))
    
    fig.update_layout(
        title="Surcharges vs. Driver Tips (2025)",
        yaxis=dict(title="Avg Surcharge Paid ($)"),
        yaxis2=dict(title="Avg Tip Received ($)", overlaying='y', side='right'),
        legend=dict(x=0.1, y=1.1, orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: WEATHER ELASTICITY ---
with tab3:
    st.header("The Rain Tax")
    
    corr = weather_df['trip_count'].corr(weather_df['precipitation_mm'])
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Rain Elasticity Score", f"{corr:.4f}")
        if abs(corr) < 0.3:
            st.info("Inelastic Demand: People keep riding taxis even when it rains.")
        else:
            st.warning("Elastic Demand: Rain significantly changes ridership.")
            
    with col2:
        fig = px.scatter(
            weather_df, 
            x="precipitation_mm", 
            y="trip_count", 
            trendline="ols",
            title="Correlation: Daily Rain vs. Trip Volume"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: AUDIT & FRAUD ---
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ghost Trip Detection")
        st.markdown("Trips with impossible physics (Speed > 65MPH or 0 distance).")
        st.dataframe(ghost_df, use_container_width=True)
        
    with col2:
        st.subheader("Surcharge Leakage")
        st.markdown("Top Locations where trips end in the zone but pay **$0 surcharge**.")
        st.bar_chart(leakage_df.set_index("pickup_loc")["missing_surcharge_count"])