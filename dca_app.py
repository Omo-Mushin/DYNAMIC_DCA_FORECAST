import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io

# -------------------------------
# CLEANING AND FORECAST LOGIC
# -------------------------------
def clean_excel_data(df):
    df = df.dropna(subset=["Well Name", "Year", "Oil/Gas", "Potential (bopd)/ (MMscf/d)"]).copy()
    df.rename(columns={"Year": "Quarter"}, inplace=True)
    df["Well Name"] = df["Well Name"].str.strip()
    df["Oil/Gas"] = df["Oil/Gas"].str.upper().str.strip()

    def clean_potential(val):
        if isinstance(val, str) and "MMscf" in val:
            return float(val.split()[0]) * 1000
        return float(val)

    df["Potential"] = df["Potential (bopd)/ (MMscf/d)"].apply(clean_potential)
    return df

def quarter_to_date(quarter_str):
    qmap = {'Q1': 1, 'Q2': 4, 'Q3': 7, 'Q4': 10}
    tokens = quarter_str.replace('-', ' ').split()
    year = None
    quarter = None
    for token in tokens:
        if token.upper() in qmap:
            quarter = token.upper()
        elif token.isdigit() and len(token) == 4:
            year = int(token)
    if quarter and year:
        return datetime(year, qmap[quarter], 1)
    else:
        raise ValueError(f"Could not parse quarter string: {quarter_str}")

def exponential_decline(q0, decline_rate, plateau_months, total_months=120):
    monthly_decline = 1 - (1 - decline_rate)**(1/12)
    return [q0 if m < plateau_months else q0 * (1 - monthly_decline)**(m - plateau_months)
            for m in range(total_months)]

def generate_forecast(df):
    forecasts = []
    for _, row in df.iterrows():
        start_date = quarter_to_date(row["Quarter"])
        fluid = row["Oil/Gas"]
        potential = row["Potential"]
        decline_rate = 0.20 if fluid == "GAS" else 0.15
        plateau_months = 6 if potential < 600 else 24

        production = exponential_decline(potential, decline_rate, plateau_months)
        dates = [start_date + timedelta(days=30.44 * i) for i in range(120)]

        forecasts.append(pd.DataFrame({
            "Well Name": row["Well Name"],
            "Date": dates,
            "Production": production,
            "Fluid Type": fluid,
            "Initial Potential": potential
        }))
    return pd.concat(forecasts, ignore_index=True)

# -------------------------------
# PLOTTING FUNCTIONS
# -------------------------------
def plot_monthly_bars(forecast, selected_years):
    forecast["Month"] = forecast["Date"].dt.to_period("M")
    monthly = forecast[forecast["Date"].dt.year.isin(selected_years)]
    months = monthly["Month"].unique()

    n_cols = 2
    n_rows = (len(months) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4.5))
    axes = axes.flatten()

    for i, month in enumerate(months):
        ax = axes[i]
        df_m = monthly[monthly["Month"] == month]
        prod = df_m.groupby('Well Name')["Production"].sum().reset_index()
        prod.set_index("Well Name").plot(kind="bar", stacked=True, ax=ax, colormap='tab20')
        ax.set_title(f"{month.strftime('%b %Y')}")
        ax.set_ylabel("Production")
        ax.set_xlabel("Well")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout(h_pad=2.5)
    st.pyplot(fig)

def plot_yearly_summary(forecast, selected_years):
    df = forecast.copy()
    df["Year"] = df["Date"].dt.year
    yearly = df[df["Year"].isin(selected_years)]
    year_prod = yearly.groupby(["Year", "Well Name"])["Production"].sum().unstack(fill_value=0)
    cumulative = year_prod.sum(axis=1).cumsum()
    x = np.arange(len(year_prod.index))

    fig, ax1 = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(x))
    for well in year_prod.columns:
        ax1.bar(x, year_prod[well], bottom=bottom, label=well)
        bottom += year_prod[well].values

    ax1.set_xticks(x)
    ax1.set_xticklabels(year_prod.index)
    ax1.set_ylabel("Annual Production")
    ax1.legend(title="Well", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative.values, color='black', linestyle='--', marker='o', linewidth=2, label='Cumulative')
    ax2.set_ylabel("Cumulative Production")
    ax2.legend(loc='upper center')
    st.pyplot(fig)

# -------------------------------
# STREAMLIT APP
# -------------------------------
st.set_page_config(layout="wide")
st.title("📊 Dynamic Decline Curve Analysis (DCA) Forecasting App")

uploaded = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded:
    df_input = pd.read_excel(uploaded)
    cleaned_df = clean_excel_data(df_input)
    forecast_df = generate_forecast(cleaned_df)
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

    # Sidebar filters
    st.sidebar.header("🔍 Filter Options")
    unique_years = sorted(forecast_df["Date"].dt.year.unique())
    selected_years = st.sidebar.multiselect("Select Year(s)", unique_years, default=unique_years[:1])

    all_wells = sorted(forecast_df["Well Name"].unique())
    selected_wells = st.sidebar.multiselect("Select Wells to Display", all_wells, default=all_wells)

    filtered_forecast_df = forecast_df[forecast_df["Well Name"].isin(selected_wells)]

    st.subheader("📅 Yearly Summary")
    plot_yearly_summary(filtered_forecast_df, selected_years)

    st.subheader("📊 Monthly Production by Well")
    plot_monthly_bars(filtered_forecast_df, selected_years)

    # Full Forecast View
    st.subheader("📅 📈 Full Forecast Across All Years")
    with st.expander("Show Full Forecast", expanded=False):
        view_type = st.radio("View Type", ["Grid Line Plot (Per Well)", "Stacked Yearly Bars + Cumulative"])

        if view_type == "Grid Line Plot (Per Well)":
            well_names = filtered_forecast_df['Well Name'].unique()
            n_wells = len(well_names)
            n_cols = 3
            n_rows = (n_wells + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3), sharex=False)
            axes = axes.flatten()

            for i, well in enumerate(well_names):
                ax = axes[i]
                well_data = filtered_forecast_df[filtered_forecast_df['Well Name'] == well]
                color = 'orange' if well_data['Fluid Type'].iloc[0] == 'OIL' else 'green'
                ax.plot(well_data['Date'], well_data['Production'], color=color)
                ax.set_title(well, fontsize=10)
                ax.set_xlabel("Date")
                ax.set_ylabel("Production")
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True)

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            st.pyplot(fig)

        else:
            plot_yearly_summary(filtered_forecast_df, forecast_df["Date"].dt.year.unique())

    # Download forecast results
    output = io.BytesIO()
    forecast_df.to_excel(output, index=False)
    st.download_button("📥 Download Forecast Excel", output.getvalue(), file_name="forecast_output.xlsx")

else:
    st.info("👆 Upload your Excel file to begin.")
