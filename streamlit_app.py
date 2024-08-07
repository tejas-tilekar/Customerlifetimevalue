import streamlit as st
import pandas as pd
import lifetimes
import numpy as np
import altair as alt
import warnings
from sklearn.cluster import KMeans

# Set a random seed for reproducibility
np.random.seed(42)
warnings.filterwarnings("ignore")

# Streamlit app title and description
st.markdown(""" # Customer Lifetime Prediction App

Upload the RFM data and get your customer lifetime prediction on the fly!
""")

st.image("https://www.retently.com/wp-content/uploads/2017/02/customer-lifetime-value.png", use_column_width=True)

# File uploader for the RFM data
data = st.file_uploader("File Uploader")

# Sidebar content
st.sidebar.image("http://logok.org/wp-content/uploads/2014/06/City-of-Melbourne-logo-M.png", width=120)
st.sidebar.markdown(""" **Project by Anshu Lalwani & Tejas Tilekar** """)

st.sidebar.title("Input Features :pencil:")

# Slider for the number of days
days = st.sidebar.slider("Select The No. Of Days", min_value=1, max_value=365, step=1, value=30)

# Slider for the profit margin
profit = st.sidebar.slider("Select the Profit Margin", min_value=0.01, max_value=0.09, step=0.01, value=0.05)

# Display selected input features in the sidebar
slider_data = {
    "Days": days,
    "Profit": profit
}

st.sidebar.markdown("""
### Selected Input Features :page_with_curl:
""")
features = pd.DataFrame(slider_data, index=[0])
st.sidebar.write(features)

st.sidebar.markdown("""
Before uploading the file, please select the input features first.

Also, please make sure the columns are in proper format. For reference, you can download the [dummy data](https://github.com/tejas-tilekar/Customer/blob/main/sample_file.csv).

**Note:** Only use "CSV" file.
""")

# Function to load and process the data
def load_data(data, days, profit):
    input_data = pd.read_csv(data)
    input_data = pd.DataFrame(input_data.iloc[:, 1:])

    # Pareto/NBD model fitting
    pareto_model = lifetimes.ParetoNBDFitter(penalizer_coef=0.0)
    pareto_model.fit(input_data["frequency"], input_data["recency"], input_data["T"])
    input_data["p_not_alive"] = 1 - pareto_model.conditional_probability_alive(input_data["frequency"], input_data["recency"], input_data["T"])
    input_data["p_alive"] = pareto_model.conditional_probability_alive(input_data["frequency"], input_data["recency"], input_data["T"])
    input_data["predicted_purchases"] = pareto_model.conditional_expected_number_of_purchases_up_to_time(days, input_data["frequency"], input_data["recency"], input_data["T"])

    # Gamma-Gamma model fitting
    input_data = input_data[(input_data["frequency"] > 0) & (input_data["monetary_value"] > 0)]
    ggf_model = lifetimes.GammaGammaFitter(penalizer_coef=0.0)
    ggf_model.fit(input_data["frequency"], input_data["monetary_value"])
    input_data["expected_avg_sales_"] = ggf_model.conditional_expected_average_profit(input_data["frequency"], input_data["monetary_value"])
    input_data["predicted_clv"] = ggf_model.customer_lifetime_value(pareto_model, input_data["frequency"], input_data["recency"], input_data["T"], input_data["monetary_value"], time=days, freq='D', discount_rate=0.01)
    input_data["profit_margin"] = input_data["predicted_clv"] * profit

    # Reset index
    input_data = input_data.reset_index(drop=True)

    # K-Means clustering
    col = ["predicted_purchases", "expected_avg_sales_", "predicted_clv", "profit_margin"]
    new_df = input_data[col]
    
    # Add check for data size
    if len(new_df) > 0:
        k_model = KMeans(n_clusters=4, init="k-means++", max_iter=1000).fit(new_df)
        labels = k_model.labels_
        input_data["Labels"] = pd.Series(labels).map({0: "Low", 1: "V_High", 2: "Medium", 3: "High"})
    else:
        input_data["Labels"] = "No Data"

    # Display the input data
    st.write(input_data)

    # Add a count bar chart
    fig = alt.Chart(input_data).mark_bar().encode(
        y="Labels:N",
        x="count(Labels):Q"
    )

    # Add annotations to the chart
    text = fig.mark_text(
        align="left",
        baseline="middle",
        dx=3
    ).encode(
        text="count(Labels):Q"
    )

    chart = fig + text

    # Display the chart
    st.altair_chart(chart, use_container_width=True)

    # Button to download the result
    if st.button("Download"):
        st.write("Successfully Downloaded!!! Please Check Your Default Download Location... :smile:")
        return input_data.to_csv("customer_lifetime_prediction_result.csv")

# Call the function if data is uploaded
if data is not None:
    st.markdown("""
    ## Customer Lifetime Prediction Result :bar_chart:
    """)
    load_data(data, days, profit)
else:
    st.text("Please Upload the CSV File")
