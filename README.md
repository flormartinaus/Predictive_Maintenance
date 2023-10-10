# Predictive_Maintenance

##Predictive Maintenance for Device Failures

##Background and Purpose

Organizations that rely on a multitude of devices transmitting sensor readings seek an efficient solution to predict impending failures. The implementation of a predictive maintenance model can offer significant benefits, such as cost savings and reduced downtime, compared to traditional time-based or reactive maintenance approaches.

The primary objective of this project is to develop a predictive maintenance solution that not only identifies potential device failures but also minimizes both false positives and false negatives. By proactively addressing issues before they escalate, the organization aims to enhance operational efficiency and device reliability.

##Exploratory Data Analysis
The initial phase of this project involves exploratory data analysis (EDA) to gain insights into the dataset and inform subsequent modeling. Below are key steps and findings from the EDA process:

##Data Loading and Preprocessing:

The dataset is loaded and inspected for data completeness and duplicate entries.
Summary statistics reveal that there are no missing entries, and duplicate columns (e.g., 'metric8') are identified and removed.


##Data Overview:

The dataset comprises over 124,000 entries from 1,169 unique devices, providing a substantial dataset for analysis.

##Class Imbalance:

An initial examination of the 'failure' variable indicates a class imbalance, with device failures being relatively rare events.

##Statistical Test:

A chi-squared test for independence is performed to assess the relationship between 'failure' and 'metric1.' The results indicate no significant relationship between these variables.

##Time Series Analysis:

A time series analysis is conducted to visualize the total number of device failures per month. A cyclic pattern is observed, suggesting a potential four-month cycle, although the exact cause remains uncertain.

##Device Failures Analysis:

Devices with multiple failures are investigated, revealing that only a small fraction of devices (approximately 9%) exhibited recurring failure states. The impact of this observation on the predictive model is considered negligible, leading to the decision not to include device IDs as predictor variables.

##Correlation Analysis:

A correlation matrix is calculated and visualized to explore relationships between numeric variables. The heatmap provides insights into variable dependencies and interactions.

##Interactive Visualization:

An interactive bar chart is created using Plotly Express, allowing for a more engaging exploration of device failures over time. Customizations are applied to improve readability.
 
