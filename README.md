# E-Commerce Business Intelligence & Customer Experience Analytics Platform

An end-to-end Business Intelligence and Analytics platform built with
Python and Streamlit using the Olist Brazilian E-Commerce Public
Dataset.

The platform transforms raw e-commerce data into interactive dashboards
covering sales performance, products and categories, delivery
operations, customer experience, payments, regional performance, seller
intelligence, and machine-learning-based delivery-delay risk prediction.

------------------------------------------------------------------------

## 📌 Project Overview

The objective of this project is to build a practical e-commerce
analytics platform that helps analyze business performance and customer
experience across multiple dimensions.

The application provides interactive analysis of:

-   Sales and order performance
-   Product and category performance
-   Delivery and operational efficiency
-   Customer reviews and experience
-   Payment behavior
-   Regional performance
-   Seller performance
-   Delivery-delay risk

The project combines traditional Business Intelligence analytics with a
machine-learning component for delivery-delay prediction.

------------------------------------------------------------------------

## 🎯 Business Objectives

The platform is designed to answer practical business questions such as:

-   How are sales and orders performing over time?
-   Which product categories generate the highest sales value?
-   Which categories have the highest item volumes?
-   How do product prices and freight values vary?
-   How does delivery performance change across time and regions?
-   What proportion of orders experience delivery delays?
-   How does customer satisfaction vary?
-   Which payment methods are most commonly used?
-   Which regions generate the most business activity?
-   How do sellers compare in terms of sales and item activity?
-   Can delivery-delay risk be estimated before the final delivery
    outcome is known?

------------------------------------------------------------------------

# 📊 Dataset

This project uses the **Olist Brazilian E-Commerce Public Dataset**.

The original dataset consists of multiple relational tables covering
different parts of the e-commerce process.

### Source tables

The project works with the following Olist datasets:

-   `olist_customers_dataset.csv`
-   `olist_geolocation_dataset.csv`
-   `olist_order_items_dataset.csv`
-   `olist_order_payments_dataset.csv`
-   `olist_order_reviews_dataset.csv`
-   `olist_orders_dataset.csv`
-   `olist_products_dataset.csv`
-   `olist_sellers_dataset.csv`
-   `product_category_name_translation.csv`

These datasets were cleaned, transformed, validated, and organized into
analytical datasets used by the application.

------------------------------------------------------------------------

# 🏗️ Data Architecture

A major design decision in this project is maintaining **two separate
analytical grains**.

The application does not permanently flatten order-level and item-level
data into one table.

This prevents duplicated records from incorrectly inflating order-level
KPIs.

------------------------------------------------------------------------

## 1. Order-Level Dataset

### `master_df`

**Grain:** One row per order.

The order-level dataset is used for:

-   Order KPIs
-   Sales/order-level metrics
-   Delivery analysis
-   Customer experience
-   Reviews
-   Payments
-   Regional analysis
-   Order status analysis

### Validated shape

``` text
99,441 rows × 42 columns
```

------------------------------------------------------------------------

## 2. Order-Item-Level Dataset

### `order_item_master`

**Grain:** One row per order item.

The item-level dataset is used for:

-   Product analysis
-   Category analysis
-   Seller analysis
-   Item-level sales
-   Item quantity
-   Freight analysis
-   Product/category trends

### Validated shape

``` text
112,650 rows × 35 columns
```

------------------------------------------------------------------------

## ⚠️ Why Two Grains?

An order can contain multiple items.

If item-level records are incorrectly merged into an order-level
dataset, order-level metrics such as:

-   order count
-   customer count
-   delivery metrics
-   payment metrics

can become duplicated.

Therefore, the application preserves the correct grain for each
analytical use case.

------------------------------------------------------------------------

# 🛠️ Technology Stack

  Technology     Purpose
  -------------- --------------------------------------------------
  Python         Data processing, analytics and application logic
  Pandas         Data manipulation and transformation
  NumPy          Numerical operations
  Plotly         Interactive visualizations
  Streamlit      Interactive dashboard application
  Scikit-learn   Machine learning
  SQL            Data analysis and transformation concepts
  Git            Version control
  GitHub         Project repository and sharing
  PowerShell     Local development and validation

------------------------------------------------------------------------

## 📁 Repository Structure

```text
Olist-Ecommerce-Analytics/
│
├── 📁 01_data/
│   ├── olist_master_cleaned.csv
│   ├── olist_order_item_master.csv
│   └── ml_delay_dataset.csv
│
├── 📁 02_pages/
│   ├── 01_overview.py
│   ├── 02_products_categories.py
│   ├── 03_delivery_operations.py
│   ├── 04_customer_experience.py
│   ├── 05_payments.py
│   ├── 06_regional.py
│   ├── 07_sellers.py
│   └── 08_ml_delay_predictor.py
│
├── 📁 03_src/
│   ├── data_loader.py
│   ├── filters.py
│   ├── kpis.py
│   ├── charts.py
│   └── ml_module.py
│
├── 📁 04_validation/
│   ├── _validate_app.py
│   ├── _validate_charts.py
│   ├── _validate_customer.py
│   ├── _validate_delivery.py
│   ├── _validate_filters.py
│   ├── _validate_kpis.py
│   ├── _validate_loader.py
│   ├── _validate_overview.py
│   ├── _validate_payments.py
│   ├── _validate_products.py
│   └── _validate_regional.py
│
├── 📁 05_notebook/
│   ├── BrojoMohanDutta_Olist_Ecommerce_Analytics_AICTE_IBM.ipynb
│   └── [additional notebook/source file]
│
├── 📁 06_documentation/
│   ├── BrojoMohanDutta_Olist_Ecommerce_Analytics_AICTE_IBM_Report.docx
│   └── [additional documentation file]
│
├── 📄 app.py
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 _inventory.py
├── 📄 .gitignore
└── 📄 .gitattributes
```
------------------------------------------------------------------------

# 📈 Dashboard Pages

## 1. Executive Overview

The Executive Overview provides a high-level view of the e-commerce
business.

It focuses on:

-   Overall sales performance
-   Order volume
-   Business KPIs
-   Time-based trends
-   High-level business patterns

This page is designed as the primary management overview of the
platform.

------------------------------------------------------------------------

## 2. Products & Categories

This page analyzes product and category performance using the order-item
analytical grain.

Key analysis includes:

-   Sales value by category
-   Items sold by category
-   Average item price
-   Average freight
-   Monthly item sales trends
-   Category price versus volume
-   Category-level detail

This helps identify high-volume and high-value product categories.

------------------------------------------------------------------------

## 3. Delivery & Operations

This page focuses on order fulfillment and delivery performance.

Key areas include:

-   Delivery time
-   Estimated versus actual delivery
-   Delayed orders
-   Delivery performance trends
-   Operational patterns
-   Delivery-related KPIs

The analysis is designed to identify operational patterns that may
affect customer experience.

------------------------------------------------------------------------

## 4. Customer Experience

This page focuses on customer reviews and experience-related metrics.

Key analysis includes:

-   Review scores
-   Review-score distribution
-   Customer experience patterns
-   Delivery and customer-experience relationships
-   Customer satisfaction metrics

The objective is to understand how customers experienced their orders.

------------------------------------------------------------------------

## 5. Payments

The Payments page analyzes payment behavior across orders.

It includes analysis of:

-   Payment methods
-   Payment values
-   Payment installments
-   Payment distributions
-   Payment trends

This provides insight into how customers complete their purchases.

------------------------------------------------------------------------

## 6. Regional Analysis

The Regional Analysis page examines geographic patterns in the
marketplace.

It includes analysis of:

-   Customer locations
-   Seller/customer regions
-   State-level performance
-   Regional order activity
-   Regional sales patterns

This helps identify geographic differences in marketplace activity.

------------------------------------------------------------------------

## 7. Seller Intelligence

Seller Intelligence analyzes seller performance using the order-item
analytical grain.

Key analysis includes:

-   Seller sales value
-   Items sold
-   Seller activity
-   Freight-related metrics
-   Top sellers by sales value
-   Top sellers by item activity

Seller identifiers are based on the original Olist seller IDs.

The source dataset does not provide business names for these seller IDs,
so the platform does not invent seller names.

------------------------------------------------------------------------

## 8. Delay Predictor

The Delay Predictor is the machine-learning component of the platform.

It estimates the probability that an order may experience a delivery
delay.

### Prediction target

``` text
is_delayed
```

The model uses information that can be available before the final
delivery outcome.

Post-outcome information is excluded from the prediction features to
reduce data leakage.

### Dataset used for modeling

``` text
Labeled orders:     96,470
Unlabeled orders:    2,971
Delay rate:          ~8.11%
```

The application provides:

-   Delay probability
-   Prediction outcome
-   Interpreted risk information

The prediction is a statistical estimate and should not be interpreted
as a guaranteed future outcome.

------------------------------------------------------------------------

# 🔎 Interactive Filtering

The dashboard provides centralized filtering functionality.

The default filter configuration is:

``` text
Date Start:       2017-01-01
Date End:         2018-10-17
Order Status:     delivered
Include 2016:     OFF
```

Additional filters are available depending on the dashboard page and its
analytical purpose.

The filtering system is designed to preserve the appropriate data grain
for each page.

------------------------------------------------------------------------

# 📐 Analytical Principles

## Two-Grain Architecture

Order-level and item-level datasets are maintained separately.

This prevents duplicated item records from inflating order-level
metrics.

------------------------------------------------------------------------

## Data Quality

The project includes:

-   Data cleaning
-   Data transformation
-   Data validation
-   KPI validation
-   Chart validation
-   Application validation
-   Regression testing
-   Runtime verification

------------------------------------------------------------------------

## Business-Focused Analytics

The dashboards are designed around practical business questions rather
than only presenting raw descriptive statistics.

------------------------------------------------------------------------

## No Unsupported Profit Calculation

The Olist dataset does not provide explicit Cost of Goods Sold (COGS).

Therefore, this project does **not** treat:

``` text
Price - Freight = Profit
```

as true business profit.

Sales value, freight, and other available metrics are reported using
their actual business meaning.

------------------------------------------------------------------------

## Correlation Does Not Imply Causation

Observed relationships between variables are treated as associations.

The project does not automatically interpret statistical relationships
as causal effects.

------------------------------------------------------------------------

# 🧪 Validation & Testing

The project contains dedicated validation scripts for major application
components.

Validated areas include:

-   Application integrity
-   Chart functions
-   Customer Experience
-   Delivery & Operations
-   Filters
-   KPIs
-   Data Loader
-   Executive Overview
-   Payments
-   Products & Categories
-   Regional Analysis

### Final automated regression

All available automated validation suites passed successfully during
final regression testing.

The project also completed a final Streamlit runtime launch check, with
the dashboard pages loading successfully.

------------------------------------------------------------------------

# 🚀 Installation & Setup

## 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Olist-Ecommerce-Analytics
```

Replace:

``` text
<YOUR_GITHUB_REPOSITORY_URL>
```

with the final GitHub repository URL.

------------------------------------------------------------------------

## 2. Create a virtual environment

``` bash
python -m venv .venv
```

------------------------------------------------------------------------

## 3. Activate the environment

### Windows PowerShell

``` powershell
.venv\Scripts\Activate.ps1
```

------------------------------------------------------------------------

## 4. Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 5. Run the Streamlit application

``` bash
streamlit run app.py
```

The application will open in the browser.

------------------------------------------------------------------------

# 📋 Requirements

All Python dependencies required by the application are listed in:

``` text
requirements.txt
```

The project is designed to run locally using Python and the packages
specified in the requirements file.

------------------------------------------------------------------------

# ⭐ Project Highlights

-   End-to-end e-commerce analytics platform
-   Interactive Streamlit dashboard
-   Two-grain analytical architecture
-   99,441 order-level records
-   112,650 order-item records
-   Interactive Plotly charts
-   Centralized dashboard filtering
-   Business KPI framework
-   Product and category analytics
-   Delivery and operations analytics
-   Customer experience analytics
-   Payment analytics
-   Regional analysis
-   Seller intelligence
-   Machine-learning delivery-delay predictor
-   Data leakage controls
-   Automated validation framework
-   Final runtime testing
-   GitHub-ready project

------------------------------------------------------------------------

# ⚠️ Analytical Limitations

This project is based on a public historical dataset and therefore has
several limitations.

-   The dataset represents Olist marketplace activity rather than the
    entire Brazilian e-commerce market.
-   Historical patterns should not automatically be treated as current
    market conditions.
-   Seller IDs do not provide business names.
-   The dataset does not contain explicit COGS, so true profit cannot be
    calculated.
-   Machine-learning predictions represent estimated probabilities, not
    guaranteed outcomes.
-   Observational relationships should not automatically be interpreted
    as causal effects.

------------------------------------------------------------------------

# 👤 Author

**Brojo Mohan Dutta**

B.Sc. (Honours) Mathematics\
Serampore College, University of Calcutta

### Focus

Data Analytics \| Business Intelligence \| SQL \| Python \| Statistics
\| Data Visualization \| Machine Learning

------------------------------------------------------------------------

## ⭐ Project

**E-Commerce Business Intelligence & Customer Experience Analytics
Platform**

Built as a practical end-to-end data analytics and business intelligence
project using the Olist Brazilian E-Commerce Public Dataset.
