# Data Analysis of Customer Sales

**Data Analysis of Customer Sales** is a Django-based web application designed for data analysis, visualization, and reporting. This application allows users to upload CSV files, perform data cleansing, visualize patterns, and export insights in both Excel and PDF formats. The application is optimized with neumorphic UI design elements and includes tools for effective data inspection and transformation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Exporting Reports](#exporting-reports)
- [License](#license)

## Overview

The **Data Analysis of Customer Sales** project supports the following functionalities:
1. Uploading and previewing data.
2. Cleaning and transforming data by handling missing values, normalizing numeric fields, and formatting dates.
3. Providing visual insights like correlation matrices and trend lines using Chart.js and Seaborn.
4. Generating and exporting detailed reports in both PDF and Excel formats.

## Features

- **CSV File Upload**: Upload CSV files and preview data directly within the app.
- **Data Cleansing**: Handles missing values, fills nulls based on specified strategies, and normalizes data.
- **Data Transformation**: Processes dates, removes duplicates, and applies numeric normalization.
- **Exploratory Data Analysis (EDA)**: Interactive visualizations, including amount spent by product category, correlation matrix, and trend line visualizations.
- **Detailed Insights**: Displays metrics like total customers, average age, and popular categories.
- **Report Export**: Export data analysis results as Excel or PDF files, complete with data tables, charts, and summary insights.

## Directory Structure

```plaintext
data_analysis_of_customer_sales/
├── analysis/                               # Core app for data analysis
│   ├── static/                             # Static assets for styling and JavaScript
│   │   └── analysis/css/styles.css         # CSS file for neumorphic styling
│   ├── templates/                          # HTML templates for views
│   │   └── analysis/
│   │       ├── analysis.html               # Data analysis page
│   │       ├── index.html                  # Landing page
│   │       ├── report.html                 # PDF report template
│   ├── forms.py                            # Django form for file upload
│   ├── views.py                            # Views for data handling and report generation
│   └── urls.py                             # URL configuration for the app
├── data_analysis_of_customer_sales/        # Django project folder
│   ├── settings.py                         # Django settings
│   ├── urls.py                             # Project-wide URL routing
├── manage.py                               # Django management script
└── db.sqlite3                              # SQLite database (auto-generated)
```

## Installation

### Prerequisites

- **Python 3.7+**
- **Django**: Install Django with `pip install django`.
- **Additional Packages**:
  - **Pandas**: `pip install pandas`
  - **Matplotlib** and **Seaborn** for plotting: `pip install matplotlib seaborn`
  - **ReportLab** for PDF generation: `pip install reportlab`

### Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/KCprsnlcc/data_analysis_of_customer_sales.git
   cd data_analysis_of_customer_sales
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run migrations to set up the database:

   ```bash
   python manage.py migrate
   ```

4. Start the Django development server:

   ```bash
   python manage.py runserver
   ```

5. Open your browser and navigate to `http://127.0.0.1:8000/`.

## Usage

1. **Home Page**: Access the home page, which provides an introduction and navigation to the data analysis page.
2. **Upload Data**: Navigate to the **Data Analysis** page, where you can upload a CSV file containing customer data.
3. **Data Preview**: Preview uploaded data in a table format with pagination and search functionality.
4. **Data Cleansing & Transformation**: The app automatically handles missing values, applies normalization, and removes duplicates.
5. **EDA Visualizations**: 
   - **Category Spend**: A bar chart showing spending by product category.
   - **Correlation Matrix**: Displays correlations between numeric fields as a heatmap.
   - **Trend Line**: Shows spending trends over time.

## Exporting Reports

The application supports exporting the analysis results in Excel and PDF formats:

- **Excel Export**: Click on the “Export as Excel” button to download a structured Excel file with data tables, data types, missing values, and calculated insights.
- **PDF Export**: Click on the “Export as PDF” button to download a PDF report. This report includes data tables, charts, and a summary of insights generated from the data.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.