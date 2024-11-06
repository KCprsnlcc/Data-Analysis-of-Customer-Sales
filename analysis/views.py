# analysis/views.py

from django.shortcuts import render, redirect
from .forms import UploadFileForm
import pandas as pd
from django.conf import settings
import os
import json
from django.contrib import messages
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from django.http import HttpResponse
from django.template.loader import render_to_string
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet
import datetime
import matplotlib
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

matplotlib.use('Agg')  # Use a non-interactive backend

def index(request):
    return render(request, 'analysis/index.html')

def analysis_view(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Handle the uploaded file
            uploaded_file = request.FILES['file']
            # Save the uploaded file to a temporary location
            temp_file_path = os.path.join(settings.MEDIA_ROOT, 'uploaded_data.csv')
            with open(temp_file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            messages.success(request, 'File uploaded successfully!')
            return redirect('analysis')
    else:
        form = UploadFileForm()

    # Determine which file to load: uploaded or default
    uploaded_file_path = os.path.join(settings.MEDIA_ROOT, 'uploaded_data.csv')
    if os.path.exists(uploaded_file_path):
        csv_path = uploaded_file_path
    else:
        csv_path = os.path.join(settings.BASE_DIR, 'analysis', 'data', 'customer_data.csv')

    try:
        df = pd.read_csv(csv_path, delimiter=',')
    except Exception as e:
        messages.error(request, f"Error loading CSV file: {e}")
        df = pd.DataFrame()  # Empty DataFrame

    # Initialize variables
    preview = None
    data_types_html = None
    missing_values_html = None
    chart_data_json = None
    correlation_matrix_image = None
    trend_line_image = None
    insights = None
    category_spend = None
    search_query = request.GET.get('search', '')  # Get search query from GET parameters
    page = request.GET.get('page', 1)  # Get current page number

    if not df.empty:
        # 1. Loading the Data with Search Filtering
        if search_query:
            # Implement search filtering based on relevant columns, e.g., Customer_ID, Gender, Product_Category
            # Example: filter rows where any field contains the search query
            df = df[df.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)]

        # Implement Pagination
        paginator = Paginator(df.to_dict('records'), 10)  # Show 10 records per page
        try:
            data_page = paginator.page(page)
        except PageNotAnInteger:
            data_page = paginator.page(1)
        except EmptyPage:
            data_page = paginator.page(paginator.num_pages)

        # Convert current page data to DataFrame for further analysis
        df_page = pd.DataFrame(data_page.object_list)

        # Convert paginated data to HTML table
        preview = df_page.to_html(classes='table table-striped', index=False)

        # 2. Inspecting Data Types and Missing Values
        data_types = df.dtypes.to_frame(name='Data Type').reset_index()
        data_types.columns = ['Column', 'Data Type']
        data_types_html = data_types.to_html(classes='table table-bordered', index=False)

        missing_values = df.isnull().sum().to_frame(name='Missing Values').reset_index()
        missing_values.columns = ['Column', 'Missing Values']
        missing_values_html = missing_values.to_html(classes='table table-bordered', index=False)

        # 3. Handling Missing Values
        # Fill missing Age with median
        if 'Age' in df.columns:
            df['Age'].fillna(df['Age'].median(), inplace=True)
        # Fill missing Gender with mode
        if 'Gender' in df.columns:
            if not df['Gender'].mode().empty:
                df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
            else:
                df['Gender'].fillna('U', inplace=True)  # U for Unknown
        # Fill missing Amount_Spent with mean
        if 'Amount_Spent' in df.columns:
            df['Amount_Spent'].fillna(df['Amount_Spent'].mean(), inplace=True)
        # Drop rows where Product_Category is missing
        if 'Product_Category' in df.columns:
            df.dropna(subset=['Product_Category'], inplace=True)

        # 4. Data Cleaning and Transformation
        # Convert Purchase_Date to datetime
        if 'Purchase_Date' in df.columns:
            df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], errors='coerce')
        # Drop duplicates
        df.drop_duplicates(inplace=True)
        # Normalize Amount_Spent
        if 'Amount_Spent' in df.columns:
            df['Amount_Spent_Normalized'] = (df['Amount_Spent'] - df['Amount_Spent'].min()) / (df['Amount_Spent'].max() - df['Amount_Spent'].min())

        # 5. Exploratory Data Analysis (EDA)
        # Example: Amount Spent by Product Category
        category_spend = df.groupby('Product_Category')['Amount_Spent'].sum().reset_index().sort_values(by='Amount_Spent', ascending=False)
        category_spend_html = category_spend.to_html(classes='table table-hover', index=False)

        # Prepare data for Chart.js
        categories = category_spend['Product_Category'].tolist()
        amounts = category_spend['Amount_Spent'].tolist()

        # Convert to JSON for JavaScript
        chart_data = {
            'labels': categories,
            'datasets': [{
                'label': 'Amount Spent',
                'data': amounts,
                'backgroundColor': 'rgba(54, 162, 235, 0.6)',
                'borderColor': 'rgba(54, 162, 235, 1)',
                'borderWidth': 1
            }]
        }

        # Serialize chart data to JSON
        chart_data_json = json.dumps(chart_data)

        # 6. Reporting Insights
        insights = {
            'total_customers': df['Customer_ID'].nunique(),
            'average_age': df['Age'].mean() if 'Age' in df.columns else 'N/A',
            'total_amount_spent': df['Amount_Spent'].sum() if 'Amount_Spent' in df.columns else 'N/A',
            'average_amount_spent': df['Amount_Spent'].mean() if 'Amount_Spent' in df.columns else 'N/A',
            'most_popular_category': category_spend.iloc[0]['Product_Category'] if not category_spend.empty else 'N/A',
            'report_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Detailed EDA
        # Correlation Matrix
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        correlation_matrix_image = None
        if len(numeric_columns) >= 2:
            corr = df[numeric_columns].corr()
            # Plot correlation matrix using Seaborn
            plt.figure(figsize=(8,6))
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix')

            # Save plot to a PNG image in memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            image_png = buf.getvalue()
            buf.close()
            # Encode PNG image to base64 string
            correlation_matrix_image = base64.b64encode(image_png).decode('utf-8')
            correlation_matrix_image = f'data:image/png;base64,{correlation_matrix_image}'

        # Trend Lines (Example: Amount Spent Over Time)
        trend_line_image = None
        if 'Purchase_Date' in df.columns and df['Purchase_Date'].notna().any():
            df_sorted = df.sort_values('Purchase_Date')
            trend = df_sorted.groupby(df_sorted['Purchase_Date'].dt.to_period('M'))['Amount_Spent'].sum().reset_index()
            trend['Purchase_Date'] = trend['Purchase_Date'].dt.to_timestamp()

            # Plot trend line using Seaborn
            plt.figure(figsize=(10,6))
            sns.lineplot(x='Purchase_Date', y='Amount_Spent', data=trend, marker='o')
            plt.title('Amount Spent Over Time')
            plt.xlabel('Purchase Date')
            plt.ylabel('Amount Spent')

            # Save plot to a PNG image in memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            image_png = buf.getvalue()
            buf.close()
            # Encode PNG image to base64 string
            trend_line_image = base64.b64encode(image_png).decode('utf-8')
            trend_line_image = f'data:image/png;base64,{trend_line_image}'

        context = {
            'form': form,
            'preview': preview,
            'data_types': data_types_html,
            'missing_values': missing_values_html,
            'category_spend': category_spend_html,
            'insights': insights,
            'chart_data': chart_data_json,
            'correlation_matrix': correlation_matrix_image,
            'trend_line': trend_line_image,
            'paginator': paginator,
            'data_page': data_page,
            'search_query': search_query,
        }

        return render(request, 'analysis/analysis.html', context)

    else:
        context = {'form': form, 'error': 'No data available. Please upload a CSV file.'}
        return render(request, 'analysis/analysis.html', context)

def export_excel(request):
    # Determine which file to load: uploaded or default
    uploaded_file_path = os.path.join(settings.MEDIA_ROOT, 'uploaded_data.csv')
    if os.path.exists(uploaded_file_path):
        csv_path = uploaded_file_path
    else:
        csv_path = os.path.join(settings.BASE_DIR, 'analysis', 'data', 'customer_data.csv')

    try:
        df = pd.read_csv(csv_path, delimiter=',')
    except Exception as e:
        messages.error(request, f"Error loading CSV file: {e}")
        return redirect('analysis')

    if not df.empty:
        # 1. Loading the Data
        # Preview is not needed here; we write the full DataFrame

        # 2. Inspecting Data Types and Missing Values
        data_types = df.dtypes.to_frame(name='Data Type').reset_index()
        data_types.columns = ['Column', 'Data Type']

        missing_values = df.isnull().sum().to_frame(name='Missing Values').reset_index()
        missing_values.columns = ['Column', 'Missing Values']

        # 3. Handling Missing Values
        # Fill missing Age with median
        if 'Age' in df.columns:
            df['Age'].fillna(df['Age'].median(), inplace=True)
        # Fill missing Gender with mode
        if 'Gender' in df.columns:
            if not df['Gender'].mode().empty:
                df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
            else:
                df['Gender'].fillna('U', inplace=True)  # U for Unknown
        # Fill missing Amount_Spent with mean
        if 'Amount_Spent' in df.columns:
            df['Amount_Spent'].fillna(df['Amount_Spent'].mean(), inplace=True)
        # Drop rows where Product_Category is missing
        if 'Product_Category' in df.columns:
            df.dropna(subset=['Product_Category'], inplace=True)

        # 4. Data Cleaning and Transformation
        # Convert Purchase_Date to datetime
        if 'Purchase_Date' in df.columns:
            df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], errors='coerce')
        # Drop duplicates
        df.drop_duplicates(inplace=True)
        # Normalize Amount_Spent
        if 'Amount_Spent' in df.columns:
            df['Amount_Spent_Normalized'] = (df['Amount_Spent'] - df['Amount_Spent'].min()) / (df['Amount_Spent'].max() - df['Amount_Spent'].min())

        # 5. Exploratory Data Analysis (EDA)
        # Example: Amount Spent by Product Category
        category_spend = df.groupby('Product_Category')['Amount_Spent'].sum().reset_index().sort_values(by='Amount_Spent', ascending=False)

        # 6. Reporting Insights
        insights = {
            'total_customers': df['Customer_ID'].nunique(),
            'average_age': df['Age'].mean() if 'Age' in df.columns else 'N/A',
            'total_amount_spent': df['Amount_Spent'].sum() if 'Amount_Spent' in df.columns else 'N/A',
            'average_amount_spent': df['Amount_Spent'].mean() if 'Amount_Spent' in df.columns else 'N/A',
            'most_popular_category': category_spend.iloc[0]['Product_Category'] if not category_spend.empty else 'N/A',
            'report_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Detailed EDA
        # Correlation Matrix
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        correlation_matrix = None
        if len(numeric_columns) >= 2:
            corr = df[numeric_columns].corr()
            correlation_matrix = corr

        # Trend Lines (Example: Amount Spent Over Time)
        trend_line = None
        if 'Purchase_Date' in df.columns and df['Purchase_Date'].notna().any():
            df_sorted = df.sort_values('Purchase_Date')
            trend = df_sorted.groupby(df_sorted['Purchase_Date'].dt.to_period('M'))['Amount_Spent'].sum().reset_index()
            trend['Purchase_Date'] = trend['Purchase_Date'].dt.to_timestamp()
            trend_line = trend

        # Prepare Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write different sheets
            df.to_excel(writer, sheet_name='Raw Data', index=False)
            data_types.to_excel(writer, sheet_name='Data Types', index=False)
            missing_values.to_excel(writer, sheet_name='Missing Values', index=False)
            category_spend.to_excel(writer, sheet_name='Category Spend', index=False)
            if correlation_matrix is not None:
                correlation_matrix.to_excel(writer, sheet_name='Correlation Matrix')
            if trend_line is not None:
                trend_line.to_excel(writer, sheet_name='Trend Line', index=False)

            # Insights
            insights_df = pd.DataFrame(list(insights.items()), columns=['Metric', 'Value'])
            insights_df.to_excel(writer, sheet_name='Insights', index=False)

        output.seek(0)

        # Prepare response
        response = HttpResponse(output.read(),
                                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        filename = f"analysis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        response['Content-Disposition'] = f'attachment; filename={filename}'
        return response

    else:
        messages.error(request, 'No data available to export.')
        return redirect('analysis')

def export_pdf(request):
    # Determine which file to load: uploaded or default
    uploaded_file_path = os.path.join(settings.MEDIA_ROOT, 'uploaded_data.csv')
    if os.path.exists(uploaded_file_path):
        csv_path = uploaded_file_path
    else:
        csv_path = os.path.join(settings.BASE_DIR, 'analysis', 'data', 'customer_data.csv')

    try:
        df = pd.read_csv(csv_path, delimiter=',')
    except Exception as e:
        messages.error(request, f"Error loading CSV file: {e}")
        return redirect('analysis')

    if not df.empty:
        # 1. Loading the Data
        # Preview is not needed here; we will include full data or a subset

        # 2. Inspecting Data Types and Missing Values
        data_types = df.dtypes.to_frame(name='Data Type').reset_index()
        data_types.columns = ['Column', 'Data Type']

        missing_values = df.isnull().sum().to_frame(name='Missing Values').reset_index()
        missing_values.columns = ['Column', 'Missing Values']

        # 3. Handling Missing Values
        # Fill missing Age with median
        if 'Age' in df.columns:
            df['Age'].fillna(df['Age'].median(), inplace=True)
        # Fill missing Gender with mode
        if 'Gender' in df.columns:
            if not df['Gender'].mode().empty:
                df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
            else:
                df['Gender'].fillna('U', inplace=True)  # U for Unknown
        # Fill missing Amount_Spent with mean
        if 'Amount_Spent' in df.columns:
            df['Amount_Spent'].fillna(df['Amount_Spent'].mean(), inplace=True)
        # Drop rows where Product_Category is missing
        if 'Product_Category' in df.columns:
            df.dropna(subset=['Product_Category'], inplace=True)

        # 4. Data Cleaning and Transformation
        # Convert Purchase_Date to datetime
        if 'Purchase_Date' in df.columns:
            df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], errors='coerce')
        # Drop duplicates
        df.drop_duplicates(inplace=True)
        # Normalize Amount_Spent
        if 'Amount_Spent' in df.columns:
            df['Amount_Spent_Normalized'] = (df['Amount_Spent'] - df['Amount_Spent'].min()) / (df['Amount_Spent'].max() - df['Amount_Spent'].min())

        # 5. Exploratory Data Analysis (EDA)
        # Example: Amount Spent by Product Category
        category_spend = df.groupby('Product_Category')['Amount_Spent'].sum().reset_index().sort_values(by='Amount_Spent', ascending=False)

        # 6. Reporting Insights
        insights = {
            'total_customers': df['Customer_ID'].nunique(),
            'average_age': df['Age'].mean() if 'Age' in df.columns else 'N/A',
            'total_amount_spent': df['Amount_Spent'].sum() if 'Amount_Spent' in df.columns else 'N/A',
            'average_amount_spent': df['Amount_Spent'].mean() if 'Amount_Spent' in df.columns else 'N/A',
            'most_popular_category': category_spend.iloc[0]['Product_Category'] if not category_spend.empty else 'N/A',
            'report_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Detailed EDA
        # Correlation Matrix
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        correlation_matrix_image = None
        if len(numeric_columns) >= 2:
            corr = df[numeric_columns].corr()
            # Plot correlation matrix using Seaborn
            plt.figure(figsize=(8,6))
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix')

            # Save plot to a PNG image in memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            image_png = buf.getvalue()
            buf.close()
            # Encode PNG image to base64 string
            correlation_matrix_image = base64.b64encode(image_png).decode('utf-8')
            correlation_matrix_image = f'data:image/png;base64,{correlation_matrix_image}'

        # Trend Lines (Example: Amount Spent Over Time)
        trend_line_image = None
        if 'Purchase_Date' in df.columns and df['Purchase_Date'].notna().any():
            df_sorted = df.sort_values('Purchase_Date')
            trend = df_sorted.groupby(df_sorted['Purchase_Date'].dt.to_period('M'))['Amount_Spent'].sum().reset_index()
            trend['Purchase_Date'] = trend['Purchase_Date'].dt.to_timestamp()

            # Plot trend line using Seaborn
            plt.figure(figsize=(10,6))
            sns.lineplot(x='Purchase_Date', y='Amount_Spent', data=trend, marker='o')
            plt.title('Amount Spent Over Time')
            plt.xlabel('Purchase Date')
            plt.ylabel('Amount Spent')

            # Save plot to a PNG image in memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            image_png = buf.getvalue()
            buf.close()
            # Encode PNG image to base64 string
            trend_line_image = base64.b64encode(image_png).decode('utf-8')
            trend_line_image = f'data:image/png;base64,{trend_line_image}'

        # Create a PDF using ReportLab
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
        elements = []
        styles = getSampleStyleSheet()
        style_heading = styles['Heading1']
        style_subheading = styles['Heading2']
        style_normal = styles['Normal']

        # Title
        elements.append(Paragraph("Data Analysis Report", style_heading))
        elements.append(Spacer(1, 12))

        # Report Date
        elements.append(Paragraph(f"<strong>Date:</strong> {insights['report_date']}", style_normal))
        elements.append(Spacer(1, 12))

        # 1. Data Preview
        elements.append(Paragraph("1. Loading the Data", style_subheading))
        elements.append(Spacer(1, 12))
        # Convert DataFrame to list of lists for ReportLab Table
        preview_df = df.head()
        data_preview = [preview_df.columns.tolist()] + preview_df.values.tolist()
        table_preview = Table(data_preview)
        table_preview.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING',(0,0),(-1,0),12),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        elements.append(table_preview)
        elements.append(Spacer(1, 24))

        # 2. Data Types and Missing Values
        elements.append(Paragraph("2. Inspecting Data Types and Missing Values", style_subheading))
        elements.append(Spacer(1, 12))

        # Data Types
        elements.append(Paragraph("Data Types", styles['Heading3']))
        data_types_list = [data_types.columns.tolist()] + data_types.values.tolist()
        table_data_types = Table(data_types_list, hAlign='LEFT')
        table_data_types.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING',(0,0),(-1,0),12),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        elements.append(table_data_types)
        elements.append(Spacer(1, 12))

        # Missing Values
        elements.append(Paragraph("Missing Values", styles['Heading3']))
        data_missing = [missing_values.columns.tolist()] + missing_values.values.tolist()
        table_missing = Table(data_missing, hAlign='LEFT')
        table_missing.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING',(0,0),(-1,0),12),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        elements.append(table_missing)
        elements.append(Spacer(1, 24))

        # 3. Amount Spent by Product Category
        elements.append(Paragraph("3. Amount Spent by Product Category", style_subheading))
        elements.append(Spacer(1, 12))
        data_category_spend = [category_spend.columns.tolist()] + category_spend.values.tolist()
        table_category_spend = Table(data_category_spend, hAlign='LEFT')
        table_category_spend.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING',(0,0),(-1,0),12),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        elements.append(table_category_spend)
        elements.append(Spacer(1, 24))

        # 4. Correlation Matrix
        if correlation_matrix_image:
            elements.append(Paragraph("4. Correlation Matrix", style_subheading))
            elements.append(Spacer(1, 12))
            # Add Correlation Matrix Image
            correlation_image = RLImage(io.BytesIO(base64.b64decode(correlation_matrix_image.split(',')[1])), width=6*inch, height=4*inch)
            elements.append(correlation_image)
            elements.append(Spacer(1, 24))

        # 5. Trend Line: Amount Spent Over Time
        if trend_line_image:
            elements.append(Paragraph("5. Trend Line: Amount Spent Over Time", style_subheading))
            elements.append(Spacer(1, 12))
            # Add Trend Line Image
            trend_image = RLImage(io.BytesIO(base64.b64decode(trend_line_image.split(',')[1])), width=6*inch, height=4*inch)
            elements.append(trend_image)
            elements.append(Spacer(1, 24))

        # 6. Reporting Insights
        elements.append(Paragraph("6. Reporting Insights", style_subheading))
        elements.append(Spacer(1, 12))
        # Create a list of insights
        insights_list = [
            f"Total Customers: {insights['total_customers']}",
            f"Average Age: {insights['average_age']:.2f}" if isinstance(insights['average_age'], float) else f"Average Age: {insights['average_age']}",
            f"Total Amount Spent: ${insights['total_amount_spent']:.2f}" if isinstance(insights['total_amount_spent'], (float, int)) else f"Total Amount Spent: {insights['total_amount_spent']}",
            f"Average Amount Spent: ${insights['average_amount_spent']:.2f}" if isinstance(insights['average_amount_spent'], (float, int)) else f"Average Amount Spent: {insights['average_amount_spent']}",
            f"Most Popular Category: {insights['most_popular_category']}",
        ]
        for item in insights_list:
            elements.append(Paragraph(item, styles['Normal']))
            elements.append(Spacer(1, 6))

        # Build the PDF
        doc.build(elements)

        # Get the PDF from the buffer
        pdf = buffer.getvalue()
        buffer.close()

        # Prepare response
        response = HttpResponse(pdf, content_type='application/pdf')
        filename = f"analysis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        response['Content-Disposition'] = f'attachment; filename={filename}'
        return response

    else:
        messages.error(request, 'No data available to export.')
        return redirect('analysis')
