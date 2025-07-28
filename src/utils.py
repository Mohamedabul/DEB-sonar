import os
from weasyprint import HTML, CSS
import markdown
import base64
from PIL import Image
import logging


def clear_analysis_outputs():
    images_dir = "images"
    if os.path.exists(images_dir):
        for fname in os.listdir(images_dir):
            fpath = os.path.join(images_dir, fname)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
            except Exception as e:
                pass
    # Remove markdown reports
    for report_file in ["initial_analysis_report.md", "final_analysis_report.md"]:
        try:
            if os.path.exists(report_file):
                os.remove(report_file)
        except Exception as e:
            pass


def convert_md_to_pdf(md_content, image_paths):
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['tables'])
    
    # Add custom CSS for better PDF formatting
    css = CSS(string='''
        @page {
            margin: 2.5cm;
            @top-right {
                content: "Page " counter(page) " of " counter(pages);
            }
        }
        body { 
            font-family: "Arial", "Helvetica", sans-serif;
            font-size: 12pt;
            line-height: 1.5;
        }
        h1 { 
            color: #2c3e50;
            font-size: 24pt;
            margin-bottom: 20px;
        }
        h2 { 
            color: #34495e;
            font-size: 20pt;
            margin-top: 30px;
        }
        h3 { 
            color: #2c3e50;
            font-size: 16pt;
        }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0;
            font-size: 11pt;
        }
        th, td { 
            border: 1px solid #ddd; 
            padding: 12px; 
            text-align: left;
        }
        th { 
            background-color: #f8f9fa;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        img { 
            max-width: 100%; 
            height: auto;
            margin: 20px 0;
        }
        code {
            font-family: "Courier New", monospace;
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
        }
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
    ''')
    
    # Create HTML document with images
    html = f"""
    <html>
        <head>
            <meta charset="UTF-8">
            <title>Effort Analysis Report</title>
        </head>
        <body>
            <h1>Effort Analysis Report</h1>
            {html_content}
        </body>
    </html>
    """
    
    try:
        # Generate PDF with error handling
        pdf = HTML(string=html).write_pdf(
            stylesheets=[css],
            presentational_hints=True
        )
        return pdf
    except Exception as e:
        logging.error(f"Error generating PDF: {str(e)}")
        raise

def get_download_link(pdf_bytes, filename):
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="display: inline-block; padding: 10px 20px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px; margin: 20px 0;">Download PDF Report</a>'
    return href