import os

class DocumentGenerator:
    @staticmethod
    def export_markdown(text, filename, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if not filename.endswith(".md"):
            filename += ".md"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Saved Markdown document to {filepath}")
        return filepath

    @staticmethod
    def export_text(text, filename, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if not filename.endswith(".txt"):
            filename += ".txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Saved Plain Text document to {filepath}")
        return filepath

    @staticmethod
    def export_pdf(text, filename, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        filepath = os.path.join(output_dir, filename)
        
        # Try using reportlab or fpdf if installed, otherwise save as text/html layout
        try:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            # Split lines and write
            for line in text.split('\n'):
                # Encode to latin-1 to avoid unicode characters crash in basic fpdf
                clean_line = line.encode('latin-1', 'replace').decode('latin-1')
                pdf.cell(200, 10, txt=clean_line, ln=True)
            pdf.output(filepath)
            print(f"Saved PDF (FPDF) to {filepath}")
            return filepath
        except Exception as e1:
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.pdfgen import canvas
                c = canvas.Canvas(filepath, pagesize=letter)
                width, height = letter
                y = height - 40
                c.setFont("Helvetica", 10)
                for line in text.split('\n'):
                    if y < 40:
                        c.showPage()
                        y = height - 40
                        c.setFont("Helvetica", 10)
                    c.drawString(40, y, line)
                    y -= 15
                c.save()
                print(f"Saved PDF (ReportLab) to {filepath}")
                return filepath
            except Exception as e2:
                # Fallback: create a mock PDF or write a simple HTML report
                print(f"PDF libraries not found. Saving as a structured HTML document layout for browser rendering.")
                html_path = filepath.replace(".pdf", ".html")
                html_content = f"<html><body style='font-family:sans-serif;padding:40px;line-height:1.6;'><h2>Generated Report</h2><pre>{text}</pre></body></html>"
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                return html_path
