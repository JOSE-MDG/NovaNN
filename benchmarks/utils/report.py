from typing import Any, Optional, Union
from pathlib import Path
from datetime import datetime
import json


class BenchmarkReport:
    """
    Generate professional benchmark reports in multiple formats.

    Usage:
        report = BenchmarkReport("Backward Overhead Benchmark")
        report.add_section("Performance vs Depth")
        report.add_row(["Depth", "NovaNN (ms)", "PyTorch (ms)", "Ratio"])
        report.add_row([2, 1.23, 0.98, 1.26])
        report.add_row([4, 2.45, 1.87, 1.31])

        # Save as markdown
        report.save_markdown("results.md")

        # Print to console
        report.print_table()
    """

    title: str
    description: str
    sections: list
    current_section: dict[str, list]
    metadata: dict

    def __init__(self, title: str, description: str = ""):

        self.title = title
        self.description = description
        self.sections = []
        self.current_section = None
        self.metadata = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": title,
            "description": description,
        }

    def add_metadata(self, key: str, value: Any):
        """Add custom metadata to report."""
        self.metadata[key] = value

    def add_section(self, section_name: str, description: str = ""):
        """
        Add a new section to the report.

        Args:
            section_name: Name of the section
            description: Optional section description
        """
        section = {
            "name": section_name,
            "description": description,
            "headers": [],
            "rows": [],
            "summary": {},
        }
        self.sections.append(section)
        self.current_section = section

    def add_row(self, row: list[Any]):
        """
        Add a data row to current section.

        Args:
            row: List of values for the row
        """
        if self.current_section is None:
            raise ValueError("No section active. Call add_section() first.")

        # If this is the first row, treat it as headers
        if not self.current_section["headers"] and not self.current_section["rows"]:
            self.current_section["headers"] = [str(v) for v in row]
        else:
            self.current_section["rows"].append(row)

    def add_summary(self, key: str, value: Any):
        """
        Add summary statistics to current section.

        Args:
            key: Summary key
            value: Summary value
        """
        if self.current_section is None:
            raise ValueError("No section active. Call add_section() first.")

        self.current_section["summary"][key] = value

    def _format_value(self, value: Any, precision: int = 2) -> str:
        """Format a value for display."""
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        elif isinstance(value, int):
            return str(value)
        elif value is None:
            return "N/A"
        else:
            return str(value)

    def _get_column_widths(self, section: dict) -> list[int]:
        """Calculate optimal column widths for a section."""
        headers = section["headers"]
        rows = section["rows"]

        if not headers:
            return []

        widths = [len(str(h)) for h in headers]

        for row in rows:
            for i, val in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(self._format_value(val)))

        return widths

    def print_table(self, section_index: Optional[int] = None):
        """
        Print table to console.

        Args:
            section_index: If provided, print only that section. Otherwise print all.
        """
        sections = (
            [self.sections[section_index]]
            if section_index is not None
            else self.sections
        )

        print(f"\n{'='*80}")
        print(f"{self.title.upper()}")
        if self.description:
            print(f"{self.description}")
        print(f"{'='*80}\n")

        for section in sections:
            self._print_section(section)

    def _print_section(self, section: dict):
        """Print a single section as table."""
        print(f"--- {section['name']} ---")
        if section["description"]:
            print(f"{section['description']}")
        print()

        headers = section["headers"]
        rows = section["rows"]

        if not headers:
            print("(No data)")
            print()
            return

        widths = self._get_column_widths(section)

        # Print header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        print(header_line)
        print("-" * len(header_line))

        # Print rows
        for row in rows:
            row_line = " | ".join(
                self._format_value(val).rjust(w) for val, w in zip(row, widths)
            )
            print(row_line)

        # Print summary if exists
        if section["summary"]:
            print()
            for key, value in section["summary"].items():
                print(f"{key}: {self._format_value(value)}")

        print()

    def save_markdown(self, filepath: Union[str, Path]):
        """
        Save report as Markdown file.

        Args:
            filepath: Path to save the markdown file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            # Title and metadata
            f.write(f"# {self.title}\n\n")
            if self.description:
                f.write(f"{self.description}\n\n")

            f.write(f"**Generated:** {self.metadata['generated_at']}\n\n")

            # Sections
            for section in self.sections:
                self._write_section_markdown(f, section)

            f.write("\n---\n")
            f.write("*Report generated by NovaNN Benchmark Suite*\n")

        print(f"✓ Markdown report saved to {filepath}")

    def _write_section_markdown(self, f, section: dict):
        """Write a section in markdown format."""
        f.write(f"## {section['name']}\n\n")
        if section["description"]:
            f.write(f"{section['description']}\n\n")

        headers = section["headers"]
        rows = section["rows"]

        if not headers:
            f.write("*(No data)*\n\n")
            return

        # Write header
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|\n")

        # Write rows
        for row in rows:
            formatted_row = [self._format_value(val) for val in row]
            f.write("| " + " | ".join(formatted_row) + " |\n")

        f.write("\n")

        # Write summary
        if section["summary"]:
            f.write("**Summary:**\n\n")
            for key, value in section["summary"].items():
                f.write(f"- {key}: {self._format_value(value)}\n")
            f.write("\n")

    def save_json(self, filepath: Union[str, Path]):
        """
        Save report as JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": self.metadata,
            "sections": self.sections,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"✓ JSON report saved to {filepath}")

    def save_html(self, filepath: Union[str, Path]):
        """
        Save report as HTML file.

        Args:
            filepath: Path to save the HTML file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        html = self._generate_html()

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"✓ HTML report saved to {filepath}")

    def _generate_html(self) -> str:
        """Generate HTML report."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
        }}
        .metadata {{
            background-color: #fff;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .summary {{
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: #777;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <div class="metadata">
        <p><strong>Description:</strong> {self.description or 'N/A'}</p>
        <p><strong>Generated:</strong> {self.metadata['generated_at']}</p>
    </div>
"""

        for section in self.sections:
            html += self._generate_section_html(section)

        html += """
    <div class="footer">
        <p>Report generated by NovaNN Benchmark Suite</p>
    </div>
</body>
</html>
"""
        return html

    def _generate_section_html(self, section: dict) -> str:
        """Generate HTML for a section."""
        html = f"<h2>{section['name']}</h2>\n"
        if section["description"]:
            html += f"<p>{section['description']}</p>\n"

        headers = section["headers"]
        rows = section["rows"]

        if not headers:
            html += "<p><em>No data</em></p>\n"
            return html

        html += "<table>\n<thead>\n<tr>\n"
        for header in headers:
            html += f"<th>{header}</th>\n"
        html += "</tr>\n</thead>\n<tbody>\n"

        for row in rows:
            html += "<tr>\n"
            for val in row:
                html += f"<td>{self._format_value(val)}</td>\n"
            html += "</tr>\n"

        html += "</tbody>\n</table>\n"

        if section["summary"]:
            html += '<div class="summary">\n<strong>Summary:</strong>\n<ul>\n'
            for key, value in section["summary"].items():
                html += f"<li>{key}: {self._format_value(value)}</li>\n"
            html += "</ul>\n</div>\n"

        return html


class QuickTable:
    """
    Quick table generator for simple use cases.

    Usage:
        table = QuickTable(["Name", "Time (ms)", "Memory (MB)"])
        table.add_row(["NovaNN", 1.23, 45.6])
        table.add_row(["PyTorch", 0.98, 38.2])
        table.print()
    """

    headers: list[str]
    rows: list

    def __init__(self, headers: list[str]):
        self.headers = headers
        self.rows = []

    def add_row(self, row: list[Any]):
        """Add a row to the table."""
        self.rows.append(row)

    def print(self):
        """Print table to console."""
        if not self.headers:
            return

        # Calculate column widths
        widths = [len(str(h)) for h in self.headers]
        for row in self.rows:
            for i, val in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(val)))

        # Print header
        header_line = " | ".join(str(h).ljust(w) for h, w in zip(self.headers, widths))
        print(header_line)
        print("-" * len(header_line))

        # Print rows
        for row in self.rows:
            row_line = " | ".join(str(val).rjust(w) for val, w in zip(row, widths))
            print(row_line)

    def to_markdown(self) -> str:
        """Convert table to markdown string."""
        if not self.headers:
            return ""

        md = "| " + " | ".join(self.headers) + " |\n"
        md += "|" + "|".join(["-" * (len(h) + 2) for h in self.headers]) + "|\n"

        for row in self.rows:
            md += "| " + " | ".join(str(val) for val in row) + " |\n"

        return md
