"""
COALINDIA Analysis PDF Report Generator
Creates comprehensive PDF report with all backtest results, tables, and analysis
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, black, white, blue, red, green
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

class COALINDIAReportGenerator:
    """PDF Report generator for COALINDIA analysis"""
    
    def __init__(self, analysis_results):
        self.results = analysis_results
        self.analyzer = analysis_results['analyzer']
        self.summary_tables = analysis_results['summary_tables']
        self.period_analysis = analysis_results['period_analysis']
        self.best_configs = analysis_results['best_configs']
        
        self.doc = None
        self.story = []
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Define custom styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='COALINDIAMainTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=HexColor('#2c5f2d'),
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='COALINDIASectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=15,
            spaceBefore=20,
            textColor=HexColor('#2c5f2d'),
            fontName='Helvetica-Bold'
        ))
        
        # Subsection style
        self.styles.add(ParagraphStyle(
            name='COALINDIASubSection',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=15,
            textColor=HexColor('#48833d'),
            fontName='Helvetica-Bold'
        ))
        
        # Success style for COALINDIA
        self.styles.add(ParagraphStyle(
            name='COALINDIASuccessText',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=HexColor('#2c5f2d'),
            fontName='Helvetica-Bold'
        ))

    def create_cover_page(self):
        """Create the report cover page"""
        # Main title
        title = Paragraph("COALINDIA Stock Analysis", self.styles['COALINDIAMainTitle'])
        self.story.append(title)
        self.story.append(Spacer(1, 20))
        
        # Subtitle
        subtitle = Paragraph("EMA 9/15 Crossover Strategy - Comprehensive Backtest Report", self.styles['Heading2'])
        subtitle.alignment = TA_CENTER
        self.story.append(subtitle)
        self.story.append(Spacer(1, 40))
        
        # Summary information
        if hasattr(self.analyzer, 'raw_data') and self.analyzer.raw_data is not None:
            start_date = self.analyzer.raw_data.index[0].strftime("%B %d, %Y")
            end_date = self.analyzer.raw_data.index[-1].strftime("%B %d, %Y")
            total_bars = len(self.analyzer.raw_data)
            price_min = self.analyzer.raw_data['Close'].min()
            price_max = self.analyzer.raw_data['Close'].max()
        else:
            start_date = "N/A"
            end_date = "N/A"
            total_bars = 0
            price_min = 0
            price_max = 0
        
        summary_data = [
            ["Analysis Parameter", "Value"],
            ["Stock Symbol", "COALINDIA"],
            ["Strategy Type", "EMA 9/15 Crossover"],
            ["Data Period", f"{start_date} to {end_date}"],
            ["Total Data Points", f"{total_bars:,} bars"],
            ["Base Timeframe", "5 minutes"],
            ["Analysis Timeframes", "15min, 30min"],
            ["Risk:Reward Ratios", "1:1, 1:2, 1:3"],
            ["Stop Loss Types", "Fixed, Trailing"],
            ["Price Range", f"₹{price_min:,.1f} - ₹{price_max:,.1f}"],
            ["Report Generated", datetime.now().strftime("%B %d, %Y at %I:%M %p")]
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 2.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5f2d')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6')),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        self.story.append(summary_table)
        self.story.append(PageBreak())

    def create_executive_summary(self):
        """Create executive summary section"""
        self.story.append(Paragraph("📊 Executive Summary", self.styles['COALINDIASectionHeader']))
        
        # Analysis overview
        overview = f"""
        This comprehensive analysis examines COALINDIA stock performance using EMA 9/15 crossover strategies 
        across multiple configurations. We tested {len(self.analyzer.results)} different strategy combinations 
        including various timeframes, risk-reward ratios, and stop-loss mechanisms.
        """
        self.story.append(Paragraph(overview, self.styles['Normal']))
        self.story.append(Spacer(1, 15))
        
        # Key findings
        if self.best_configs:
            findings = """
            <b>Key Findings:</b><br/><br/>
            """
            
            for metric, config in self.best_configs.items():
                metric_name = metric.replace('_', ' ').replace('highest', 'Best').replace('best', 'Best').replace('lowest', 'Lowest').title()
                config_name = config['config_name']
                
                if 'return' in metric:
                    value = f"{config['total_return']:.2f}%"
                elif 'sharpe' in metric:
                    value = f"{config['sharpe_ratio']:.2f}"
                elif 'drawdown' in metric:
                    value = f"{config['max_drawdown']:.2f}%"
                elif 'win_rate' in metric:
                    value = f"{config['win_rate']:.1f}%"
                elif 'profit_factor' in metric:
                    value = f"{config['profit_factor']:.2f}"
                else:
                    value = str(config.get(metric.replace('highest_', '').replace('best_', '').replace('lowest_', ''), 'N/A'))
                
                findings += f"• <b>{metric_name}:</b> {config_name} ({value})<br/>"
            
            self.story.append(Paragraph(findings, self.styles['Normal']))
        
        self.story.append(Spacer(1, 20))

    def create_main_results_table(self):
        """Create main results comparison table"""
        self.story.append(Paragraph("📈 Complete Strategy Comparison", self.styles['COALINDIASectionHeader']))
        
        if 'main_results' in self.summary_tables:
            df = self.summary_tables['main_results']
            
            # Select only essential columns (remove repetitive Timeframe, Risk Ratio, Stop Type)
            essential_columns = ['Configuration', 'Return (%)', 'B&H Return (%)', 'Sharpe', 
                               'Max DD (%)', 'Win Rate (%)', 'Profit Factor', 'Trades', 'Avg Trade (%)']
            df_filtered = df[essential_columns]
            
            # Prepare table data
            table_data = [df_filtered.columns.tolist()]
            for _, row in df_filtered.iterrows():
                table_data.append([str(val) for val in row.values])
            
            # Create table with optimized column widths
            col_widths = [1.4*inch, 0.8*inch, 0.8*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.6*inch, 0.7*inch]
            
            main_table = Table(table_data, colWidths=col_widths)
            main_table.setStyle(TableStyle([
                # Header row
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5f2d')),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                
                # Special styling for Return column header (index 1)
                ('BACKGROUND', (1, 0), (1, 0), HexColor('#28a745')),  # Green background for Return header
                ('FONTSIZE', (1, 0), (1, 0), 10),  # Larger font for Return header
                
                # Data rows
                ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dee2e6')),
                
                # Alternate row colors
                ('BACKGROUND', (0, 2), (-1, 2), white),
                ('BACKGROUND', (0, 4), (-1, 4), white),
                ('BACKGROUND', (0, 6), (-1, 6), white),
                ('BACKGROUND', (0, 8), (-1, 8), white),
                ('BACKGROUND', (0, 10), (-1, 10), white),
                ('BACKGROUND', (0, 12), (-1, 12), white),
                
                # Red text for negative returns in Return column (index 1)
                ('TEXTCOLOR', (1, 1), (1, -1), HexColor('#dc3545')),
            ]))
            
            # Apply conditional formatting for positive returns (green text)
            for row_idx in range(1, len(table_data)):
                return_value = table_data[row_idx][1]  # Return column
                if return_value != 'N/A' and not return_value.startswith('-'):
                    main_table.setStyle(TableStyle([
                        ('TEXTCOLOR', (1, row_idx), (1, row_idx), HexColor('#28a745')),
                    ]))
            
            self.story.append(main_table)
            
        self.story.append(Spacer(1, 20))

    def create_comparison_tables(self):
        """Create comparison tables by different criteria"""
        self.story.append(Paragraph("🔍 Performance Analysis by Categories", self.styles['COALINDIASectionHeader']))
        
        # Timeframe comparison
        if 'timeframe_comparison' in self.summary_tables:
            self.story.append(Paragraph("Timeframe Comparison", self.styles['COALINDIASubSection']))
            df = self.summary_tables['timeframe_comparison'].reset_index()
            
            table_data = [df.columns.tolist()]
            for _, row in df.iterrows():
                formatted_row = [str(row.iloc[0])]  # Timeframe
                for val in row.iloc[1:]:
                    formatted_row.append(f"{val:.2f}" if isinstance(val, (int, float)) else str(val))
                table_data.append(formatted_row)
            
            tf_table = Table(table_data, colWidths=[1*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            tf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#48833d')),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6')),
            ]))
            
            self.story.append(tf_table)
            self.story.append(Spacer(1, 15))
        
        # Risk ratio comparison  
        if 'risk_ratio_comparison' in self.summary_tables:
            self.story.append(Paragraph("Risk:Reward Ratio Comparison", self.styles['COALINDIASubSection']))
            df = self.summary_tables['risk_ratio_comparison'].reset_index()
            
            table_data = [df.columns.tolist()]
            for _, row in df.iterrows():
                formatted_row = [str(row.iloc[0])]  # Risk ratio
                for val in row.iloc[1:]:
                    formatted_row.append(f"{val:.2f}" if isinstance(val, (int, float)) else str(val))
                table_data.append(formatted_row)
            
            risk_table = Table(table_data, colWidths=[1*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#48833d')),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6')),
            ]))
            
            self.story.append(risk_table)
            self.story.append(Spacer(1, 15))
        
        # Stop type comparison
        if 'stop_type_comparison' in self.summary_tables:
            self.story.append(Paragraph("Stop Loss Type Comparison", self.styles['COALINDIASubSection']))
            df = self.summary_tables['stop_type_comparison'].reset_index()
            
            table_data = [df.columns.tolist()]
            for _, row in df.iterrows():
                formatted_row = [str(row.iloc[0])]  # Stop type
                for val in row.iloc[1:]:
                    formatted_row.append(f"{val:.2f}" if isinstance(val, (int, float)) else str(val))
                table_data.append(formatted_row)
            
            stop_table = Table(table_data, colWidths=[1*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            stop_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#48833d')),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6')),
            ]))
            
            self.story.append(stop_table)
        
        self.story.append(PageBreak())

    def create_period_analysis_tables(self):
        """Create daily, weekly, monthly analysis tables"""
        self.story.append(Paragraph("📅 Time Period Performance Analysis", self.styles['COALINDIASectionHeader']))
        
        periods = ['daily', 'weekly', 'monthly']
        
        for period in periods:
            if period in self.period_analysis and self.period_analysis[period]:
                self.story.append(Paragraph(f"{period.title()} Performance Analysis", self.styles['COALINDIASubSection']))
                
                # Convert period data to table format
                period_data = self.period_analysis[period]
                
                if period_data:
                    # Get first config to determine columns
                    first_config = list(period_data.values())[0]
                    columns = ['Configuration'] + list(first_config.keys())
                    
                    table_data = [columns]
                    
                    for config_key, metrics in period_data.items():
                        # Extract readable config name
                        if config_key in self.analyzer.results:
                            result = self.analyzer.results[config_key]
                            config_name = f"{result['timeframe']} {result['stop_type']} {result['risk_ratio']}"
                        else:
                            config_name = config_key
                        
                        row = [config_name]
                        for metric_value in metrics.values():
                            if isinstance(metric_value, (int, float)):
                                row.append(f"{metric_value:.2f}")
                            else:
                                row.append(str(metric_value))
                        table_data.append(row)
                    
                    # Create table
                    col_widths = [1.5*inch] + [0.9*inch] * (len(columns) - 1)
                    period_table = Table(table_data, colWidths=col_widths)
                    period_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5f2d')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), white),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 8),
                        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
                        ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6')),
                    ]))
                    
                    self.story.append(period_table)
                    self.story.append(Spacer(1, 15))
        
        self.story.append(PageBreak())

    def create_coalindia_specific_insights(self):
        """Create COALINDIA-specific market insights"""
        self.story.append(Paragraph("⚡ COALINDIA Market Insights", self.styles['COALINDIASectionHeader']))
        
        insights = """
        <b>COALINDIA Trading Characteristics:</b><br/><br/>
        
        <b>1. Sector Profile:</b><br/>
        • Coal India Limited is India's largest coal mining company<br/>
        • Public sector undertaking with government backing<br/>
        • Dividend-paying stock with consistent payouts<br/>
        • Sensitive to energy policy and environmental regulations<br/><br/>
        
        <b>2. Price Movement Patterns:</b><br/>
        • Generally exhibits lower volatility compared to private sector stocks<br/>
        • Influenced by commodity cycles and government policies<br/>
        • Seasonal demand patterns from power and steel sectors<br/>
        • Support levels often maintained due to dividend yield attraction<br/><br/>
        
        <b>3. Strategy Considerations for COALINDIA:</b><br/>
        • EMA crossovers may be more reliable due to lower noise<br/>
        • Longer timeframes might capture policy-driven trends better<br/>
        • Risk management crucial during regulatory announcement periods<br/>
        • Consider fundamental factors alongside technical signals<br/><br/>
        
        <b>4. Trading Volume Analysis:</b><br/>
        • Higher institutional participation compared to retail-heavy stocks<br/>
        • Volume spikes often coincide with quarterly results or policy changes<br/>
        • Lower intraday volatility suitable for swing trading strategies<br/>
        • Good liquidity for position entry and exit<br/><br/>
        
        <b>⚠️ COALINDIA-Specific Risks:</b><br/>
        • Environmental and regulatory policy changes<br/>
        • Coal demand shifts due to renewable energy adoption<br/>
        • Government divestment plans and stake sales<br/>
        • Monsoon impact on mining operations
        """
        
        self.story.append(Paragraph(insights, self.styles['Normal']))
        self.story.append(Spacer(1, 20))

    def create_recommendations(self):
        """Create recommendations section"""
        self.story.append(Paragraph("💡 COALINDIA Strategy Recommendations", self.styles['COALINDIASectionHeader']))
        
        recommendations = """
        <b>Based on the COALINDIA analysis, here are targeted recommendations:</b><br/><br/>
        
        <b>1. Optimal Configuration Selection:</b><br/>
        • Review the best performing configurations from our analysis<br/>
        • Consider COALINDIA's lower volatility profile when choosing risk ratios<br/>
        • Factor in dividend ex-dates which can affect technical patterns<br/><br/>
        
        <b>2. Timeframe Optimization:</b><br/>
        • 15-minute charts may capture intraday institutional flows better<br/>
        • 30-minute timeframes could reduce false signals from low volatility periods<br/>
        • Consider daily charts for longer-term position trades<br/><br/>
        
        <b>3. COALINDIA-Specific Risk Management:</b><br/>
        • Use wider stops during earnings and policy announcement periods<br/>
        • Consider fundamental support levels (book value, dividend yield)<br/>
        • Monitor coal prices and energy sector trends<br/><br/>
        
        <b>4. Market Timing Considerations:</b><br/>
        • Avoid trading during monsoon-related news flows<br/>
        • Pay attention to government budget announcements<br/>
        • Monitor energy sector rotation and institutional flows<br/>
        • Consider quarterly earnings seasonality<br/><br/>
        
        <b>5. Portfolio Integration:</b><br/>
        • COALINDIA can serve as a defensive component in energy sector allocation<br/>
        • Pairs well with other PSU stocks for sector-based strategies<br/>
        • Consider as a dividend-focused holding with technical overlay<br/>
        • Use as hedge against private sector energy stocks<br/><br/>
        
        <b>⚠️ Important Disclaimers:</b><br/>
        • Past performance does not guarantee future results<br/>
        • COALINDIA is subject to regulatory and policy risks<br/>
        • Always use proper risk management and position sizing<br/>
        • Consider transaction costs and dividend impacts in live trading<br/>
        • Monitor ESG trends affecting coal sector investments
        """
        
        self.story.append(Paragraph(recommendations, self.styles['Normal']))
        self.story.append(Spacer(1, 20))

    def create_methodology_section(self):
        """Create methodology and technical details section"""
        self.story.append(Paragraph("🔬 Methodology & Technical Details", self.styles['COALINDIASectionHeader']))
        
        methodology = """
        <b>COALINDIA Analysis Framework:</b><br/>
        • <b>Data Source:</b> 5-minute OHLCV bars from COALINDIA<br/>
        • <b>Analysis Period:</b> Comprehensive historical backtest<br/>
        • <b>Strategy Core:</b> EMA 9/15 crossover with bi-directional trading<br/>
        • <b>Risk Management:</b> Fixed and trailing stop-loss mechanisms<br/><br/>
        
        <b>Technical Indicators:</b><br/>
        • <b>EMA 9:</b> Fast exponential moving average for trend detection<br/>
        • <b>EMA 15:</b> Slow exponential moving average for confirmation<br/>
        • <b>Crossover Logic:</b> Both long and short position generation<br/>
        • <b>Stop Loss:</b> 1% base with configurable risk-reward ratios<br/><br/>
        
        <b>Configuration Matrix:</b><br/>
        • <b>Timeframes:</b> 15-minute and 30-minute resampling<br/>
        • <b>Risk Ratios:</b> 1:1, 1:2, and 1:3 risk-to-reward testing<br/>
        • <b>Stop Types:</b> Fixed stops vs. dynamic trailing stops<br/>
        • <b>Total Configs:</b> 12 comprehensive combinations<br/><br/>
        
        <b>Performance Metrics:</b><br/>
        • <b>Return Analysis:</b> Total return vs buy-and-hold comparison<br/>
        • <b>Risk Metrics:</b> Sharpe ratio, maximum drawdown, Calmar ratio<br/>
        • <b>Trade Statistics:</b> Win rate, profit factor, average trade<br/>
        • <b>Execution:</b> Realistic commission assumptions (0.1%)<br/><br/>
        
        <b>COALINDIA-Specific Considerations:</b><br/>
        • <b>Sector Context:</b> PSU coal mining characteristics<br/>
        • <b>Volatility Profile:</b> Lower volatility vs private sector peers<br/>
        • <b>Liquidity Analysis:</b> Institutional vs retail participation<br/>
        • <b>Fundamental Overlay:</b> Policy and regulatory impact assessment
        """
        
        self.story.append(Paragraph(methodology, self.styles['Normal']))

    def create_footer(self):
        """Create report footer"""
        footer_text = f"""
        <br/><br/>
        <b>Report Generated:</b> {datetime.now().strftime("%B %d, %Y at %I:%M %p")}<br/>
        <b>Analysis Framework:</b> backtesting.py with COALINDIA adaptations<br/>
        <b>Stock:</b> Coal India Limited (COALINDIA)<br/>
        <b>Strategy:</b> EMA 9/15 Crossover (Bi-directional)<br/>
        <b>Configurations Tested:</b> {len(self.analyzer.results)}<br/>
        <b>Sector:</b> Energy - Coal Mining<br/>
        <b>Report Version:</b> 1.0 - COALINDIA Specialized
        """
        self.story.append(Paragraph(footer_text, self.styles['Normal']))

    def generate_pdf(self, filename="COALINDIA_Analysis_Report.pdf"):
        """Generate the complete PDF report"""
        # Create document
        self.doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build the report content
        print("📄 Creating COALINDIA PDF report sections...")
        
        self.create_cover_page()
        self.create_executive_summary()
        self.create_main_results_table()
        self.create_comparison_tables()
        self.create_period_analysis_tables()
        self.create_coalindia_specific_insights()
        self.create_recommendations()
        self.create_methodology_section()
        self.create_footer()
        
        # Build PDF
        print("📝 Generating PDF document...")
        self.doc.build(self.story)
        print(f"✅ PDF report generated: {filename}")
        
        return filename

def generate_coalindia_pdf_report(analysis_results, filename="COALINDIA_Analysis_Report.pdf"):
    """Generate PDF report from COALINDIA analysis results"""
    try:
        report_generator = COALINDIAReportGenerator(analysis_results)
        pdf_path = report_generator.generate_pdf(filename)
        
        # Get file size
        file_size = os.path.getsize(pdf_path) / 1024
        print(f"📊 COALINDIA Analysis Report Complete!")
        print(f"📁 File: {pdf_path}")
        print(f"📏 Size: {file_size:.1f} KB")
        
        return pdf_path
        
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        return None

def main():
    """Main function for standalone PDF generation"""
    print("🚀 Running COALINDIA analysis and generating PDF report...")
    
    # Import and run analysis
    from coalindia_analysis import main as run_analysis
    analysis_results = run_analysis()
    
    if analysis_results:
        # Generate PDF report
        pdf_path = generate_coalindia_pdf_report(analysis_results)
        
        if pdf_path:
            print(f"\n🎉 Complete! PDF report saved: {pdf_path}")
            
            # Try to open PDF
            try:
                import subprocess
                import platform
                
                if platform.system() == "Windows":
                    os.startfile(pdf_path)
                elif platform.system() == "Darwin":
                    subprocess.run(["open", pdf_path])
                else:
                    subprocess.run(["xdg-open", pdf_path])
                    
                print("📖 PDF opened successfully!")
                
            except Exception as e:
                print(f"⚠️ Could not auto-open PDF: {e}")
                print(f"📍 Manually open: {os.path.abspath(pdf_path)}")
        
    else:
        print("❌ Analysis failed, cannot generate PDF report")

if __name__ == "__main__":
    main()