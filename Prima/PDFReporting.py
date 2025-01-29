# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:26:21 2025

@author: Anthony XU
"""

import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

def create_performance_table(performance_df):
    """Crée une table formatée pour les indicateurs de performance"""
    data = [[Paragraph("<b>Indicateur</b>", style=ParagraphStyle('Header', textColor=colors.white)),
             Paragraph("<b>Valeur</b>", style=ParagraphStyle('Header', textColor=colors.white))]]
    for _, row in performance_df.iterrows():
        value = row['Valeur']
        if isinstance(value, (float, int)) and 'Ratio' not in row['Indicateur']:
            formatted_value = f"{value:.2f} %"
        else:
            formatted_value = str(value)

        data.append([
            Paragraph(str(row['Indicateur'])),
            Paragraph(formatted_value)
        ])

    return Table(data, colWidths=[2.75*inch, 2.75*inch], style=[
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2196F3')),  # Bleu plus clair
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
    ])

def create_positions_table(prices_df):
    """Crée une table formatée pour les positions actuelles"""
    headers = [Paragraph(f"<b>{col}</b>", style=ParagraphStyle('Header', textColor=colors.white))
              for col in prices_df.columns]
    data = [headers]

    for _, row in prices_df.iterrows():
        formatted_row = []
        for col, val in row.items():
            if "Prix" in col and isinstance(val, (float, int)):
                formatted_val = f"{val:.2f} €"
            elif "Rendement" in col and isinstance(val, (float, int)):
                formatted_val = f"{val:.2f} %"
            else:
                formatted_val = str(val)
            formatted_row.append(Paragraph(formatted_val))
        data.append(formatted_row)

    table = Table(data, colWidths=[1.2*inch] * len(prices_df.columns))
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2196F3')),  # Bleu plus clair
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
    ]))
    return table

def create_portfolio_chart(portfolio_value):
    """Crée un graphique d'évolution du portfolio avec un style amélioré"""
    # Créer une nouvelle figure avec une taille adaptée au format A4
    plt.figure(figsize=(11, 6))

    # Configuration du style
    plt.style.use('bmh')  # Utilisation d'un style matplotlib intégré

    # Tracer la ligne avec un style amélioré
    plt.plot(portfolio_value.index, portfolio_value.values,
        color='#1976d2',  # Bleu plus vif pour la ligne
        linewidth=2,
        linestyle='-',
        marker='o',
        markersize=4,
        markerfacecolor='white',
        markeredgecolor='#1976d2')

    # Configurer les titres et labels
    plt.title('Évolution de la Valeur du Portfolio',
          fontsize=14,
          fontweight='bold',
          pad=20,
          color='#1a237e')

    plt.xlabel('Date', fontsize=12, labelpad=10)
    plt.ylabel('Valeur (€)', fontsize=12, labelpad=10)

    # Personnaliser la grille
    plt.grid(True, linestyle='--', alpha=0.7)

    # Formater l'axe des ordonnées avec le symbole €
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.2f} €'))

    # Rotation des dates sur l'axe x pour une meilleure lisibilité
    plt.xticks(rotation=45, ha='right')

    # Ajuster automatiquement les marges
    plt.tight_layout()

    # Ajouter une bordure légère autour du graphique
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Améliorer la lisibilité des valeurs sur les axes
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Ajuster les marges pour éviter la troncature
    plt.subplots_adjust(bottom=0.2)

    # Définir les limites de l'axe y pour commencer à 0
    plt.ylim(bottom=0)  # Le haut sera automatiquement ajusté

    # Sauvegarder le graphique avec une résolution améliorée
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer,
                format='png',
                dpi=300,  # Augmenter la résolution
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

    img_buffer.seek(0)
    return Image(img_buffer, width=450, height=250)  # Dimensions ajustées pour le PDF

def create_var_table(VaR_Value):
    """Crée une table formatée pour les résultats VaR avec conversion des symboles"""
    df_copy = VaR_Value.copy()

    positive_style = ParagraphStyle(
        'Positive',
        fontSize=10,
        textColor=colors.HexColor('#00695c'),
        alignment=1,
        leading=12,
        fontName='Helvetica-Bold'
    )

    negative_style = ParagraphStyle(
        'Negative',
        fontSize=10,
        textColor=colors.HexColor('#d32f2f'),
        alignment=1,
        leading=12,
        fontName='Helvetica-Bold'
    )

    headers = [Paragraph(f"<b>{text}</b>", style=ParagraphStyle('Header', textColor=colors.white))
              for text in ["Indice"] + list(df_copy.columns)]
    data = [headers]

    for idx, row in df_copy.iterrows():
        formatted_row = [Paragraph(str(idx))]
        for val in row:
            if isinstance(val, (float, int)):
                if any(metric in str(idx) for metric in ['VaR', 'Risk']):
                    formatted_val = f"{val:.1f}%" if val >= 0 else f"({abs(val):.1f})%"
                else:
                    formatted_val = f"{val:.1f}"
                cell_style = ParagraphStyle(
                    'Compact',
                    fontSize=8,
                    leading=10,
                    alignment=1
                )
            elif isinstance(val, bool) or val in ["✅", "❌"]:
                if val is True or val == "✅":
                    formatted_val = "OUI"
                    cell_style = positive_style
                else:
                    formatted_val = "NON"
                    cell_style = negative_style
            else:
                formatted_val = str(val)
                cell_style = ParagraphStyle(
                    'Compact',
                    fontSize=8,
                    leading=10,
                    alignment=1
                )

            formatted_row.append(Paragraph(formatted_val, cell_style))
        data.append(formatted_row)

    available_width = A4[0] - 100
    first_col_width = available_width * 0.25
    remaining_width = available_width * 0.75
    other_col_width = remaining_width / len(df_copy.columns)

    col_widths = [first_col_width] + [other_col_width] * len(df_copy.columns)

    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2196F3')),  # Bleu plus clair
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ('BOX', (0, 0), (-1, -1), 1, colors.grey),
        ('MINROWHEIGHT', (0, 0), (-1, -1), 20),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
    ]))
    return table

def generate_pdf_report(portfolio_data, portfolio_value, capital_initial,
                       performance_df, transactions_df, prices_df,
                       identifiant, selected_portfolio, VaR_Value):
    """Génère un rapport PDF complet du portfolio avec mise en page optimisée"""
    buffer = io.BytesIO()

    # Augmenter les marges pour une meilleure lisibilité
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    # Nouvelle palette de couleurs professionnelle avec meilleur contraste
    COLORS = {
        'primary': colors.HexColor('#2196F3'),    # Bleu principal plus clair
        'secondary': colors.HexColor('#1565C0'),  # Bleu secondaire ajusté
        'accent': colors.HexColor('#0D47A1'),     # Bleu accent foncé
        'text': colors.HexColor('#212121'),       # Noir plus doux pour le texte
        'background': colors.white,
        'table_header': colors.HexColor('#2196F3'),  # Cohérent avec les autres tableaux
    }

    # Styles optimisés
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,
        textColor=COLORS['secondary'],  # Utilise le bleu secondaire pour meilleur contraste
        leading=30
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=COLORS['accent'],  # Utilise le bleu accent pour les sous-titres
        leading=20
    )

    # Construction du contenu
    elements = []

    # Titre
    elements.append(Paragraph(f"Rapport du Portfolio - {selected_portfolio} de {identifiant}", title_style))
    elements.append(Spacer(1, 20))

    # Information générale avec style de texte amélioré
    elements.append(Paragraph("Informations Générales", heading_style))
    info_data = [
        [Paragraph("<b>Information</b>", style=ParagraphStyle('Header', textColor=colors.white)),
         Paragraph("<b>Valeur</b>", style=ParagraphStyle('Header', textColor=colors.white))],
        ["Date du rapport:", datetime.now().strftime("%d/%m/%Y")],
        ["Capital initial:", f"{capital_initial:,.2f} €"],
        ["Valeur actuelle:", f"{portfolio_value.iloc[-1]:,.2f} €"]
    ]

    # Ajuster la largeur de la table d'informations
    available_width = A4[0] - 100
    info_table = Table(info_data, colWidths=[available_width/2, available_width/2])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLORS['table_header']),  # Utilise la même couleur que les autres tableaux
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), COLORS['text']),  # Utilise la couleur de texte définie
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 20))

    # Ajouter les autres sections
    elements.append(Paragraph("Indicateurs de Performance", heading_style))
    elements.append(create_performance_table(performance_df))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Évolution du Portfolio", heading_style))
    chart = create_portfolio_chart(portfolio_value)
    elements.append(chart)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Positions Actuelles", heading_style))
    elements.append(create_positions_table(prices_df))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Résultats VaR", heading_style))
    elements.append(create_var_table(VaR_Value))

    # Générer le PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer



def add_pdf_download_button_dynamic(portfolio_data, portfolio_value, capital_initial,
                                  performance_df, transactions_df, prices_df,
                                  identifiant, selected_portfolio, VaR_Value):
    """Version avec clé dynamique basée sur le portfolio sélectionné"""
    pdf_file = generate_pdf_report(
        portfolio_data, portfolio_value, capital_initial,
        performance_df, transactions_df, prices_df,
        identifiant, selected_portfolio, VaR_Value
    )

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rapport_portfolio_{selected_portfolio}_{now}.pdf"

    st.download_button(
        label="📄 Télécharger le rapport PDF",
        data=pdf_file,
        file_name=filename,
        mime="application/pdf",
        key=f"pdf_download_{selected_portfolio}"  # Clé dynamique
    )