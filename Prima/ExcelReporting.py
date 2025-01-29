# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:41:52 2025

@author: Anthony XU
"""

import pandas as pd
import xlsxwriter
from datetime import datetime
import io
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

def add_var_backtesting_sheet(workbook, VaR_Value):
    """
    Ajoute une feuille Excel pour afficher les résultats des backtests de VaR.
    """
    # Ajouter une feuille pour les backtests
    ws_var = workbook.add_worksheet('VaR Backtesting')

    # Formats pour l'écriture
    header_format = workbook.add_format({
        'bold': True,
        'font_size': 12,
        'bg_color': '#4B0082',
        'font_color': 'white',
        'align': 'center',
        'border': 1
    })
    
    cell_format = workbook.add_format({
        'font_size': 11,
        'align': 'center',
        'border': 1
    })

    percent_format = workbook.add_format({
        'num_format': '0.00%',
        'font_size': 11,
        'align': 'center',
        'border': 1
    })

    # Écriture des en-têtes (noms des colonnes)
    ws_var.write(0, 0, 'Indice', header_format)  # Première colonne pour les indices
    for col_idx, col_name in enumerate(VaR_Value.columns, start=1):
        ws_var.write(0, col_idx, col_name, header_format)

    # Écriture des données (lignes)
    for row_idx, (index, row) in enumerate(VaR_Value.iterrows(), start=1):
        # Écrire l'indice (nom de la VaR) dans la première colonne
        ws_var.write(row_idx, 0, index, cell_format)
        
        # Écrire les valeurs des colonnes
        for col_idx, value in enumerate(row, start=1):
            if isinstance(value, str) and '%' in value:  # Gestion des pourcentages
                ws_var.write(row_idx, col_idx, float(value.replace('%', '').strip()) / 100, percent_format)
            elif isinstance(value, (float, int)):  # Gestion des nombres
                ws_var.write(row_idx, col_idx, value, cell_format)
            else:  # Gestion des autres types
                ws_var.write(row_idx, col_idx, value, cell_format)


def generate_excel_report(portfolio_data, portfolio_value, capital_initial, performance_df, transactions_df, prices_df, identifiant, selected_portfolio, VaR_Value):
    """
    Génère un rapport Excel complet du portfolio avec toutes les métriques et graphiques.
    """
    # Créer un buffer pour le fichier Excel
    output = io.BytesIO()
    
    # Créer le workbook Excel
    workbook = xlsxwriter.Workbook(output, {'nan_inf_to_errors': True})
    
    # Formats
    header_format = workbook.add_format({
        'bold': True,
        'font_size': 12,
        'bg_color': '#4B0082',
        'font_color': 'white',
        'align': 'center',
        'border': 1
    })
    
    cell_format = workbook.add_format({
        'font_size': 11,
        'align': 'center',
        'border': 1
    })
    
    percent_format = workbook.add_format({
        'num_format': '0.00%',
        'font_size': 11,
        'align': 'center',
        'border': 1
    })
    
    currency_format = workbook.add_format({
        'num_format': '#,##0.00 €',
        'font_size': 11,
        'align': 'center',
        'border': 1
    })
    
    date_format = workbook.add_format({
        'num_format': 'dd/mm/yyyy',
        'font_size': 11,
        'align': 'center',
        'border': 1
    })
    
    # 1. Feuille Résumé
    ws_summary = workbook.add_worksheet('Résumé')
    ws_summary.set_column('A:E', 20)
    
    # En-tête du rapport
    ws_summary.merge_range('A1:E1', f'Rapport du Portfolio - {selected_portfolio}', header_format)
    ws_summary.write('A2', 'Date du rapport:', cell_format)
    ws_summary.write('B2', datetime.now(), date_format)
    ws_summary.write('A3', 'Capital initial:', cell_format)
    ws_summary.write('B3', capital_initial, currency_format)
    ws_summary.write('A4', 'Valeur actuelle:', cell_format)
    ws_summary.write('B4', portfolio_value.iloc[-1], currency_format)
    
    # Indicateurs de performance
    ws_summary.merge_range('A6:E6', 'Indicateurs de Performance', header_format)
    ws_summary.write('A7', 'Indicateur', header_format)
    ws_summary.write('B7', 'Valeur', header_format)
    
    for i, row in performance_df.iterrows():
        ws_summary.write(i+7, 0, row['Indicateur'], cell_format)
        ws_summary.write(i+7, 1, row['Valeur'], cell_format)
    
    # 2. Feuille Positions
    ws_positions = workbook.add_worksheet('Positions Actuelles')
    ws_positions.set_column('A:E', 20)
    
    # En-tête des positions
    headers = list(prices_df.columns)
    for col, header in enumerate(headers):
        ws_positions.write(0, col, header, header_format)
    
    # Données des positions
    for row_idx, row in prices_df.iterrows():
        for col_idx, value in enumerate(row):
            try:
                if 'Prix' in headers[col_idx]:
                    if isinstance(value, str):
                        value = float(value.replace('€', '').replace(',', '').strip())
                    ws_positions.write(row_idx + 1, col_idx, value, currency_format)
                elif 'Rendement' in headers[col_idx]:
                    if isinstance(value, str):
                        value = float(value.replace('%', '').strip()) / 100
                    ws_positions.write(row_idx + 1, col_idx, value, percent_format)
                elif 'Date' in headers[col_idx]:
                    if isinstance(value, str):
                        try:
                            date_value = datetime.strptime(value, '%d/%m/%Y')
                            ws_positions.write(row_idx + 1, col_idx, date_value, date_format)
                        except ValueError:
                            ws_positions.write(row_idx + 1, col_idx, value, cell_format)
                    else:
                        ws_positions.write(row_idx + 1, col_idx, value, cell_format)
                else:
                    ws_positions.write(row_idx + 1, col_idx, value, cell_format)
            except (ValueError, TypeError):
                ws_positions.write(row_idx + 1, col_idx, value, cell_format)
    
    # 3. Feuille Transactions 
    ws_transactions = workbook.add_worksheet('Historique Transactions')
    ws_transactions.set_column('A:G', 20)
    
    # En-tête des transactions
    headers = list(transactions_df.columns)
    
    for col, header in enumerate(headers):
        # Renommer "performance" en "performance (%)" si trouvé
        if 'performance' in header.lower():
            ws_transactions.write(0, col, 'Performance (%)', header_format)
        else:
            ws_transactions.write(0, col, header, header_format)
    
    # Données des transactions
    for row_idx, row in transactions_df.iterrows():
        for col_idx, value in enumerate(row):
            try:
                if pd.isna(value):
                    ws_transactions.write(row_idx + 1, col_idx, "", cell_format)
                    continue
                    
                if 'prix' in headers[col_idx].lower():
                    if isinstance(value, str):
                        value = float(value.replace('€', '').replace(',', '').strip())
                    ws_transactions.write(row_idx + 1, col_idx, value, currency_format)
                elif 'performance' in headers[col_idx].lower():
                    try:
                        # Si la valeur est une chaîne contenant un pourcentage
                        if isinstance(value, str) and '%' in value:
                            value = float(value.replace('%', '').strip()) / 100
                        # Si la valeur est directement un float ou un int
                        elif isinstance(value, (float, int)):
                            pass  # Pas de conversion nécessaire
                        else:
                            value = float(value)  # Tenter une conversion générique si possible
                
                        # Écrire la valeur brute (exemple : 0.23)
                        ws_transactions.write(row_idx + 1, col_idx, value, cell_format)
                    except (ValueError, TypeError):
                        # Écrire une cellule vide en cas d'erreur
                        ws_transactions.write(row_idx + 1, col_idx, "", cell_format)

                elif 'date' in headers[col_idx].lower():
                    if isinstance(value, str) and value:
                        try:
                            date_value = datetime.strptime(value, '%d/%m/%Y')
                            ws_transactions.write(row_idx + 1, col_idx, date_value, date_format)
                        except ValueError:
                            ws_transactions.write(row_idx + 1, col_idx, value, cell_format)
                    else:
                        ws_transactions.write(row_idx + 1, col_idx, value, cell_format)
                else:
                    ws_transactions.write(row_idx + 1, col_idx, value, cell_format)
            except (ValueError, TypeError):
                ws_transactions.write(row_idx + 1, col_idx, str(value), cell_format)

    
    # 4. Feuille Graphiques
    ws_charts = workbook.add_worksheet('Graphiques')
    
    # Graphique d'évolution du portfolio
    chart_portfolio = workbook.add_chart({'type': 'line'})
    
    # Préparation des données pour le graphique
    ws_data = workbook.add_worksheet('_Data')
    
    # Convertir l'index en dates si nécessaire
    dates = portfolio_value.index
    if hasattr(dates, 'strftime'):
        dates = dates.strftime('%Y-%m-%d')
    
    ws_data.write_column('A1', dates)
    ws_data.write_column('B1', portfolio_value.values)
    
    chart_portfolio.add_series({
        'name': 'Valeur du Portfolio',
        'categories': '=_Data!$A$1:$A$' + str(len(portfolio_value)),
        'values': '=_Data!$B$1:$B$' + str(len(portfolio_value)),
        'line': {'color': 'blue', 'width': 2},
    })
    
    chart_portfolio.set_title({'name': 'Évolution de la Valeur du Portfolio'})
    chart_portfolio.set_x_axis({'name': 'Date'})
    chart_portfolio.set_y_axis({'name': 'Valeur (€)'})
    
    ws_charts.insert_chart('A1', chart_portfolio, {'x_scale': 2, 'y_scale': 2})
    
    # 5. Feuille VaR

    add_var_backtesting_sheet(workbook, VaR_Value)
    
    # Masquer la feuille de données
    ws_data.hide()
    
    # Fermer le workbook
    workbook.close()
    
    # Préparer le fichier pour le téléchargement
    output.seek(0)
    return output

def add_excel_download_button_dynamic(portfolio_data, portfolio_value, capital_initial, 
                                    performance_df, transactions_df, prices_df, 
                                    identifiant, selected_portfolio, VaR_Value):
    """
    Version avec clé dynamique basée sur le portfolio sélectionné
    """
    excel_file = generate_excel_report(
        portfolio_data, portfolio_value, capital_initial,
        performance_df, transactions_df, prices_df,
        identifiant, selected_portfolio, VaR_Value
    )
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rapport_portfolio_{selected_portfolio}_{now}.xlsx"
    
    st.download_button(
        label="📥 Télécharger le rapport Excel",
        data=excel_file,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"excel_download_{selected_portfolio}"  # Clé dynamique
    )