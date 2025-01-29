"""
Created on Thu Dec  5 16:02:01 2024

@author: Darvin/Anthony
"""

# ======================== Imports ========================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import os
from Prima_function import *
from VaR_Analysis import *
from ExcelReporting import *

from dateutil.relativedelta import relativedelta
from PDFReporting import *


# ======================== Configuration ========================
st.set_page_config(
    page_title="PRIMA",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: rgba(255, 255, 255, 0.8);'>
    <p style='margin: 0; font-size: 12px; color: #666;'>© 2024 Darvin & Anthony. Tous droits réservés.</p>
</div>
""", unsafe_allow_html=True)

# Chemins et constantes
os.listdir('C:/Users/darvi/Desktop/Proj Var/Project_file')
DATA_FILEPATH = "Data_Connect.csv"

# Dictionnaire de correspondance des tickers CAC40
cac40_yfinance = {
    "Accor": "AC.PA", "Air Liquide": "AI.PA", "Airbus": "AIR.PA",
    "Atos": "ATO.PA", "BNP Paribas": "BNP.PA", "Bouygues": "EN.PA",
    "Capgemini": "CAP.PA", "Carrefour": "CA.PA", "Crédit Agricole": "ACA.PA",
    "Danone": "BN.PA", "Engie": "ENGI.PA", "Kering": "KER.PA",
    "Legrand": "LR.PA", "L'Oréal": "OR.PA", "LVMH": "MC.PA",
    "Michelin": "ML.PA", "Orange": "ORA.PA", "Pernod Ricard": "RI.PA",
    "Sanofi": "SAN.PA", "Saint-Gobain": "SGO.PA", "Schneider Electric": "SU.PA",
    "Société Générale": "GLE.PA", "Sodexo": "SW.PA", "Thales": "HO.PA",
    "Veolia Environnement": "VIE.PA", "Vinci": "DG.PA", "Vivendi": "VIV.PA",
    "Worldline": "WLN.PA"
}

# ======================== Fonctions de Gestion des Données ========================

def get_Data(filepath):
    """Charge et valide les données utilisateur depuis le CSV"""
    try:
        data = pd.read_csv(filepath, header=None, sep=";")
        dico_data = {}

        for _, row in data.iterrows():
            user_id = row[0]
            user_data = []
            for value in row:
                try:
                    if isinstance(value, str) and '.' in value:
                        converted_value = float(value)
                    else:
                        converted_value = str(value)
                except ValueError:
                    converted_value = str(value)
                user_data.append(converted_value)
            dico_data[user_id] = user_data

        return dico_data
    except FileNotFoundError:
        st.error("Le fichier de données est introuvable.")
        return {}
    except Exception as e:
        return {}

def save_data(dico_data):
    """Sauvegarde les données utilisateur dans le CSV"""
    rows = []
    for user_id, user_data in dico_data.items():
        row = []
        for item in user_data:
            if isinstance(item, (int, float)):
                item = str(item)
            row.append(str(item))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("Data_Connect.csv", sep=";", index=False, header=False)

# ======================== Fonctions de Gestion des Prix ========================

def get_Prices(Ticker_list, entry_dates_dict, cac40_yfinance):
    """Récupère les prix historiques des actifs"""
    try:
        earliest_date = min(datetime.strptime(date, "%d/%m/%Y")
                          for date in entry_dates_dict.values())
        start_date = earliest_date.strftime("%Y-%m-%d")

        S = yf.download(Ticker_list, start=start_date, interval='1d')
        if S.empty:
            st.error("Impossible de télécharger les données pour les tickers fournis.")
            return pd.DataFrame(), pd.DataFrame()

        S_Close = S['Close']
        S_Close_100 = pd.DataFrame(index=S_Close.index)

        for ticker_name, entry_date in entry_dates_dict.items():
            if ticker_name in cac40_yfinance:
                ticker_symbol = cac40_yfinance[ticker_name]
                entry_date_dt = pd.to_datetime(entry_date, format="%d/%m/%Y").tz_localize(None)

                # Convert S_Close index to timezone-naive for comparison
                close_index_naive = S_Close.index.tz_localize(None)
                mask = close_index_naive >= entry_date_dt

                if mask.any():
                    series = S_Close[ticker_symbol][mask]
                    if not series.empty:
                        S_Close_100[ticker_symbol] = series / series.iloc[0] * 100

        return S_Close, S_Close_100
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données : {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def get_price_at_date(ticker_name, selected_date, entry_date, entry_price, cac40_yfinance):
    """
    Calcule le prix à une date donnée
    Returns: float or None
    """
    try:
        ticker_symbol = cac40_yfinance.get(ticker_name)
        if not ticker_symbol:
            return None

        entry_date_dt = datetime.strptime(entry_date, "%d/%m/%Y")
        selected_datetime = datetime.combine(selected_date, datetime.min.time())

        if selected_datetime < entry_date_dt:
            return float(entry_price)

        start_date = entry_date_dt.strftime("%Y-%m-%d")
        end_date = (selected_datetime + timedelta(days=1)).strftime("%Y-%m-%d")

        ticker_data = yf.download(ticker_symbol, start=start_date, end=end_date)

        if ticker_data.empty:
            return float(entry_price)

        try:
            entry_date_price = float(ticker_data['Close'][ticker_data.index >= start_date].iloc[0])
            selected_date_price = float(ticker_data['Close'][ticker_data.index <= end_date].iloc[-1])
            calculated_price = float(entry_price * (selected_date_price / entry_date_price))
            return round(calculated_price, 2)
        except (IndexError, ValueError):
            return float(entry_price)

    except Exception as e:
        st.error(f"Erreur lors du calcul du prix : {str(e)}")
        return float(entry_price)

# ======================== Fonctions de Visualisation ========================

def graphique(S, List_Ticker, Base_Ticker):
    """Trace les courbes des prix normalisés"""
    fig, ax = plt.subplots(figsize=(10, 6))

    if len(List_Ticker) == 1:
        ax.plot(S.index, S, label=Base_Ticker[0])
    else:
        for i, ticker in enumerate(List_Ticker):
            ax.plot(S.index, S[ticker], label=Base_Ticker[i])

    ax.set_title('Évolution temporelle des prix (Base 100)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Prix (Base 100)', fontsize=12)
    ax.legend()
    ax.grid(True)
    return fig

def BP_BR(Dico_ID, identifiant, selected_portfolio, selected_option):
    """
    Analyse les meilleures et pires performances du portefeuille sur une période donnée
    Affiche les résultats avec des métriques Streamlit
    """
    today = date.today()

    # Définir la période d'analyse
    if selected_option == "Journalier":
        T = today - relativedelta(days=1)
        period_text = "24h"
    elif selected_option == "Mensuel":
        T = today - relativedelta(months=1)
        period_text = "30 jours"
    elif selected_option == "Annuel":
        T = today - relativedelta(years=1)
        period_text = "1 an"

    # Récupérer les données du portefeuille
    portfolios = get_user_portfolios(Dico_ID, identifiant)
    current_portfolio = portfolios[selected_portfolio]
    current_tickers = current_portfolio["tickers"]
    entry_dates = current_portfolio["dates"]

    # Initialiser les dictionnaires pour stocker les variations
    price_changes = {}
    returns = {}

    # Calculer les variations pour chaque actif
    for ticker, entry_price in current_tickers.items():
        if ticker in cac40_yfinance:
            # Prix à la date T
            price_T = get_price_at_date(
                ticker,
                T,
                entry_dates[ticker],
                entry_price,
                cac40_yfinance
            )

            # Prix actuel
            current_price = get_price_at_date(
                ticker,
                today,
                entry_dates[ticker],
                entry_price,
                cac40_yfinance
            )

            if price_T and current_price:
                # Calculer la variation de prix absolue
                price_change = current_price - price_T
                price_changes[ticker] = price_change

                # Calculer le rendement en pourcentage
                ret = (current_price - price_T) / price_T * 100
                returns[ticker] = ret

    # Trouver les meilleures et pires performances
    best_price = max(price_changes.items(), key=lambda x: x[1])
    worst_price = min(price_changes.items(), key=lambda x: x[1])
    best_return = max(returns.items(), key=lambda x: x[1])
    worst_return = min(returns.items(), key=lambda x: x[1])


    # Affichage avec st.metrics selon le choix
    col1, col2 = st.columns(2)

    with col1:
        # Radio button pour choisir le type d'affichage
        display_type = st.radio(
            f"Afficher les variations {selected_option} en :",
            ["Prix (€)", "Rendement (%)"],
            horizontal=False
        )
    with col2:
        if display_type == "Prix (€)":
            st.metric(
                label=f"{best_price[0]}",
                value=f"{best_price[1]:,.2f} €",
                delta=f"{returns[best_price[0]]:.1f}%",
                help = "Actif avec le différentiel de prix le plus élevé"
            )
            st.metric(
                label=f"{worst_price[0]}",
                value=f"{worst_price[1]:,.2f} €",
                delta=f"{returns[worst_price[0]]:.1f}%",
                help = "Actif avec le différentiel de prix le plus faible"
            )
        else:  # Rendement (%)
            st.metric(
                label=f"{best_return[0]}",
                value=f"{best_return[1]:.1f}%",
                delta=f"{price_changes[best_return[0]]:.2f} €",
                help = "Actif avec le Rendement le plus élevé"
            )
            st.metric(
                label=f"{worst_return[0]}",
                value=f"{worst_return[1]:.1f}%",
                delta=f"{price_changes[worst_return[0]]:.2f} €",
                help = "Actif avec le Rendement le plus faible"
            )

    return {
        'best_price': best_price,
        'worst_price': worst_price,
        'best_return': best_return,
        'worst_return': worst_return
    }

def calculate_period_performance(Dico_ID, identifiant, selected_portfolio, selected_option):
    """
    Analyse la performance du portefeuille sur une période donnée (Journalier, Mensuel, Annuel)
    Affiche les résultats avec des métriques Streamlit

    Args:
        Dico_ID: Dictionnaire contenant les données utilisateur
        identifiant: Identifiant de l'utilisateur
        selected_portfolio: Nom du portefeuille sélectionné
        selected_option: Période sélectionnée ("Journalier", "Mensuel", "Annuel")

    Returns:
        dict: Résultats de l'analyse de performance
    """
    today = date.today()

    # Définir la période d'analyse
    if selected_option == "Journalier":
        T = today - relativedelta(days=1)
        period_label = "Performance 24H"
        delta = 2
    elif selected_option == "Mensuel":
        T = today - relativedelta(months=1)
        period_label = "Performance 30J"
        delta = 22
    else:  # Annuel
        T = today - relativedelta(years=1)
        period_label = "Performance 1AN"
        delta = 255

    # Récupérer les données du portefeuille
    portfolios = get_user_portfolios(Dico_ID, identifiant)
    current_portfolio = portfolios[selected_portfolio]
    current_tickers = current_portfolio["tickers"]
    entry_dates = current_portfolio["dates"]

    # Convertir les tickers en symboles yfinance
    tick = [cac40_yfinance[ticker] for ticker in current_tickers if ticker in cac40_yfinance]

    # Initialisation des variables
    total_value = 0
    total_initial_value = 0

    # Récupérer les données de prix
    price_data = PriceToRdt(tick, entry_dates)
    price_data.columns = current_tickers

    valid_tickers = []  # Pour suivre les tickers avec données valides

    # Calcul pour chaque actif
    for ticker, entry_price in current_tickers.items():
        date_object = datetime.strptime(entry_dates[ticker], "%d/%m/%Y").date()

        if ticker not in price_data.columns:
            continue

        # Calculer la série de prix ajustés
        prices = price_data[ticker].copy()
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        if len(prices) == 0:
            continue

        # Calculer les prix ajustés
        adjusted_prices = (prices + 1).cumprod() * float(entry_price)
        if len(adjusted_prices) < delta:
            st.warning(f"Données insuffisantes pour {ticker}")
            continue

        current_price = adjusted_prices.iloc[-1]

        if date_object <= T:
            # Position plus ancienne que la période
            initial_price = adjusted_prices.iloc[-delta]
        else:
            # Nouvelle position - prendre le premier prix disponible
            initial_price = adjusted_prices.iloc[0]

        # Ajouter aux totaux
        total_value += current_price
        total_initial_value += initial_price
        valid_tickers.append(ticker)

    # Calculer les performances
    result = {
        'total_value': total_value,
        'total_initial_value': total_initial_value,
        'valid_tickers': valid_tickers,
        'period_label': period_label
    }

    if total_initial_value > 0 and len(valid_tickers) > 0:
        result['gain_loss'] = total_value - total_initial_value
        result['performance'] = (total_value / total_initial_value - 1)
        result['has_data'] = True
    else:
        result['has_data'] = False

    return result


def get_Df_Rdts(Dico_ID,Portfolio, Cap, performance_df, entry_dates, identifiant, selected_portfolio):
    """Affiche les métriques de performance et les graphiques"""

    st.markdown(
        """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 27px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-size: 13.5px; /* Taille du label principal */
        }
        div[class*="stRadio"] label {
            font-size: 10px; /* Taille des options */
        }
        </style>
        """, unsafe_allow_html=True
    )

    with st.container():

        J = st.columns([0.8,0.8,0.8,1.1,1.1],border=True)

        with J[0]:
            with st.container():
              st.markdown('<div class="metric-card">', unsafe_allow_html=True)
              st.metric(
                  label="💰 Capitaux investis",
                  value=f"{Cap:,.2f} €",
                  help="Montant total des investissements"
              )
              st.markdown('</div>', unsafe_allow_html=True)

        with J[1]:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if len(Portfolio) > 1:
                    value = Portfolio.iloc[-1] - Cap
                    Rdt = (Portfolio.iloc[-1] - Cap) / Cap
                    st.metric(
                        label="📈 Performance Globale",
                        value=f'{value:,.2f} €',
                        delta=f'{Rdt*100:.2f} %',
                        help="Performance totale depuis le début"
                    )
                else:
                    st.metric("📈 Performance Globale", "N/A", delta="0.00 %")
                st.markdown('</div>', unsafe_allow_html=True)

    # Section de la colonne J[3] modifiée
        # Dans la section J[3]
        with J[3]:
            # Radio button pour choisir la période
            cols2 = st.columns([1, 1])
            with cols2[0]:
                selected_option = st.radio(
                    "📅 Sélectionnez une période :",
                    ["Journalier", "Mensuel", "Annuel"]
                )

            # Calcul et affichage des performances
            performance_result = calculate_period_performance(
                Dico_ID, identifiant, selected_portfolio, selected_option
            )

            # Affichage des métriques
            with cols2[1]:
                if performance_result['has_data']:
                    st.metric(
                        label=performance_result['period_label'],
                        value=f"{performance_result['gain_loss']:,.2f} €",
                        delta=f"{performance_result['performance']*100:.2f} %",
                        help=f"Performance du portefeuille sur la période {selected_option.lower()}"
                    )
                else:
                    st.metric(performance_result['period_label'], "Données insuffisantes", delta="N/A")
        with J[4]:
            BP_BR(Dico_ID, identifiant, selected_portfolio, selected_option)
        with J[2]:
            transaction_file = f"transactions_{identifiant}.csv"
            transactions_df = pd.read_csv(transaction_file, encoding='utf-8')
            List_transaction = transactions_df.values
            Dico_transaction = dict()

            for i in range(len(List_transaction)):
                if selected_portfolio == List_transaction[i, 0]:
                    clef = List_transaction[i, 3]
                    if clef not in Dico_transaction:
                        Dico_transaction[clef] = [List_transaction[i, 2], List_transaction[i, 4]]
                    else:
                        if List_transaction[i, 2] == 'close':
                            Dico_transaction[clef][0] = 'close'
                            Dico_transaction[clef][1] = List_transaction[i, 4] - Dico_transaction[clef][1]

            profits_clos = sum(val[1] for val in Dico_transaction.values() if val[0] == 'close')

            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="🎯 Profits positions closes",
                value=f"{profits_clos:,.2f} €",
                help="Total des profits réalisés sur les positions fermées"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1],border = True)
    with col1:
        st.line_chart(Portfolio, y_label='Evolution du capital', x_label='Date')

    with col2:
        st.markdown(
            """
            <div style="text-align: center;">
                <h3>Indicateurs de Performances</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.dataframe(performance_df, hide_index=True,use_container_width=True)



# ======================== Fonctions de Gestion du Portefeuille ========================

def get_user_portfolios(dico_id, identifiant):
    """
    Version modifiée pour gérer les dates par actif
    """
    if identifiant in dico_id:
        user_data = dico_id[identifiant]
        portfolios = {}

        for i in range(2, len(user_data), 3):
            if i + 2 < len(user_data):
                portfolio_name = user_data[i]
                tickers_data = user_data[i + 1]
                creation_date = user_data[i + 2]

                if (portfolio_name and
                    not pd.isna(portfolio_name) and
                    str(portfolio_name).strip() and
                    str(portfolio_name).strip().lower() != 'nan' and
                    tickers_data and
                    not pd.isna(tickers_data) and
                    str(tickers_data).strip() and
                    str(tickers_data).strip().lower() != 'nan'):

                    tickers_dict, dates_dict = parse_ticker_data(tickers_data)

                    # Utiliser la date de création du portefeuille pour tout actif sans date
                    for ticker in tickers_dict:
                        if not dates_dict.get(ticker):
                            dates_dict[ticker] = creation_date

                    portfolios[portfolio_name] = {
                        "tickers": tickers_dict,
                        "dates": dates_dict,
                        "creation_date": creation_date
                    }
        return portfolios
    return {}

def get_last_prices_df(List_Ticker, Base_Ticker, tickers_dict, entry_dates, cac40_yfinance):
    """
    Version modifiée pour utiliser les dates d'entrée individuelles
    """
    price_data = []
    all_data = pd.DataFrame()
    # Récupérer les données pour chaque ticker depuis sa date d'entrée
    for ticker, entry_date in entry_dates.items():
        ticker_symbol = cac40_yfinance.get(ticker)
        if ticker_symbol in List_Ticker:
            start_date = datetime.strptime(entry_date, "%d/%m/%Y").strftime("%Y-%m-%d")
            data = yf.download(ticker_symbol, start=start_date)['Close']
            all_data[ticker_symbol] = data

    for ticker, entry_date in entry_dates.items():
        ticker_symbol = cac40_yfinance.get(ticker)
        start_date = datetime.strptime(entry_date, "%d/%m/%Y").strftime("%Y-%m-%d")

        # Récupérer la première valeur non-NaN
        initial_price = float(all_data[ticker_symbol].dropna().iloc[0])
        current_price = float(all_data[ticker_symbol].dropna().iloc[-1])

        actual_price = float(tickers_dict[ticker] * (current_price / initial_price))
        rendement = float(((actual_price - tickers_dict[ticker]) / tickers_dict[ticker]) * 100)
        price_data.append({
            'Actif': ticker,
            'Prix d\'achat (€)': round(tickers_dict[ticker], 2),
            'Prix actuel (€)': round(actual_price, 2),
            'Rendement (%)': round(rendement, 2),
            'Date d\'entrée': entry_date
        })

    if price_data:
        last_prices_df = pd.DataFrame(price_data)

        # Créer le graphique
        S_Close, S_Close_100 = get_Prices([cac40_yfinance[t] for t in Base_Ticker],
                                        entry_dates,
                                        cac40_yfinance)
        fig = graphique(S_Close_100, [cac40_yfinance[t] for t in Base_Ticker], Base_Ticker)

        # Affichage
        col1, col2 = st.columns(2,border=True)
        with col1:
            st.pyplot(fig)

        with col2:
            st.subheader("Analyse du portefeuille")
            st.dataframe(last_prices_df, hide_index=True,use_container_width=True)

        return last_prices_df
    else:
        st.warning("Impossible de récupérer les données de prix")
        return pd.DataFrame()

def update_portfolio(dico_id, identifiant, portfolio_name, current_tickers, entry_dates,
                    transaction_type, transaction_date, performance_data=None):
    """
    Mise à jour du portefeuille et de l'historique des transactions
    """
    try:
        user_data = dico_id[identifiant]

        # Mise à jour du portefeuille
        portfolio_updated = False
        for i in range(2, len(user_data), 3):
            if user_data[i] == portfolio_name:
                # Utiliser le nouveau format incluant les dates
                tickers_str = format_ticker_data(current_tickers, entry_dates)
                user_data[i + 1] = tickers_str
                portfolio_updated = True
                break

        if not portfolio_updated:
            st.error("Portefeuille non trouvé")
            return False

        # Mise à jour de l'historique des transactions
        transaction_file = f"transactions_{identifiant}.csv"

        try:
            transactions_df = pd.read_csv(transaction_file)
        except FileNotFoundError:
            transactions_df = pd.DataFrame(columns=[
                'portfolio', 'date', 'type', 'ticker', 'price',
                'entry_date', 'exit_date', 'performance'
            ])

        # Créer la nouvelle transaction selon le type
        if transaction_type == "add":
            # Pour un ajout, récupérer le dernier ticker ajouté
            latest_ticker = list(current_tickers.keys())[-1]
            new_transaction = {
                'portfolio': portfolio_name,
                'date': transaction_date,
                'type': 'add',
                'ticker': latest_ticker,
                'price': current_tickers[latest_ticker],
                'entry_date': entry_dates[latest_ticker],
                'exit_date': None,
                'performance': None
            }
        elif transaction_type == "close" and performance_data:
            new_transaction = {
                'portfolio': portfolio_name,
                'date': transaction_date,
                'type': 'close',
                'ticker': performance_data['ticker'],
                'price': performance_data['exit_price'],
                'entry_date': entry_dates.get(performance_data['ticker'], ''),
                'exit_date': transaction_date,
                'performance': performance_data['performance']
            }
        else:
            st.error("Type de transaction non valide")
            return False

        # Ajouter la nouvelle transaction
        transactions_df = pd.concat([transactions_df, pd.DataFrame([new_transaction])],
                                  ignore_index=True)

        # Sauvegarde des modifications
        transactions_df.to_csv(transaction_file, index=False)
        dico_id[identifiant] = user_data
        save_data(dico_id)

        return True

    except Exception as e:
        st.error(f"Erreur lors de la mise à jour : {str(e)}")
        return False

def display_transaction_history(identifiant, selected_portfolio=None):
    """
    Affiche l'historique des transactions d'un portefeuille
    """
    try:
        transaction_file = f"transactions_{identifiant}.csv"
        transactions_df = pd.read_csv(transaction_file, encoding='utf-8')

        if selected_portfolio:
            transactions_df = transactions_df[transactions_df['portfolio'] == selected_portfolio]

        if not transactions_df.empty:
            st.subheader("Historique des transactions")

            # Copie du DataFrame pour ne pas modifier l'original
            display_df = transactions_df.copy()

            # Formater les dates
            for date_col in ['date', 'entry_date', 'exit_date']:
                display_df[date_col] = pd.to_datetime(display_df[date_col],
                                                    format='%d/%m/%Y',
                                                    errors='coerce')
                display_df[date_col] = display_df[date_col].dt.strftime('%d/%m/%Y')

            # Formater les valeurs numériques
            # Prix avec symbole euro
            display_df['price'] = display_df['price'].apply(
                lambda x: f"{float(x):.2f}€" if pd.notnull(x) else ""
            )

            # Performance avec symbole pourcentage
            display_df['performance'] = display_df['performance'].apply(
                lambda x: f"{float(x):.2f}%" if pd.notnull(x) else ""
            )

            # Renommer les colonnes pour l'affichage
            columns_mapping = {
                'date': 'Date de transaction',
                'type': 'Type',
                'ticker': 'Actif',
                'price': 'Prix',
                'entry_date': 'Date d\'entrée',
                'exit_date': 'Date de sortie',
                'performance': 'Performance'
            }

            # Sélectionner et réorganiser les colonnes
            display_df = display_df[['date', 'type', 'ticker', 'price',
                                   'entry_date', 'exit_date', 'performance']]
            display_df = display_df.rename(columns=columns_mapping)

            st.dataframe(display_df, hide_index=True)
        else:
            st.info("Aucune transaction enregistrée")

    except FileNotFoundError:
        st.info("Aucun historique de transactions disponible")
    except Exception as e:
        st.error(f"Erreur lors de l'affichage de l'historique : {str(e)}")


def format_ticker_data(tickers_dict, dates_dict):
    """
    Format: "ticker1:prix1:date1;ticker2:prix2:date2"
    """
    return ';'.join([f"{ticker}:{price}:{dates_dict.get(ticker)}"
                    for ticker, price in tickers_dict.items()])

def parse_ticker_data(ticker_str):
    """
    Parse le format "ticker1:prix1:date1;ticker2:prix2:date2"
    Retourne deux dictionnaires : un pour les prix et un pour les dates
    """
    tickers_dict = {}
    dates_dict = {}
    if ticker_str and isinstance(ticker_str, str):
        for item in ticker_str.split(';'):
            if ':' in item:
                parts = item.split(':')
                if len(parts) >= 3:
                    ticker, price, date = parts[:3]
                    tickers_dict[ticker] = float(price)
                    dates_dict[ticker] = date
                elif len(parts) == 2:
                    ticker, price = parts
                    tickers_dict[ticker] = float(price)
                    dates_dict[ticker] = None
    return tickers_dict, dates_dict

def PriceToRdt(List_Ticker, dates_dict):
    """
    Calcule les rendements en tenant compte des dates d'entrée individuelles

    Args:
        List_Ticker: Liste des symboles ticker
        dates_dict: Dictionnaire des dates d'entrée par ticker
    """
    all_data = pd.DataFrame()

    # Récupérer les données pour chaque ticker depuis sa date d'entrée
    for ticker, entry_date in dates_dict.items():
        ticker_symbol = cac40_yfinance.get(ticker)
        if ticker_symbol in List_Ticker:
            start_date = datetime.strptime(entry_date, "%d/%m/%Y").strftime("%Y-%m-%d")
            data = yf.download(ticker_symbol, start=start_date)['Close']
            all_data[ticker_symbol] = data

    # Calculer les rendements
    returns = all_data.pct_change()

    # Remplir les NaN avec 0 (pas de rendement avant la date d'entrée)
    returns = returns.fillna(0)

    return returns

def calculate_portfolio_value(tickers_dict, entry_dates, ponderations, Rdt):
    """Calcule la valeur du portefeuille en tenant compte des dates d'entrée"""
    # Convert Rdt index to timezone-naive if it has timezone info
    if Rdt.index.tz is not None:
        Rdt.index = Rdt.index.tz_localize(None)

    # Initialize portfolio
    Portfolio = pd.Series(0, index=Rdt.index)

    # Add each position at its entry date
    for ticker, price in tickers_dict.items():
        if ticker in cac40_yfinance:
            ticker_symbol = cac40_yfinance[ticker]
            # Convert entry date to timezone-naive datetime
            entry_date = pd.to_datetime(entry_dates[ticker], format="%d/%m/%Y")

            # Calculate position evolution
            mask = Rdt.index >= entry_date
            position_value = pd.Series(0, index=Rdt.index)
            position_value[mask] = price * (1 + Rdt[ticker_symbol][mask]).cumprod()

            Portfolio = Portfolio + position_value

    return Portfolio

# ======================== Pages de l'Application ========================

# ======================== Création de compte ======================== #

def page_create(dico_id, cac40_yfinance):
    # Style CSS personnalisé
    st.markdown("""
        <style>
            .main-container {
                max-width: 800px;
                margin: auto;
                padding: 2rem;
            }
            .stButton button {
                width: 100%;
                margin-top: 1rem;
                background-color: #FF4B4B;
                color: white;
            }
            .form-section {
                background-color: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 2rem;
            }
            .section-title {
                color: #333;
                margin-bottom: 1.5rem;
            }
            .asset-selector {
                border: 1px solid #eee;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
            }
            .selected-assets {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 1rem;
                margin-top: 1rem;
            }
            .asset-chip {
                display: inline-block;
                background-color: #e9ecef;
                padding: 0.3rem 0.8rem;
                border-radius: 16px;
                margin: 0.2rem;
                font-size: 0.9rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # Container principal
    with st.container():
        col1, main_col, col2 = st.columns([1, 3, 1])
        with main_col:
            # En-tête
            st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>Création de compte</h1>", unsafe_allow_html=True)

            # Formulaire principal
            with st.form("create_account_form"):
                # Section 1: Informations de base
                st.markdown("<h3 class='section-title'>📝 Informations de compte</h3>", unsafe_allow_html=True)
                with st.container():
                    col_id, col_pwd = st.columns(2)
                    with col_id:
                        new_identifiant = st.text_input("👤 Identifiant", placeholder="Votre identifiant")
                    with col_pwd:
                        new_mdp = st.text_input("🔒 Mot de passe", type="password", placeholder="Votre mot de passe")

                # Section 2: Informations du portefeuille
                st.markdown("<h3 class='section-title'>💼 Configuration du portefeuille</h3>", unsafe_allow_html=True)
                new_portofolio = st.text_input("📊 Nom du portefeuille", placeholder="Mon portefeuille")
                default_date = st.date_input(
                    "📅 Date de début de suivi",
                    value=datetime.now(),
                    min_value=datetime(2000, 1, 1),
                    max_value=datetime.now(),
                    format="DD/MM/YYYY"
                )
                new_period1 = default_date.strftime("%d/%m/%Y")

                # Section 3: Sélection des actifs
                st.markdown("<h3 class='section-title'>🎯 Sélection des actifs</h3>", unsafe_allow_html=True)
                selected_tickers = {}
                entry_dates = {}

                # Nouvelle interface de sélection d'actifs
                # 1. Multi-sélecteur d'actifs
                available_tickers = list(cac40_yfinance.keys())
                selected_stocks = st.multiselect(
                    "Sélectionnez vos actifs",
                    available_tickers,
                    placeholder="Choisissez un ou plusieurs actifs"
                )

                # 2. Configuration des actifs sélectionnés
                if selected_stocks:
                    st.markdown("<div class='selected-assets'>", unsafe_allow_html=True)
                    st.markdown("#### 📈 Configuration des actifs sélectionnés")

                    # Affichage en tableau pour une meilleure organisation
                    for ticker in selected_stocks:
                        with st.container():
                            st.markdown(f"##### {ticker}")
                            col1, col2 = st.columns(2)

                            with col1:
                                price = st.number_input(
                                    "💰 Prix d'achat",
                                    min_value=0.0,
                                    step=0.01,
                                    key=f"price_{ticker}"
                                )
                            with col2:
                                ticker_date = st.date_input(
                                    "📅 Date d'achat",
                                    value=default_date,
                                    min_value=datetime(2000, 1, 1),
                                    max_value=datetime.now(),
                                    format="DD/MM/YYYY",
                                    key=f"date_{ticker}"
                                )

                            if price > 0:
                                selected_tickers[ticker] = price
                                entry_dates[ticker] = ticker_date.strftime("%d/%m/%Y")

                            st.markdown("<hr>", unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                # Résumé des sélections
                if selected_tickers:
                    st.markdown("#### 📋 Résumé de votre sélection")
                    for ticker, price in selected_tickers.items():
                        st.markdown(f"""
                            <div class='asset-chip'>
                                {ticker}: {price}€ ({entry_dates[ticker]})
                            </div>
                        """, unsafe_allow_html=True)

                # Boutons d'action
                col_submit, col_return = st.columns(2)
                with col_submit:
                    submit_button = st.form_submit_button("✅ Créer le compte")
                with col_return:
                    return_button = st.form_submit_button("↩️ Retour")

                # Traitement de la soumission
                if submit_button:
                    if new_identifiant and new_mdp and new_period1:
                        if new_identifiant in dico_id:
                            st.error("⚠️ Cet identifiant existe déjà.")
                        elif not selected_stocks:  # Vérification des actifs sélectionnés
                            st.error("⚠️ Veuillez sélectionner au moins un actif.")
                        else:
                            # Vérification des prix pour les actifs sélectionnés
                            missing_prices = [ticker for ticker in selected_stocks if ticker not in selected_tickers]
                            if missing_prices:
                                st.error(f"⚠️ Veuillez saisir les prix d'achat pour : {', '.join(missing_prices)}")
                            else:
                                tickers_str = format_ticker_data(selected_tickers, entry_dates)
                                dico_id[new_identifiant] = [
                                    new_identifiant,
                                    new_mdp,
                                    new_portofolio,
                                    tickers_str,
                                    new_period1
                                ]
                                save_data(dico_id)

                                # Création du fichier de transactions
                                transaction_file = f"transactions_{new_identifiant}.csv"
                                transactions_data = [
                                    {
                                        'portfolio': new_portofolio,
                                        'date': entry_dates[ticker],
                                        'type': 'add',
                                        'ticker': ticker,
                                        'price': price,
                                        'entry_date': entry_dates[ticker],
                                        'exit_date': None,
                                        'performance': None
                                    }
                                    for ticker, price in selected_tickers.items()
                                ]
                                transactions_df = pd.DataFrame(transactions_data)
                                transactions_df.to_csv(transaction_file, index=False)

                                st.success("✅ Compte créé avec succès!")
                                st.session_state.page = "login"
                                st.rerun()
                    else:
                        st.error("⚠️ Tous les champs sont obligatoires.")

                if return_button:
                    st.session_state.page = "login"
                    st.rerun()

# ======================== Créer un nouveau portefeuille ======================== #

def create_portfolio(dico_id, identifiant, portfolio_name, tickers_with_prices, period1):
    """
    Crée un nouveau portefeuille pour un utilisateur existant avec gestion des dates

    Args:
        dico_id: dictionnaire des utilisateurs
        identifiant: identifiant de l'utilisateur
        portfolio_name: nom du nouveau portefeuille
        tickers_with_prices: dictionnaire {ticker: prix_achat}
        period1: date de création du portefeuille
    """
    if identifiant in dico_id:
        user_data = dico_id[identifiant]
        existing_portfolios = get_user_portfolios(dico_id, identifiant)

        if portfolio_name in existing_portfolios:
            return False

        # S'assurer que la date est au bon format
        if isinstance(period1, datetime):
            period1 = period1.strftime("%d/%m/%Y")

        # Ajouter le nouveau portefeuille aux données utilisateur
        user_data.extend([portfolio_name, tickers_with_prices, period1])
        dico_id[identifiant] = user_data

        # Sauvegarder les modifications
        save_data(dico_id)
        return True
    return False

# ======================== Créer un nouveau portefeuille ======================== #

def page_new_portfolio():
    # Style CSS personnalisé
    st.markdown("""
        <style>
            .main-container {
                max-width: 800px;
                margin: auto;
                padding: 2rem;
            }
            .stButton button {
                width: 100%;
                margin-top: 1rem;
                background-color: #FF4B4B;
                color: white;
            }
            .form-section {
                background-color: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 2rem;
            }
            .section-title {
                color: #333;
                margin-bottom: 1.5rem;
            }
            .selected-assets {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 1rem;
                margin-top: 1rem;
            }
            .asset-chip {
                display: inline-block;
                background-color: #e9ecef;
                padding: 0.3rem 0.8rem;
                border-radius: 16px;
                margin: 0.2rem;
                font-size: 0.9rem;
            }
            .summary-section {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                margin-top: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # Container principal
    with st.container():
        col1, main_col, col2 = st.columns([1, 3, 1])
        with main_col:
            # En-tête
            st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>Nouveau portefeuille</h1>", unsafe_allow_html=True)

            identifiant = st.session_state.Identifiant

            # Formulaire principal
            with st.form("new_portfolio_form"):
                # Section 1: Informations de base
                st.markdown("<h3 class='section-title'>📝 Informations du portefeuille</h3>", unsafe_allow_html=True)
                portfolio_name = st.text_input("📊 Nom du portefeuille", placeholder="Mon nouveau portefeuille")
                default_date = st.date_input(
                    "📅 Date de début",
                    value=datetime.now(),
                    min_value=datetime(2000, 1, 1),
                    max_value=datetime.now(),
                    format="DD/MM/YYYY"
                )
                default_date_str = default_date.strftime("%d/%m/%Y")

                # Section 2: Sélection des actifs
                st.markdown("<h3 class='section-title'>🎯 Composition du portefeuille</h3>", unsafe_allow_html=True)
                selected_tickers = {}
                entry_dates = {}

                # Sélecteur d'actifs multiple
                available_tickers = list(cac40_yfinance.keys())
                selected_stocks = st.multiselect(
                    "Sélectionnez les actifs à inclure",
                    available_tickers,
                    placeholder="Choisissez un ou plusieurs actifs"
                )

                # Configuration des actifs sélectionnés
                if selected_stocks:
                    st.markdown("<div class='selected-assets'>", unsafe_allow_html=True)
                    st.markdown("#### 📈 Configuration des actifs")

                    for ticker in selected_stocks:
                        with st.container():
                            st.markdown(f"##### {ticker}")
                            col1, col2 = st.columns(2)

                            with col1:
                                price = st.number_input(
                                    "💰 Prix d'achat",
                                    min_value=0.0,
                                    step=0.01,
                                    key=f"price_{ticker}"
                                )
                            with col2:
                                ticker_date = st.date_input(
                                    "📅 Date d'achat",
                                    value=default_date,
                                    min_value=datetime(2000, 1, 1),
                                    max_value=datetime.now(),
                                    format="DD/MM/YYYY",
                                    key=f"date_{ticker}"
                                )

                            if price > 0:
                                selected_tickers[ticker] = price
                                entry_dates[ticker] = ticker_date.strftime("%d/%m/%Y")

                            st.markdown("<hr>", unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                # Résumé des sélections
                if selected_tickers:
                    st.markdown("#### 📋 Résumé de la composition")
                    st.markdown("<div class='summary-section'>", unsafe_allow_html=True)
                    for ticker, price in selected_tickers.items():
                        st.markdown(f"""
                            <div class='asset-chip'>
                                {ticker}: {price}€ ({entry_dates[ticker]})
                            </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Boutons d'action
                col_submit, col_return = st.columns(2)
                with col_submit:
                    submit_button = st.form_submit_button("✅ Créer le portefeuille")
                with col_return:
                    return_button = st.form_submit_button("↩️ Retour au tableau de bord")

                # Traitement de la soumission
                if submit_button:
                    if not portfolio_name:
                        st.error("⚠️ Veuillez donner un nom au portefeuille.")
                    elif not selected_stocks:  # Vérification des actifs sélectionnés
                        st.error("⚠️ Veuillez sélectionner au moins un actif.")
                    else:
                        # Vérification des prix pour les actifs sélectionnés
                        missing_prices = [ticker for ticker in selected_stocks if ticker not in selected_tickers]
                        if missing_prices:
                            st.error(f"⚠️ Veuillez saisir les prix d'achat pour : {', '.join(missing_prices)}")
                        else:
                            existing_portfolios = get_user_portfolios(Dico_ID, identifiant)
                            if portfolio_name in existing_portfolios:
                                st.error("⚠️ Un portefeuille avec ce nom existe déjà.")
                            else:
                                tickers_str = format_ticker_data(selected_tickers, entry_dates)
                                user_data = Dico_ID[identifiant]
                                user_data.extend([portfolio_name, tickers_str, default_date_str])
                                Dico_ID[identifiant] = user_data
                                save_data(Dico_ID)

                                # Mise à jour des transactions
                                transaction_file = f"transactions_{identifiant}.csv"
                                transactions_data = [
                                    {
                                        'portfolio': portfolio_name,
                                        'date': entry_dates[ticker],
                                        'type': 'add',
                                        'ticker': ticker,
                                        'price': price,
                                        'entry_date': entry_dates[ticker],
                                        'exit_date': None,
                                        'performance': None
                                    }
                                    for ticker, price in selected_tickers.items()
                                ]
                                try:
                                    existing_df = pd.read_csv(transaction_file)
                                    transactions_df = pd.DataFrame(transactions_data)
                                    final_df = pd.concat([existing_df, transactions_df], ignore_index=True)
                                except FileNotFoundError:
                                    final_df = pd.DataFrame(transactions_data)

                                final_df.to_csv(transaction_file, index=False)
                                st.success("✅ Portefeuille créé avec succès!")
                                st.session_state.current_page = "dashboard"
                                st.rerun()

                if return_button:
                    st.session_state.current_page = "dashboard"
                    st.rerun()

# ======================== Page de Connexion ======================== #

def page_connexion():
    # Centrer le contenu avec un container
    container = st.container()
    with container:
        # Créer une carte centrée avec un padding
        col1, card, col2 = st.columns([1, 2, 1])
        with card:

            # Titre stylisé
            st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>Connexion</h1>", unsafe_allow_html=True)

            # Message de succès
            if "success_message" in st.session_state:
                st.success(st.session_state.success_message)
                del st.session_state.success_message

            # Formulaire avec style
            with st.form("login_form"):
                # Champs de saisie avec icons
                st.markdown("""
                    <style>
                        div[data-baseweb="input"] {
                            margin-bottom: 1rem;
                        }
                        .stButton button {
                            width: 100%;
                            margin-top: 1rem;
                            background-color: #FF4B4B;
                            color: white;
                        }
                    </style>
                """, unsafe_allow_html=True)

                Identifiant = st.text_input("👤 Identifiant", placeholder="Entrez votre identifiant").strip()
                MDP = st.text_input("🔒 Mot de passe", type="password", placeholder="Entrez votre mot de passe")

                # Boutons stylisés
                col_submit, col_create = st.columns(2)
                with col_submit:
                    submitted = st.form_submit_button("Se connecter", use_container_width=True)
                with col_create:
                    create_account = st.form_submit_button("Créer un compte", use_container_width=True)

                # Gestion de la soumission du formulaire
                if submitted:
                    if not Identifiant:
                        st.error("⚠️ Veuillez saisir un identifiant.")
                    elif Identifiant not in Dico_ID:
                        st.error("❌ Identifiant ou mot de passe incorrect.")
                    elif MDP != Dico_ID[Identifiant][1]:
                        st.error("❌ Identifiant ou mot de passe incorrect.")
                    else:
                        st.session_state.authenticated = True
                        st.session_state.Identifiant = Identifiant
                        st.session_state.current_page = "dashboard"
                        st.rerun()

                if create_account:
                    st.session_state.page = "create_account"
                    st.rerun()


# ======================== Page gestion Portfeuille ======================== #

def page_portfolio_management():
    # En-tête stylisé
    st.markdown("""
        <div style='background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem'>
            <h1 style='color: #1f2937; margin: 0'>Gestion du Portefeuille</h1>
        </div>
    """, unsafe_allow_html=True)

    identifiant = st.session_state.Identifiant



    if identifiant in Dico_ID:
        portfolios = get_user_portfolios(Dico_ID, identifiant)
        portfolio_names = list(portfolios.keys())

        # Layout en colonnes pour une meilleure organisation
        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown("### Sélection du portefeuille")
            default_portfolio = st.session_state.get('selected_portfolio', portfolio_names[0] if portfolio_names else None)

            if portfolio_names:
                selected_portfolio = st.selectbox(
                    "",  # Label vide pour un meilleur design
                    portfolio_names,
                    index=portfolio_names.index(default_portfolio) if default_portfolio in portfolio_names else 0,
                    help="Choisissez le portefeuille à gérer"
                )

                # Informations du portefeuille
                if selected_portfolio:
                    current_portfolio = portfolios[selected_portfolio]
                    st.markdown("---")
                    st.markdown("### Informations")
                    st.markdown(f"**Date de création:** {current_portfolio['creation_date']}")
                    st.markdown(f"**Nombre de positions:** {len(current_portfolio['tickers'])}")

            # Bouton de retour stylisé
            st.markdown("---")
            if st.button("📊 Retour au tableau de bord", use_container_width=True):
                st.session_state.selected_portfolio = selected_portfolio
                st.session_state.current_page = "dashboard"
                st.rerun()

        with col2:
            if selected_portfolio:
                current_portfolio = portfolios[selected_portfolio]
                current_tickers = current_portfolio["tickers"]
                entry_dates = current_portfolio["dates"]

                # Configuration des dates
                List_entry_dates = [datetime.strptime(date, "%d/%m/%Y") for date in entry_dates.values()]
                min_entry_dates = min(List_entry_dates) if List_entry_dates else datetime.now()

                # Onglets avec icônes
                tabs = st.tabs([
                    "📈 Positions actuelles",
                    "➕ Nouvelle position",
                    "🔚 Clôturer une position",
                    "📋 Historique"
                ])

                # Onglet 1: Positions actuelles
                with tabs[0]:
                    st.markdown("### Positions actuelles")
                    if current_tickers:
                        positions_df = pd.DataFrame({
                            'Actif': list(current_tickers.keys()),
                            'Prix d\'achat': [f"{price:.2f} €" for price in current_tickers.values()],
                            'Date d\'entrée': [entry_dates[ticker] for ticker in current_tickers.keys()]
                        })
                        st.dataframe(
                            positions_df,
                            hide_index=True,
                            column_config={
                                "Actif": st.column_config.TextColumn(
                                    "Actif",
                                    help="Symbole de l'actif",
                                    width="medium"
                                ),
                                "Prix d'achat": st.column_config.TextColumn(
                                    "Prix d'achat",
                                    help="Prix d'achat de l'actif",
                                    width="medium"
                                ),
                                "Date d'entrée": st.column_config.TextColumn(  # Changé en TextColumn au lieu de DateColumn
                                    "Date d'entrée",
                                    help="Date d'achat de l'actif",
                                    width="medium"
                                )
                            },
                            use_container_width=True
                        )
                    else:
                        st.info("Aucune position dans ce portefeuille")

                # Onglet 2: Ajouter une position
                with tabs[1]:
                    st.markdown("### Nouvelle position")
                    if st.session_state.show_success_add:
                        st.success("✅ Position ajoutée avec succès!")
                        st.session_state.show_success_add = False
                    else:
                        st.session_state.show_success_add = False

                    col_date, col_ticker = st.columns(2)

                    with col_date:
                        new_entry_date = st.date_input(
                            "Date d'entrée",
                            value=datetime.now(),
                            min_value=min_entry_dates,
                            max_value=datetime.now(),
                            format="DD/MM/YYYY",
                            help="Sélectionnez la date d'entrée de la position"
                        )

                    available_tickers = [t for t in cac40_yfinance.keys() if t not in current_tickers]
                    if available_tickers:
                        with col_ticker:
                            new_ticker = st.selectbox(
                                "Sélectionner un actif",
                                available_tickers,
                                help="Choisissez l'actif à ajouter au portefeuille"
                            )

                        new_price = st.number_input(
                            f"Prix d'achat pour {new_ticker}",
                            min_value=0.0,  # On commence à 0 pour pouvoir le détecter
                            step=0.01,
                            help="Entrez le prix d'achat de l'actif",
                            format="%.2f"
                        )


                        if st.button("✅ Ajouter la position", use_container_width=True):
                            try:
                                if new_price <= 0:
                                    st.error("Le montant minimum à investir doit être supérieur à 0€")
                                    return

                                current_tickers[new_ticker] = new_price
                                entry_dates[new_ticker] = new_entry_date.strftime("%d/%m/%Y")
                                update_portfolio(
                                    Dico_ID, identifiant, selected_portfolio,
                                    current_tickers, entry_dates,
                                    "add", new_entry_date.strftime("%d/%m/%Y")
                                )
                                st.session_state.show_success_add = True
                                st.rerun()
                            except Exception as e:
                                st.error(f"Erreur lors de l'ajout : {str(e)}")

                    else:
                        st.warning("⚠️ Tous les actifs disponibles sont déjà dans le portefeuille")

                # Onglet 3: Clôturer une position
                with tabs[2]:
                    st.markdown("### Clôturer une position")
                    if current_tickers:
                        if st.session_state.show_success_close:
                            st.success("✅ Position clôturée avec succès!")
                            st.session_state.show_success_close = False
                        else:
                            st.session_state.show_success_close = False
                        col_ticker, col_date = st.columns(2)

                        with col_ticker:
                            ticker_to_close = st.selectbox(
                                "Position à clôturer",
                                list(current_tickers.keys()),
                                help="Sélectionnez la position à clôturer"
                            )

                        with col_date:
                            exit_date = st.date_input(
                                "Date de sortie",
                                value=datetime.now(),
                                min_value=datetime.strptime(entry_dates[ticker_to_close], "%d/%m/%Y"),
                                max_value=datetime.now(),
                                format="DD/MM/YYYY",
                                help="Sélectionnez la date de sortie"
                            )

                        calculated_price = get_price_at_date(
                            ticker_to_close,
                            exit_date,
                            entry_dates[ticker_to_close],
                            current_tickers[ticker_to_close],
                            cac40_yfinance
                        )


                        if calculated_price is not None:
                            st.info(f"💡 Prix calculé au {exit_date.strftime('%d/%m/%Y')} : {calculated_price:.2f}€")

                            exit_price = st.number_input(
                                f"Prix de vente pour {ticker_to_close}",
                                min_value=0.0,
                                value=float(calculated_price),
                                step=0.01,
                                format="%.2f",
                                help="Entrez le prix de vente de l'actif"
                            )

                            if st.button("🔒 Clôturer la position", use_container_width=True):
                                try:
                                    entry_price = current_tickers[ticker_to_close]
                                    performance = (exit_price - entry_price) / entry_price * 100

                                    del current_tickers[ticker_to_close]
                                    del entry_dates[ticker_to_close]

                                    update_portfolio(
                                        Dico_ID, identifiant, selected_portfolio,
                                        current_tickers, entry_dates,
                                        "close", exit_date.strftime("%d/%m/%Y"),
                                        {
                                            "ticker": ticker_to_close,
                                            "entry_price": entry_price,
                                            "exit_price": exit_price,
                                            "performance": performance
                                        }
                                    )

                                    st.session_state.show_success_close = True
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Erreur lors de la clôture : {str(e)}")
                    else:
                        st.info("ℹ️ Aucune position à clôturer")

                # Onglet 4: Historique
                with tabs[3]:
                    st.markdown("### Historique des transactions")
                    display_transaction_history(identifiant, selected_portfolio)

#VAR

def run_var_analysis(portfolio_data):
    st.title('Value at Risk')


    # Création de deux colonnes
    col1, col2 = st.columns(2)

    # Dans la première colonne : slider pour le niveau de confiance
    with col1:
        alpha = st.slider(
            'Niveau de confiance (1-α)',
            min_value=0.90,
            max_value=0.99,
            value=0.95,  # valeur par défaut
            step=0.01,
            help="Plus le niveau de confiance est élevé, plus la VaR sera conservative"
        )
        alpha = 1 - alpha

    # Dans la deuxième colonne : slider pour l'horizon temporel
    with col2:
        T = st.slider(
            'Horizon temporel (jours)',
            min_value=1,
            max_value=30,  # vous pouvez ajuster cette valeur maximale selon vos besoins
            value=1,  # valeur par défaut
            step=1,
            help="Nombre de jours sur lequel la VaR est calculée"
        )

    tickers_dict = portfolio_data["tickers"]
    Base_Ticker = list(tickers_dict.keys())
    purchase_prices = list(tickers_dict.values())
    entry_dates = portfolio_data["dates"]
    Time_Period = portfolio_data["creation_date"]

    total_prices = sum(purchase_prices)

    ponderations = [purchase_prices[i]/total_prices for i in range(len(purchase_prices))]

    List_Ticker = [cac40_yfinance[ticker] for ticker in Base_Ticker if ticker in cac40_yfinance]

    Rdt = PriceToRdt(List_Ticker, entry_dates)

    Bootstrapped = Bootstraap_Rdt(Rdt, 100)

    Portfolio_Rdt = Rdt @ ponderations

    VaR_HS1 = VaR_HS(Bootstrapped, ponderations, alpha) * (T)**(1/2)
    VaR_VARCOV1 = VaR_VARCOV(Rdt, ponderations, alpha)* (T)**(1/2)
    RM = RiskMetrics(Rdt, alpha)* (T)**(1/2)
    VaR_Garch = VaR_GARCH(Rdt, ponderations, alpha)* (T)**(1/2)
    VaR_CF1 = VaR_CF(Rdt,ponderations,alpha)* (T)**(1/2)

    df = pd.DataFrame([VaR_HS1,VaR_VARCOV1,RM,VaR_Garch,VaR_CF1],index = ['VaR HS' , 'VaR VARCOV' , 'Riskmetrics' , 'Garch', 'Cornish fisher'],columns = ['Valeur'])

    # Multiplier les valeurs par 100 pour les convertir en pourcentage
    df['Valeur'] = df['Valeur'] * 100

    # Formater les valeurs en ajoutant le symbole %
    df['Valeur'] = df['Valeur'].apply(lambda x: f"{x:.2f} %")

    df['BT Kupiec'] = [Kupiec(Portfolio_Rdt,-VaR_HS1,alpha),Kupiec(Portfolio_Rdt,-VaR_VARCOV1,alpha),Kupiec(Portfolio_Rdt,-RM,alpha),Kupiec(Portfolio_Rdt,-VaR_Garch,alpha),Kupiec(Portfolio_Rdt,-VaR_CF1,alpha)]
    df['BT Christoffersen'] = [LR_christoffersen(Portfolio_Rdt,-VaR_HS1,alpha),LR_christoffersen(Portfolio_Rdt,-VaR_VARCOV1,alpha),LR_christoffersen(Portfolio_Rdt,-RM,alpha),LR_christoffersen(Portfolio_Rdt,-VaR_Garch,alpha),LR_christoffersen(Portfolio_Rdt,-VaR_CF1,alpha)]
    df['BT LjungBox'] = [Auto_cor_LB(Portfolio_Rdt,-VaR_HS1,alpha),Auto_cor_LB(Portfolio_Rdt,-VaR_VARCOV1,alpha),Auto_cor_LB(Portfolio_Rdt,-RM,alpha),Auto_cor_LB(Portfolio_Rdt,-VaR_Garch,alpha),Auto_cor_LB(Portfolio_Rdt,-VaR_CF1,alpha)]
    df['BT CP'] = [CP(Portfolio_Rdt,-VaR_HS1,alpha),CP(Portfolio_Rdt,-VaR_VARCOV1,alpha),CP(Portfolio_Rdt,-RM,alpha),CP(Portfolio_Rdt,-VaR_Garch,alpha),CP(Portfolio_Rdt,-VaR_CF1,alpha)]


    df['BT Kupiec'] = np.where(df['BT Kupiec'] > 3.841, '❌', '✅')
    df['BT Christoffersen'] = np.where(df['BT Christoffersen'] > 5.991, '❌', '✅')
    df['BT LjungBox'] = np.where(df['BT LjungBox'] < 0.05, '❌', '✅')
    df['BT CP'] = np.where(df['BT CP'] > alpha, '❌', '✅')


    st.table(df)

    plot_var_violations(portfolio_data, df)

    return df

def plot_var_violations(portfolio_data, df):
    """
    Crée un graphique des violations VaR historiques pour le portefeuille
    La couleur de la croix correspond à la VaR la plus grande qui a été violée
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from datetime import datetime

    # Récupération des données nécessaires
    tickers_dict = portfolio_data["tickers"]
    Base_Ticker = list(tickers_dict.keys())
    purchase_prices = list(tickers_dict.values())
    dates_dict = portfolio_data["dates"]

    total_prices = sum(purchase_prices)
    ponderations = [purchase_prices[i]/total_prices for i in range(len(purchase_prices))]

    # Calcul des rendements du portefeuille
    List_Ticker = [cac40_yfinance[ticker] for ticker in Base_Ticker if ticker in cac40_yfinance]

    # Obtenir les rendements avec la fonction existante
    Rdt = PriceToRdt(List_Ticker, dates_dict)
    Portfolio_Rdt = Rdt @ ponderations

    # Convertir en Series avec l'index de dates existant
    Portfolio_Rdt = pd.Series(Portfolio_Rdt, index=Rdt.index)

    # Sélection des 10% dernières données
    n_last = int(len(Portfolio_Rdt) * 0.1)
    Portfolio_Rdt_recent = Portfolio_Rdt.tail(n_last)

    # Création du graphique
    plt.figure(figsize=(15, 8))

    # Tracer les rendements
    plt.plot(Portfolio_Rdt_recent.index, Portfolio_Rdt_recent.values,
             label='Rendements', color='black', alpha=1, linewidth=1)

    # Couleurs pour les différentes méthodes VaR
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    var_methods = ['VaR HS', 'VaR VARCOV', 'Riskmetrics', 'Garch', 'Cornish fisher']

    # Créer un dictionnaire des valeurs VaR pour un accès plus facile
    var_values = {method: -float(df.loc[method, 'Valeur'].strip(' %'))/100
                 for method in var_methods}

    # Pour chaque point de données, trouver la plus grande VaR violée
    for idx, rendement in Portfolio_Rdt_recent.items():
        violations = []

        # Vérifier les violations pour chaque méthode
        for method, color in zip(var_methods, colors):
            if rendement < var_values[method]:
                violations.append((method, var_values[method], color))

        # S'il y a des violations, trouver celle avec la plus grande VaR
        if violations:
            # Trier par valeur VaR (la plus grande en valeur absolue)
            violations.sort(key=lambda x: abs(x[1]), reverse=True)
            method, var_value, color = violations[0]

            plt.scatter([idx], [rendement],
                       color=color, marker='x', s=100)

    # Tracer les lignes VaR
    for method, color in zip(var_methods, colors):
        plt.axhline(y=var_values[method], color=color, linestyle='--',
                   label=f'{method} ({df.loc[method, "Valeur"]})')

    plt.title('Comparaison des VaR et violations (10% dernières données)', pad=20)
    plt.xlabel('Date')
    plt.ylabel('Rendement')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Rotation des dates pour une meilleure lisibilité
    plt.xticks(rotation=45)
    plt.margins(x=0.02)

    # Ajuster la mise en page pour éviter que la légende soit coupée
    plt.tight_layout()

    # Afficher le graphique dans Streamlit
    st.pyplot(plt)

    # Afficher quelques statistiques sur les violations
    st.write("Statistiques des violations sur la période récente:")
    stats = {}
    for method in var_methods:
        violations = Portfolio_Rdt_recent < var_values[method]
        violation_rate = (violations.sum() / len(Portfolio_Rdt_recent)) * 100
        stats[method] = f"{violation_rate:.1f}%"

    st.dataframe(pd.DataFrame([stats], index=['Taux de violation']), hide_index=True, use_container_width=True)
    
# ======================== Page de Dashboard ======================== #

def page_dashboard():
    Identifiant = st.session_state.Identifiant
    st.title("Dashboard " + Identifiant)
    # Boutons de navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Nouveau portefeuille"):
            st.session_state.current_page = "new_portfolio"
            st.rerun()
    with col3:
        if st.button("Déconnexion"):
            st.session_state.authenticated = False
            st.rerun()

    if Identifiant in Dico_ID:
        portfolios = get_user_portfolios(Dico_ID, Identifiant)
        portfolio_names = list(portfolios.keys())

        default_portfolio = st.session_state.get('selected_portfolio', portfolio_names[0] if portfolio_names else None)

        if len(portfolio_names) > 0:
            selected_portfolio = st.selectbox("Sélectionner un portefeuille",
                                              portfolio_names,
                                              index=portfolio_names.index(default_portfolio) if default_portfolio in portfolio_names else 0
                                              )
            with col1:
                if st.button("Gérer le portefeuille"):
                    st.session_state.selected_portfolio = selected_portfolio
                    st.session_state.current_page = "portfolio"
                    st.rerun()

            portfolio_data = portfolios[selected_portfolio]

            tickers_dict = portfolio_data["tickers"]
            entry_dates = portfolio_data["dates"]
            Base_Ticker = list(tickers_dict.keys())
            purchase_prices = list(tickers_dict.values())

            # Calculer le total des prix d'achat
            total_prices = sum(purchase_prices)
            ponderations = [price/total_prices for price in purchase_prices]

            List_Ticker = [cac40_yfinance[ticker] for ticker in Base_Ticker if ticker in cac40_yfinance]

            # Utiliser les dates d'entrée individuelles pour le calcul des rendements
            Rdt = PriceToRdt(List_Ticker, entry_dates)

            # Calcul des performances
            Portfolio_Rdt = Rdt @ ponderations
            Capital_initial = total_prices
            Portfolio = calculate_portfolio_value(tickers_dict, entry_dates, ponderations, Rdt)

            # Calcul des indicateurs
            annual_return = Portfolio_Rdt.mean() * 252
            annual_volatility = Portfolio_Rdt.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility
            max_drawdown = (Portfolio / Portfolio.cummax() - 1).min()
            sortino_ratio = annual_return / (Portfolio_Rdt[Portfolio_Rdt < 0].std() * np.sqrt(252))

            performance_data = {
                'Indicateur': [
                    'Rendement Annuel',
                    'Volatilité Annuelle',
                    'Sharpe Ratio',
                    'Maximum Drawdown',
                    'Ratio de Sortino'
                ],
                'Valeur': [
                    f'{annual_return*100:.2f} %',
                    f'{annual_volatility*100:.2f} %',
                    f'{sharpe_ratio:.2f}',
                    f'{max_drawdown*100:.2f} %',
                    f'{sortino_ratio:.2f}'
                ]
            }

            performance_df = pd.DataFrame(performance_data)

            # Passer les dates d'entrée à get_Df_Rdts
            get_Df_Rdts(Dico_ID, Portfolio, Capital_initial, performance_df, entry_dates, Identifiant, selected_portfolio)
            prices_df = get_last_prices_df(List_Ticker, Base_Ticker, tickers_dict, entry_dates, cac40_yfinance)

            # Récupérer l'historique des transactions
            transaction_file = f"transactions_{Identifiant}.csv"
            try:
                transactions_df = pd.read_csv(transaction_file)
            except FileNotFoundError:
                transactions_df = pd.DataFrame()

        else:
            st.warning("Aucun portefeuille disponible.")

        VaR_Value = run_var_analysis(portfolio_data)

        btn_del1, btn_del2= st.columns([1, 1])
        with btn_del1:
            add_excel_download_button_dynamic(portfolio_data, Portfolio, Capital_initial,
                                    performance_df, transactions_df, prices_df,
                                    Identifiant, selected_portfolio, VaR_Value)

        with btn_del2:
            add_pdf_download_button_dynamic(portfolio_data, Portfolio, Capital_initial,
                                   performance_df, transactions_df, prices_df,
                                   Identifiant, selected_portfolio, VaR_Value)


# ======================== Main ========================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "page" not in st.session_state:
    st.session_state.page = "login"

if "current_page" not in st.session_state:
    st.session_state.current_page = "dashboard"

if 'show_success_add' not in st.session_state:
    st.session_state['show_success_add'] = False

if 'show_success_close' not in st.session_state:
    st.session_state['show_success_close'] = False

Dico_ID = get_Data(DATA_FILEPATH)

# Gestion de la navigation
if st.session_state.authenticated:
    if st.session_state.current_page == "dashboard":
        page_dashboard()
    elif st.session_state.current_page == "portfolio":
        page_portfolio_management()
    elif st.session_state.current_page == "new_portfolio":
        page_new_portfolio()
else:
    if st.session_state.page == "create_account":
        page_create(Dico_ID, cac40_yfinance)
    else:
        page_connexion()