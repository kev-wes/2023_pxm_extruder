# Import necessary packages
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import colorsys
import joblib

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import (
    DBSCAN, 
    OPTICS, 
    AgglomerativeClustering, 
    KMeans
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, 
    accuracy_score, 
    mean_squared_error
)
from sklearn.model_selection import GridSearchCV

import xgboost as xgb

import plotly.graph_objs as go

from shiny import (
    ui, 
    render, 
    App, 
    Inputs, 
    Outputs, 
    Session, 
    reactive
)
import shinyswatch
from htmltools import css
from shiny.ui import h2, tags

# Define the UI component of the dashboard
app_ui = ui.page_fluid(
    ui.panel_title("", "Condition-based Maintenance at Bischof+Klein"),
    ui.tags.style(
        """"
        body {
            font-family: Trebuchet MS
        }
        """
    ),
    {"style": "background-color: rgba(0, 158, 227, 0.1)"},
    ui.navset_tab(
        # Define UI for first tab
        ui.nav(
            "Übersicht",
            ui.input_date_range(
                "date_range",
                "Zeitraum",
                start="2022-05-15",
                end="2022-05-16",
                format="dd.mm.yyyy",
                language="de",
                min="2022-04-25",
                max="2022-07-11",
                separator="bis",
            ),
            ui.panel_well(
                ui.h4("Funktion dieses Reiters:"),
                ui.markdown(
                    """
                    **In diesem Reiter befinden sich allgemeine Informationen zu der Maschine.**
                    """
                ),
            ),
            ui.panel_well(
                ui.h4("Übersicht Qualität"),
                ui.markdown(
                    """
                In diesem Abschnitt befinden sich die Durchschnittswerte von zwei zentralen **Qualitätskennzahlen (Folienprofil 3-Sigma [%] & Gleitende Profiltoleranz [%])** zur Überprüfung der Prozess-Performance innerhalb des vom Nutzer gewünschten Betrachtungszeitraums. Des weiteren können die **Qualitätsabweichungen im Zeitverlauf (Folienprofil 3-Sigma [%])** betrachtet werden.
                """
                ),
                ui.row(
                    ui.column(
                        6, ui.markdown("**Durchschnittswert Folienprofil 3-Sigma [%]**")
                    ),
                    ui.column(
                        6,
                        ui.markdown(
                            "**Durchschnittswert Gleitende Profiltoleranz [%]**"
                        ),
                    ),
                ),
                ui.row(
                    ui.column(6, ui.output_text_verbatim("sigma")),
                    ui.column(6, ui.output_text_verbatim("profil")),
                ),
                ui.row(
                    ui.output_plot("quality_sigma"),
                ),
            ),
            ui.panel_well(
                ui.h4("Übersicht Energie"),
                ui.markdown(
                    """
                In diesem Abschnitt befinden sich die Top 5 Ausreißer der **Leistungs- und Prozessabwärme-Messungen** innerhalb des vom Nutzer gewählten Betrachtungszeitraums. Des weiteren können beide Features auch im Zeitverlauf betrachtet werden.
                """
                ),
                ui.row(
                    ui.column(6, ui.markdown("**Top 5 Spikes: Leistung [kW]**")),
                    ui.column(6, ui.markdown("**Top 5 Spikes: Prozessabwärme [kW]**")),
                ),
                ui.row(
                    ui.column(6, ui.output_table("energy_spikes")),
                    ui.column(6, ui.output_table("abwaerme_spikes")),
                ),
                ui.output_plot("energy_plot"),
                ui.output_plot("abwaerme_plot"),
            ),
            ui.panel_well(
                ui.h4("Liste der letzten 10 Alarme"),
                ui.markdown(
                    """
                In diesem Abschnitt befinden sind die zum aktuellen Zeitpunkt **letzten 10 Alarm-Einträge aus Ruby**.
                Diese wurden durch aussagekräftige **Kennzahlen der Maschine** zu dem Zeitpunkt des Alarmseintrags ergänzt. 
                """
                ),
                ui.output_table("alarm_table"),
            ),
            ui.panel_well(
                ui.h4("Liste der letzten 10 gelaufenen Produkte"),
                ui.markdown(
                    """
                In diesem Abschnitt befinden sich die zum aktuellen Zeitpunkt **letzten 10 gelaufenen Produkte**, wann die Produktion gestartet und wann beendet wurde.
                """
                ),
                ui.output_table("JobHistory"),
            ),
        ),
        # Define UI for second tab
        ui.nav(
            "Rezeptanalyse",
            ui.panel_well(
                ui.h4("Funktion dieses Reiters:"),
                ui.markdown(
                    """
                    **In diesem Reiter können die Hauptkomponenten und deren Anteil für ein Produkt eingegeben werden,
                    um relevante Maschinendaten für dieses Produkt zu erhalten.**
                    """
                ),
            ),
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.h4("Input für die Rezeptanalyse"),
                    ui.markdown(
                        """
                        Bitte geben Sie die **drei Hauptkomponenten** an, die für das zu analysierende Produkt in den Extrudern A, B und C verwendet werden. Zusätzlich wird der **Anteil der Hauptkomponenten** an der Mischung im jeweiligen Extruder benötigt.
                        """
                    ),
                    ui.input_select(
                        "Extruder01_material",
                        "Extruder A HK:",
                        choices=[
                            "1018MA",
                            "118NJ",
                            "1840D",
                            "2420D",
                            "2420F",
                            "2420FS.21",
                            "2426F",
                            "2501",
                            "3020F",
                            "330E",
                            "361BW",
                            "BA110CF",
                            "D139",
                            "FA03-01",
                            "FB4230",
                            "FL00018",
                            "FT5230",
                            "M5510EP",
                            "M6012EP",
                        ],
                    ),
                    ui.input_slider(
                        "Extruder01_percentage",
                        "Extruder A HK Anteil [%]:",
                        min=0,
                        max=100,
                        value=50,
                    ),
                    ui.input_select(
                        "Extruder02_material",
                        "Extruder B HK:",
                        choices=[
                            "1018MA",
                            "SP1510",
                            "BA110CF",
                            "D139",
                            "118NJ",
                            "1840D",
                            "2420D",
                            "2420F",
                            "BA110CFS.16",
                            "2426F",
                            "2501",
                            "FT5230",
                            "3020F",
                            "330E",
                            "361BW",
                            "M5510EP",
                            "FA03-01",
                            "FB4230",
                            "M6012EP",
                            "FL00018",
                        ],
                    ),
                    ui.input_slider(
                        "Extruder02_percentage",
                        "Extruder B HK Anteil [%]:",
                        min=0,
                        max=100,
                        value=50,
                    ),
                    ui.input_select(
                        "Extruder03_material",
                        "Extruder C HK:",
                        choices=[
                            "1018MA",
                            "D139",
                            "HP1018BN",
                            "118NJ",
                            "1840D",
                            "361BW",
                            "2420F",
                            "2420FS.21",
                            "2426F",
                            "LD361BW",
                            "FT5230",
                            "3020F",
                            "302F",
                            "330E",
                            "BA110CF",
                            "M5510EP",
                            "FB4230",
                            "M6012EP",
                            "FL00018",
                        ],
                    ),
                    ui.input_slider(
                        "Extruder03_percentage",
                        "Extruder C HK Anteil [%]:",
                        min=0,
                        max=100,
                        value=50,
                    ),
                    ui.markdown(
                        """
                        Sobald Sie die Komponenten und Anteile bestimmt haben, drücken Sie auf **"Submit"**, um die Kalkulation zu starten.
                        """
                    ),
                    ui.input_action_button("submit", "Submit"),
                ),
                ui.panel_main(
                    ui.panel_well(
                        ui.h4("Produkt-Code"),
                        ui.markdown(
                            "Der ermittelte Produkt-Code ist der **Code eines in der Vergangenheit hergestellten Produktes, dessen Zusammensetzung der des zu analysierenden Produktes am ähnlichsten ist.**"
                        ),
                        ui.row(
                            ui.column(2, ui.markdown("**Produkt-Code:**")),
                            ui.column(2, ui.output_text("RA02")),
                        ),
                    ),
                    ui.panel_well(
                        ui.h4("Cluster-Informationen"),
                        ui.markdown(
                            """In diesem Abschnitt finden sich **Informationen zum Cluster des ermittelten Produkt-Codes**.
                            Initial werden alle bekannten Produkte nach ihrer Ähnlichkeit gruppiert. Dadurch wird ermöglicht, dass analysierte Rezepte auf Grundlage ihres Produkt-Codes direkt einem **Cluster zugeordnet** werden können.
                            Unterhalb der Cluster-Nummer finden sich **Durchschnittswerte über alle Produkte des Clusters** für relevante Kennzahlen.
                            Diese werden anschließend in **Relation zu den Durchschnittswerten über alle Cluster** gesetzt."""
                        ),
                        ui.row(
                            ui.column(3, ui.markdown("**Cluster-Nummer:**")),
                            ui.column(2, ui.output_text("cluster_number_output")),
                        ),
                        ui.markdown("**Durchschnittswerte des Clusters:**"),
                        ui.output_table("cluster_specs"),
                        ui.markdown(
                            "**Durchschnittswerte des Clusters in Relation zu den Durschnittswerten über alle Cluster:**"
                        ),
                        ui.output_table("relative_cluster_values"),
                        ui.markdown(
                            "Die in der Tabelle gegebenen Korridore geben für das ermittelte Cluster an, **innerhalb welcher Grenzen** sich zentrale Kennzahlen bestenfalls bewegen sollten."
                        ),
                        ui.output_table("corridors"),
                        # TODO info buttons, that give more description for the specs
                        # TODO description and header for cluster
                        ui.output_plot("product_cluster", width="100%", height="750px"),
                    ),
                    ui.panel_well(
                        ui.h4("Korridore für anpassbare Werte"),
                        ui.markdown(
                            "Hier befinden sich **Werte, die der Maschinenführer aktiv einstellen kann**. Sie dienen als **Orientierung**, welche **Unter- und Obergrenzen** für die einzelnen Einstellungen eingehalten werden sollten."
                        ),
                        ui.output_table("corridors_soll"),
                    ),
                    ui.panel_well(
                        ui.h4("Qualitätskennzahlen"),
                        ui.markdown(
                            "Basierend auf dem ermittelten Cluster werden in diesem Abschnitt **Qualitätskennzahlen** ausgegeben."
                        ),
                        # TODO give header and description to quality output
                        ui.row(
                            ui.column(6, ui.markdown("**Bahngeschwindigkeit:**")),
                            ui.column(2, ui.output_text("quality_bahngeschwindigkeit")),
                        ),
                        ui.row(
                            ui.column(6, ui.markdown("**Produktanzahl:**")),
                            ui.column(2, ui.output_text("quality_productCount")),
                        ),
                        ui.row(
                            ui.column(6, ui.markdown("**Anzahl der Spikes:**")),
                            ui.column(2, ui.output_text("spikesCount")),
                        ),
                        ui.row(
                            ui.column(
                                6,
                                ui.markdown(
                                    "**Anteil der Spikes an der Gesamtzahl der Datenpunkte:**"
                                ),
                            ),
                            ui.column(2, ui.output_text("proportionSpikes")),
                        ),
                        ui.row(
                            ui.column(6, ui.markdown("**Ausschuss (total):**")),
                            ui.column(2, ui.output_text("totalWaste")),
                        ),
                        ui.row(
                            ui.column(6, ui.markdown("**Ausschuss (prozentual):**")),
                            ui.column(2, ui.output_text("averageWaste")),
                        ),
                    ),
                ),
            ),
        ),
        # Define UI for third tab
        ui.nav(
            "Qualitätsanalyse",
            ui.input_date_range(
                "date_range2",
                "Zeitraum",
                start="2022-05-15",
                end="2022-05-16",
                format="dd.mm.yyyy",
                language="de",
                min="2022-04-25",
                max="2022-07-11",
                separator="bis",
            ),
            ui.panel_well(
                ui.h4("Funktion dieses Reiters:"),
                ui.markdown(
                    """
                    **In diesem Reiter werden Qualitätskennzahlen und -informationen basierend auf dem gewählten Zeitraum
                    und der gewählten Cluster angezeigt.**
                    """
                ),
            ),
            ui.panel_well(
                ui.h4("Spikes in der Qualitätsabweichung"),
                ui.markdown(
                    "Durch diesen Graphen werden **Qualitätsabweichungen des Folienprofils**, die 8% überschreiten, identifiziert."
                ),
                ui.output_plot("spikes_plot"),
            ),
            ui.panel_well(
                ui.h4("Qualitätskennzahlen der Cluster"),
                ui.markdown(
                    "Der Nutzer kann die **zu analysierenden Cluster auswählen** (siehe Reiter Rezeptanalyse), um sich die entsprechenden **Qualitätskennzahlen** ausgeben lassen."
                ),
                ui.input_selectize(
                    "cluster",
                    "Cluster auswählen:",
                    choices=[f"Cluster {i}" for i in range(-1, 24)],
                    selected=None,
                    multiple=True,
                ),
                ui.output_table("quality_values_by_cluster"),
            ),
            ui.panel_well(
                ui.row(
                    ui.h4("Qualitätsabweichungen über den gesamten Zeitraum"),
                    ui.markdown(
                        """
                    In diesem Abschnitt ist die Qualitätskennzahl **Folienprofil 3-Sigma [%]** über den gesamten Zeitraum 
                    des zugrundeliegenden Datensatzes abgebildet. Insgesamt ist die Qualität in **5 Stufen** eingeteilt, wobei für jede Stufe angezeigt wird,
                    wie viele Messungen innerhalb der festgelegten Range liegen.
                    """
                    ),
                    ui.column(5, ui.output_table("quality_distribution_tbl")),
                    ui.column(
                        7, ui.output_plot("quality_distribution_plt", width="100%")
                    ),
                )
            ),
        ),
    ),
)

# Define the SERVER component of the dashboard 
def server(input, output, session):
    # Define function to load numeric data
    def df():
        infile = Path(__file__).parent / "reduced_numeric_data.csv"
        df = pd.read_csv(infile)
        # Use the DataFrame's to_html() function to convert it to an HTML table, and
        # then wrap with ui.HTML() so Shiny knows to treat it as raw HTML.
        df["local_time"] = pd.to_datetime(df["local_time"])
        return df
    # Define function to load alarm data
    def df_alarm():
        infile = Path(__file__).parent / "Warnings.csv"
        df_alarm = pd.read_csv(
            infile, encoding="unicode_escape", on_bad_lines="skip", sep=";"
        )

        # Cast Start column to datetime object
        format_string = "%d.%m.%y %H:%M"
        df_alarm["Start"] = pd.to_datetime(
            df_alarm["Start"], format=format_string, dayfirst=True
        )

        df_alarm["End"] = pd.to_datetime(
            df_alarm["End"], format=format_string, dayfirst=True
        )

        df_alarm = df_alarm.sort_values("Start", ascending=False)
        df_alarm = df_alarm.drop(
            labels=[
                "Summarized Warnings",
                "Type",
                "Possible Failure",
                "Mark here if relevant for Maintenance",
            ],
            axis=1,
        )

        # Rename the columns
        df_alarm = df_alarm.rename(
            columns={
                "Start": "Startzeitpunkt",
                "End": "Endzeitpunkt",
                "Comp": "Sub-Komponente",
                "Warning Text": "Warnungstext",
                "Component": "Komponente",
            }
        )

        # Replace 0 values in the "Komponente" column with a string
        df_alarm["Komponente"] = df_alarm["Komponente"].replace(
            "0", "Keine Komponente gegeben"
        )

        # Format the datetime columns
        # df_alarm["Startzeitpunkt"] = df_alarm["Startzeitpunkt"].dt.strftime("%d.%m.%Y %H:%M")
        # df_alarm["Endzeitpunkt"] = df_alarm["Endzeitpunkt"].dt.strftime("%d.%m.%Y %H:%M")

        # Filter rows with Startzeitpunkt before 11.07.2023
        df_alarm = df_alarm[df_alarm["Startzeitpunkt"] < pd.to_datetime("2022-07-11")]

        # Reorder the columns
        desired_order = [
            "Startzeitpunkt",
            "Endzeitpunkt",
            "Komponente",
            "Sub-Komponente",
            "Warnungstext",
        ]
        df_alarm = df_alarm.reindex(columns=desired_order)

        return df_alarm
    
    # Save data in variables to decrease computation time
    df = df()
    df_alarm = df_alarm()

    # Function to depict the average value of "Folienprofil 3-Sigma"
    @output
    @render.text
    def sigma():
        # Load Data
        data = df

        # Define the specific time span
        start_time = pd.to_datetime(input.date_range()[0])
        end_time = pd.to_datetime(input.date_range()[1])

        # Select rows within the time span
        selected_rows = data[
            (data["local_time"] >= start_time) & (data["local_time"] <= end_time)
        ]

        # Calculate KPIs
        avg_sigma = selected_rows["Folienprofil 3-Sigma [%]"].mean()

        # Round the results to two decimals
        avg_sigma_rounded = round(avg_sigma, 2)

        # Concatenate the "[%]" sign to the results
        avg_sigma_formatted = f"{avg_sigma_rounded}%"

        # Return KPI
        return avg_sigma_formatted

    # Function to depict the average value of "Gleitende Profiltoleranz"
    @output
    @render.text
    def profil():
        # Load Data
        data = df

        # Define the specific time span
        start_time = pd.to_datetime(input.date_range()[0])
        end_time = pd.to_datetime(input.date_range()[1])

        # Select rows within the time span
        selected_rows = data[
            (data["local_time"] >= start_time) & (data["local_time"] <= end_time)
        ]

        # Calculate KPI
        avg_profil = selected_rows["Gleitende Profiltoleranz [%]"].mean()

        # Round the results to two decimals
        avg_profil_rounded = round(avg_profil, 2)

        # Concatenate the "[%]" sign to the results
        avg_profil_formatted = f"{avg_profil_rounded}%"

        # Return KPI
        return avg_profil_formatted

    # Function to depict the value of "Folienprofil 3-Sigma" over time
    @output
    @render.plot
    def quality_sigma():
        df3 = df
        df3["local_time"] = pd.to_datetime(df3["local_time"])

        # Define the specific time span
        start_time = pd.to_datetime(input.date_range()[0])
        end_time = pd.to_datetime(input.date_range()[1])

        # Select rows within the time span
        selected_rows = df3[
            (df3["local_time"] >= start_time) & (df3["local_time"] <= end_time)
        ]

        # Plotting
        # Set the background to full transparency
        fig = plt.figure(facecolor=(0, 0, 0, 0))
        ax = fig.add_subplot(111)

        ax.plot(
            selected_rows["local_time"],
            selected_rows["Folienprofil 3-Sigma [%]"],
            color=(0, 0.1098, 0.2902),
        )
        ax.set_xlabel("Zeit", fontdict={"family": "Trebuchet MS", "size": 12})
        ax.set_ylabel(
            "Folienprofil 3-Sigma [%]", fontdict={"family": "Trebuchet MS", "size": 12}
        )
        ax.set_title(
            "Qualitätsabweichungen im Zeitverlauf",
            fontweight="bold",
            fontdict={"family": "Trebuchet MS", "size": 14},
        )
        ax.tick_params(axis="x", rotation=45)

        # Format x-axis labels to "dd.mm.yyyy hh:mm"
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y %H:%M"))

        # Remove upper and right boundaries
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Make inside of graph transparent
        ax.set_facecolor((0, 0, 0, 0))

    # Function to depict the value of "Leistung" over time
    @output
    @render.plot
    def energy_plot():
        df2 = df
        df2["local_time"] = pd.to_datetime(df2["local_time"])

        # Define the specific time span
        start_time = pd.to_datetime(input.date_range()[0])
        end_time = pd.to_datetime(input.date_range()[1])

        # Select rows within the time span
        selected_rows = df2[
            (df2["local_time"] >= start_time) & (df2["local_time"] <= end_time)
        ]

        # Plotting
        # Set the background to full transparency
        fig = plt.figure(facecolor="none")
        ax = fig.add_subplot(111)

        ax.plot(
            selected_rows["local_time"],
            selected_rows["Leistung [kW]"],
            color=(0, 0.1098, 0.2902),
        )
        ax.set_xlabel("Zeit", fontdict={"family": "Trebuchet MS", "size": 12})
        ax.set_ylabel("Leistung [kW]", fontdict={"family": "Trebuchet MS", "size": 12})
        ax.set_title(
            "Leistung im Zeitverlauf",
            fontweight="bold",
            fontdict={"family": "Trebuchet MS", "size": 14},
        )
        ax.tick_params(axis="x", rotation=45)

        # Format x-axis labels to "dd.mm.yyyy hh:mm"
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y %H:%M"))

        # Remove upper and right boundaries
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Make inside of graph transparent
        ax.set_facecolor((0, 0, 0, 0))

    # Function to depict the top five values of "Leistung" in specified timeframe
    @output
    @render.table
    def energy_spikes():
        # Load dataset
        df_energy = df

        # Define the specific time span
        start_time = pd.to_datetime(input.date_range()[0])
        end_time = pd.to_datetime(input.date_range()[1])

        # Select rows within the time span
        selected_rows = df_energy[
            (df_energy["local_time"] >= start_time)
            & (df_energy["local_time"] <= end_time)
        ]

        # Select top 5 rows with most energy consumption
        selected_rows = selected_rows.sort_values("Leistung [kW]", ascending=False)
        top5_spikes = selected_rows[["local_time", "Leistung [kW]"]].head(5)
        top5_spikes["Leistung [kW]"] = top5_spikes["Leistung [kW]"].round(2)

        # Rename the first column
        top5_spikes = top5_spikes.rename(columns={"local_time": "Zeitpunkt"})

        # Format the timestamp to "dd-mm-yyyy hh:mm:ss"
        top5_spikes["Zeitpunkt"] = top5_spikes["Zeitpunkt"].dt.strftime(
            "%d.%m.%Y %H:%M:%S"
        )

        return top5_spikes

    # Function to depict the top five values of "Prozessabwärme" in specified timeframe
    @output
    @render.table
    def abwaerme_spikes():
        # Load dataset
        df_abwaerme = df

        # Define the specific time span
        start_time = pd.to_datetime(input.date_range()[0])
        end_time = pd.to_datetime(input.date_range()[1])

        # Select rows within the time span
        selected_rows = df_abwaerme[
            (df_abwaerme["local_time"] >= start_time)
            & (df_abwaerme["local_time"] <= end_time)
        ]

        # Select top 5 rows with most energy consumption
        selected_rows = selected_rows.sort_values(
            "Prozessabwärme [kW]", ascending=False
        )
        top5_spikes = selected_rows[["local_time", "Prozessabwärme [kW]"]].head(5)
        top5_spikes["Prozessabwärme [kW]"] = top5_spikes["Prozessabwärme [kW]"].round(2)

        # Rename the first column
        top5_spikes = top5_spikes.rename(columns={"local_time": "Zeitpunkt"})

        # Format the timestamp to "dd-mm-yyyy hh:mm:ss"
        top5_spikes["Zeitpunkt"] = top5_spikes["Zeitpunkt"].dt.strftime(
            "%d.%m.%Y %H:%M:%S"
        )

        return top5_spikes

    # Function to depict the value of "Prozessabwärme" over time
    @output
    @render.plot
    def abwaerme_plot():
        df2 = df
        df2["local_time"] = pd.to_datetime(df2["local_time"])

        # Define the specific time span
        start_time = pd.to_datetime(input.date_range()[0])
        end_time = pd.to_datetime(input.date_range()[1])

        # Select rows within the time span
        selected_rows = df2[
            (df2["local_time"] >= start_time) & (df2["local_time"] <= end_time)
        ]

        # Plotting
        # Set the background to full transparency
        fig = plt.figure(facecolor="none")
        ax = fig.add_subplot(111)

        ax.plot(
            selected_rows["local_time"],
            selected_rows["Prozessabwärme [kW]"],
            color=(0, 0.1098, 0.2902),
        )
        ax.set_xlabel("Zeit", fontdict={"family": "Trebuchet MS", "size": 12})
        ax.set_ylabel(
            "Prozessabwärme [kW]", fontdict={"family": "Trebuchet MS", "size": 12}
        )
        ax.set_title(
            "Prozessabwärme im Zeitverlauf",
            fontweight="bold",
            fontdict={"family": "Trebuchet MS", "size": 14},
        )
        ax.tick_params(axis="x", rotation=45)

        # Format x-axis labels to "dd.mm.yyyy hh:mm"
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y %H:%M"))

        # Remove upper and right boundaries
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Make inside of graph transparent
        ax.set_facecolor((0, 0, 0, 0))

    # Function to depict the last ten alerts of the dataset including selected machine values
    @output
    @render.table
    def alarm_table():
        # Save alarm data in df via helper function
        df_alarm1 = df_alarm

        # Create subset df only containing most recent 10 alarms
        top10_recent_alarm = df_alarm1.head(10)

        # New df for selected alarms + machine data

        alarms_detail = pd.DataFrame()

        # Loop to concatenate alarms with machine data
        for _, alarm_row in top10_recent_alarm.iterrows():
            alarm_timestamp = alarm_row["Startzeitpunkt"]
            machine_data = df[
                df["local_time"] == alarm_timestamp
            ]  # Assuming Timestamp column name in df

            if not machine_data.empty:
                merged_data = pd.concat([alarm_row, machine_data.iloc[0]])
                alarms_detail = pd.concat([alarms_detail, merged_data], axis=1)
            else:
                alarms_detail = pd.concat([alarms_detail, alarm_row], axis=1)

        alarms_detail = alarms_detail.T.reset_index(
            drop=True
        )  # Transpose and reset index

        desired_columns = [
            "Startzeitpunkt",
            "Endzeitpunkt",
            "Komponente",
            "Sub-Komponente",
            "Warnungstext",
            "Folienprofil 3-Sigma [%]",
            "Gleitende Profiltoleranz [%]",
            "Leistung [kW]",
            "Prozessabwärme [kW]",
        ]
        alarms_detail_filtered = alarms_detail[desired_columns]
        columns_to_round = [
            "Folienprofil 3-Sigma [%]",
            "Gleitende Profiltoleranz [%]",
            "Leistung [kW]",
            "Prozessabwärme [kW]",
        ]

        # Convert columns to numeric data type
        alarms_detail_filtered.loc[:, columns_to_round] = alarms_detail_filtered.loc[
            :, columns_to_round
        ].apply(pd.to_numeric, errors="coerce")

        alarms_detail_filtered.loc[:, columns_to_round] = alarms_detail_filtered.loc[
            :, columns_to_round
        ].round(2)

        return alarms_detail_filtered

        # Render the JobHistory table

    # Function to depict the last ten jobs of the dataset
    @output
    @render.table
    def JobHistory():
        # Load the data
        data = df

        # Create an empty DataFrame for the solution
        last_ten_jobs = pd.DataFrame(
            columns=["Produkt-Code", "Startzeitpunkt", "Endzeitpunkt"]
        )

        # Sort the data by local_time with the latest date on top, drop NaN
        # values, and convert data types
        sorted_df = data.sort_values("local_time", ascending=False)
        sorted_df = sorted_df.dropna(subset=["Produkt [None]", "local_time"])
        sorted_df["Produkt [None]"] = sorted_df["Produkt [None]"].astype(int)

        # Initialize variables for the first row
        current_pc = sorted_df.iloc[0]["Produkt [None]"]
        end_time = sorted_df.iloc[0]["local_time"]
        count = 0

        # Iterate over the rows
        for index, row in sorted_df.iterrows():
            # Check if the Produkt Code in the current row is different from
            # the previous row
            start_time = row["local_time"]
            if row["Produkt [None]"] != current_pc and count < 10:
                # Only consider products that ran longer than 5 minutes
                time_difference = end_time - start_time
                if time_difference.total_seconds() > 300:
                    new_row = [current_pc, start_time, end_time]
                    last_ten_jobs.loc[len(last_ten_jobs)] = new_row

                    current_pc = row["Produkt [None]"]
                    end_time = row["local_time"]

                    count += 1

            elif count >= 10:
                break

        # Format the datetime columns
        last_ten_jobs["Startzeitpunkt"] = last_ten_jobs["Startzeitpunkt"].dt.strftime(
            "%d.%m.%Y %H:%M"
        )
        last_ten_jobs["Endzeitpunkt"] = last_ten_jobs["Endzeitpunkt"].dt.strftime(
            "%d.%m.%Y %H:%M"
        )

        # Return the table
        return last_ten_jobs

    #####################################################################################
    # CUT OVERVIEW; START RECIPE ANALYSIS
    #####################################################################################

    # Define the auxiliary function to load quality data
    def load_quality_cluster_data():
        infile = Path(__file__).parent / "quality_cluster.csv"
        data = pd.read_csv(infile)
        return data

    # Define the auxiliary function to load cluster data
    def load_clusterdata():
        infile = Path(__file__).parent / "cluster_result_2023-06-28.csv"
        data = pd.read_csv(infile, sep=";")
        return data

    # Load corridor data
    def load_corridor_data():
        infile = Path(__file__).parent / "corridors_cluster.csv"
        data = pd.read_csv(infile)
        return data

    # Load cluster specs data
    def load_cluster_specs_data():
        infile = Path(__file__).parent / "cluster_5feature_avg.csv"
        data = pd.read_csv(infile)
        return data

    # Load quality distribution data
    def load_quality_dist_data():
        infile = Path(__file__).parent / "quality_dist.csv"
        data = pd.read_csv(infile)
        return data

    # Load datasets
    numeric_data = df
    quality_cluster_data = load_quality_cluster_data()
    cluster_data = load_clusterdata()
    corridor_data = load_corridor_data()
    cluster_avg = load_cluster_specs_data()
    quality_dist = load_quality_dist_data()

    # RECIPE ANALYSIS

    # Initialize the random forrest model
    def RA01():
        categorical_cols_pred = [
            "Extruder A HK Material [None]",
            "Extruder B HK Material [None]",
            "Extruder C HK Material [None]",
        ]
        numeric_cols_pred = [
            "Extruder A HK Anteil [%]",
            "Extruder B HK Anteil [%]",
            "Extruder C HK Anteil [%]",
        ]

        # Fit one-hot encoder
        # Load recipe_dataset to fit one-hot encoder
        infile1 = Path(__file__).parent / "6F_grouped_recipes.csv"
        recipes = pd.read_csv(infile1)

        ohe = OneHotEncoder(handle_unknown="ignore")
        ohe_fitted = ohe.fit(recipes[categorical_cols_pred])

        # Load Random forest model '''
        infile2 = Path(__file__).parent / "my_random_forest.joblib"
        loaded_rf = joblib.load(infile2)
        output = [ohe_fitted, loaded_rf, categorical_cols_pred, numeric_cols_pred]
        return output

    # Safe the result in a variable to decrease computation
    RA02_input = RA01()

    # Catch user input for extruders
    @reactive.event(input.submit, ignore_none=False)
    def user_input():
        input_dict = {
            "Extruder A HK Material [None]": str(input.Extruder01_material()),
            "Extruder A HK Anteil [%]": int(input.Extruder01_percentage()),
            "Extruder B HK Material [None]": str(input.Extruder02_material()),
            "Extruder B HK Anteil [%]": int(input.Extruder02_percentage()),
            "Extruder C HK Material [None]": str(input.Extruder03_material()),
            "Extruder C HK Anteil [%]": int(input.Extruder03_percentage()),
        }
        return input_dict

    # Take user input and find the existing product with the highest similarity
    @reactive.Calc
    def RA02_without_output():
        ohe_fitted = RA02_input[0]
        loaded_rf = RA02_input[1]
        categorical_cols_pred = RA02_input[2]
        numeric_cols_pred = RA02_input[3]
        input_dict = user_input()

        # TODO: use input_dict["Extruder A HK Anteil [%]"] = RshinyInput1...
        # to fill the input_dict with user input and then run prediction in second last line

        # Build a DataFrame from the input (not the most efficient way, but easiest to understand I guess)
        df = pd.DataFrame(input_dict, index=[0])

        # Fit OneHot Encoding to Prediction set
        Pred_Prep = ohe_fitted.transform(df[categorical_cols_pred])
        Pred_Prep = pd.DataFrame(Pred_Prep.toarray())

        # Recreate prediction input from numbers and categorical columns because both data types must be handled differently
        Pred_encoded = pd.concat(
            [df[numeric_cols_pred], Pred_Prep], axis=1, ignore_index=True
        )

        # Uses the RF model to predict a fitting product_code. This product code can then be used to find the
        # corresponding cluster and show the target values for this cluster.
        # It should predict 94165975, which is one of the known product_codes from the aforementioned cluster.
        output = loaded_rf.predict(Pred_encoded)
        return output[0]

    # Return the result of the RA02 function
    @output
    @render.text
    @reactive.Calc
    def RA02():
        return RA02_without_output()

    # CLUSTER ANALYSIS
    
    # Function that names the cluster of the product code
    @reactive.Calc
    def cluster_number_without_output():
        data = corridor_data
        cluster_number = data.loc[
            data["product_code [None]"] == RA02_without_output(), "Cluster"
        ].values[0]
        return cluster_number

    # Return the result of the cluster_number_without_output function
    @output
    @render.text
    @reactive.Calc
    def cluster_number_output():
        return cluster_number_without_output()

    # Funtion to depict cluster values 
    @output
    @render.table
    @reactive.Calc
    def cluster_specs():
        # Load data
        initial_data = cluster_avg

        # Filter for row of specific cluster
        filtered_data = initial_data[
            initial_data["Cluster"] == cluster_number_without_output()
        ].round(2)

        filtered_data.drop(columns="Cluster", inplace=True)

        # Custom sort columns
        desired_column_order = [
            "Folienprofil 3-Sigma [%]",
            "Gleitende Profiltoleranz [%]",
            "Leistung [kW]",
            "Prozessabwärme [kW]",
            # Add the remaining column names in the desired order
        ]
        filtered_data = filtered_data.reindex(columns=desired_column_order)

        return filtered_data

    # Function to depict relative cluster values
    @output
    @render.table
    @reactive.Calc
    def relative_cluster_values():
        # Load data and get variables
        data = cluster_avg
        cluster = cluster_number_without_output()

        # Get cluster averages
        leistung_avg = data["Leistung [kW]"].values.mean()
        prozessabwaerme_avg = data["Prozessabwärme [kW]"].values.mean()
        sigma_avg = data["Folienprofil 3-Sigma [%]"].values.mean()
        profiltoleranz_avg = data["Gleitende Profiltoleranz [%]"].values.mean()
        row = data[data["Cluster"] == cluster]

        # Calculate relative values of cluster to cluster averages

        leistung_diff = round(
            ((row["Leistung [kW]"] - leistung_avg) / row["Leistung [kW]"]) * 100, 2
        )
        leistung_diff_formatted = leistung_diff.apply(lambda x: "{:.2f}%".format(x))

        prozessabwaerme_diff = round(
            ((row["Prozessabwärme [kW]"] - prozessabwaerme_avg) / row["Leistung [kW]"])
            * 100,
            2,
        )
        prozessabwaerme_diff_formatted = prozessabwaerme_diff.apply(
            lambda x: "{:.2f}%".format(x)
        )

        sigma_diff = round(
            ((row["Folienprofil 3-Sigma [%]"] - sigma_avg) / row["Leistung [kW]"])
            * 100,
            2,
        )
        sigma_diff_formatted = sigma_diff.apply(lambda x: "{:.2f}%".format(x))

        profiltoleranz_diff = round(
            (
                (row["Gleitende Profiltoleranz [%]"] - profiltoleranz_avg)
                / row["Leistung [kW]"]
            )
            * 100,
            4,
        )
        profiltoleranz_diff_formatted = profiltoleranz_diff.apply(
            lambda x: "{:.4f}%".format(x)
        )

        result = pd.DataFrame(
            {
                "Leistung [kW]": leistung_diff_formatted,
                "Prozessabwärme [kW]": prozessabwaerme_diff_formatted,
                "Folienprofil 3-Sigma [%]": sigma_diff_formatted,
                "Gleitende Profiltoleranz [%]": profiltoleranz_diff_formatted,
            }
        )

        # Custom sort columns
        desired_column_order = [
            "Folienprofil 3-Sigma [%]",
            "Gleitende Profiltoleranz [%]",
            "Leistung [kW]",
            "Prozessabwärme [kW]",
            # Add the remaining column names in the desired order
        ]
        result = result.reindex(columns=desired_column_order)

        return result

    # Function to take the resulting product code from the RA and depict the cluster the new product falls into
    def generate_distinct_colors(num_colors):
        colors = []
        hue_values = [i / num_colors for i in range(num_colors)]
        saturation = 0.8
        value = 0.8

        for hue in hue_values:
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            rgb_int = [int(val * 255) for val in rgb]
            rgb_str = "#{:02x}{:02x}{:02x}".format(rgb_int[0], rgb_int[1], rgb_int[2])
            colors.append(rgb_str)

        return colors

    # Function to show the product cluster distribution
    @output
    @render.plot
    @reactive.event(input.submit, ignore_none=False)
    def product_cluster():
        data = cluster_data
        # Determine the number of unique clusters
        num_colors = data["Cluster"].nunique()

        # Generate distinct colors for the clusters
        cluster_colors = generate_distinct_colors(num_colors)

        # Create a ListedColormap with the cluster colors
        cmap = ListedColormap(cluster_colors)

        # Create a scatter plot with the custom colormap
        scatter = plt.scatter(data["X"], data["Y"], c=data["Cluster"], cmap=cmap)

        # Add labels to the data points
        for x, y, product, cluster in zip(
            data["X"],
            data["Y"],
            data["product_code [None]"],
            data["Cluster"],
        ):
            plt.text(x, y, f"{cluster}")

        # Set labels for the axes
        plt.xlabel("X")
        plt.ylabel("Y")

        # Get the unique clusters in sorted order
        sorted_clusters = sorted(data["Cluster"].unique())

        # Create legend handles for each cluster with respective color
        legend_handles = [
            Patch(facecolor=cluster_colors[i], label="Cluster {}".format(cluster))
            for i, cluster in enumerate(sorted_clusters)
        ]

        # Create a separate legend with all clusters
        legend = plt.legend(
            handles=legend_handles,
            fontsize="small",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
        )

        # Set the background color of the legend
        legend.get_frame().set_facecolor("none")

        # Set the figure's facecolor to transparent
        plt.gca().set_facecolor("none")

        # Set the figure's background color to transparent
        plt.gcf().set_facecolor("none")

        # Set the figure's edgecolor to transparent
        plt.gcf().set_edgecolor("none")

    # Auxiliary function for alarm-like corridors for cluster averages
    @output
    @render.table
    @reactive.Calc
    def corridors():
        # Load data
        initial_data = corridor_data

        # Determine features
        features = [
            "Leistung [kW]",
            "Prozessabwärme [kW]",
            "Folienprofil 3-Sigma [%]",
            "Gleitende Profiltoleranz [%]",
        ]

        cluster = cluster_number_without_output()
        # Extract only rows that match the cluster
        data = initial_data[initial_data["Cluster"] == cluster]

        # Extract only rows that match the features
        data = data[data["Column Name"].isin(features)]

        # Group the lower and upper range based on feature
        data = data.groupby(["Column Name"]).mean().reset_index()

        # Round columns to two digits
        data[["Lower Range", "Upper Range"]] = data[
            ["Lower Range", "Upper Range"]
        ].round(2)

        # Select columns
        data = data[["Column Name", "Lower Range", "Upper Range"]]
        data = data.rename(
            columns={
                "Column Name": "Kennzahl",
                "Lower Range": "Untere Grenze",
                "Upper Range": "Obere Grenze",
            }
        )

        return data

    # Function to depict prescriptive corridors for values with existing Soll-Values
    @output
    @render.table
    @reactive.Calc
    def corridors_soll():
        # Load data
        initial_data = corridor_data

        # Determine features
        features = [
            "Blaskopf Zone 1 Temperatur [°C]",
            "Blaskopf Zone 2 Temperatur [°C]",
            "Blaskopf Zone 3 Temperatur [°C]",
            "Blaskopf Zone 4 Temperatur [°C]",
            "Blaskopf Zone 5 Temperatur [°C]",
            "Blaskopf Zone 6 Temperatur [°C]",
            "Blaskopf Zone 7 Temperatur [°C]",
            "Blaskopf Zone 8 Temperatur [°C]",
            "Blaskopf Zone 9 Temperatur [°C]",
        ]

        cluster = cluster_number_without_output()
        # Extract only rows that match the cluster
        data = initial_data[initial_data["Cluster"] == cluster]

        # Extract only rows that match the features
        data = data[data["Column Name"].isin(features)]

        # Group the lower and upper range based on feature
        data = data.groupby(["Column Name"]).mean().reset_index()

        # Select columns
        data = data[["Column Name", "Lower Range", "Upper Range"]]
        data = data.rename(
            columns={
                "Column Name": "Kennzahl",
                "Lower Range": "Untere Grenze",
                "Upper Range": "Obere Grenze",
            }
        )

        return data

    # Functions to depict quality KPIS
    @output
    @render.text
    @reactive.Calc
    def quality_bahngeschwindigkeit():
        initial_data = quality_cluster_data[
            ["product_code [None]", "Cluster", "mean_bahngeschwindigkeit"]
        ]

        cluster = cluster_number_without_output()

        # Extract only rows that match the cluster
        data = initial_data[initial_data["Cluster"] == cluster]

        # Group the values based on cluster and get value
        data = data.groupby(["Cluster"]).mean().reset_index()
        value = str(data["mean_bahngeschwindigkeit"].iloc[0].round(2)) + " [m/min]"

        return value

    @output
    @render.text
    @reactive.Calc
    def quality_productCount():
        initial_data = quality_cluster_data[
            ["product_code [None]", "Cluster", "Product Count"]
        ]
        cluster = cluster_number_without_output()

        # Extract only rows that match the cluster
        data = initial_data[initial_data["Cluster"] == cluster]

        # Group the values based on cluster and get value
        data = data.groupby(["Cluster"]).mean().reset_index()
        # Round to integer
        value = round(data["Product Count"].iloc[0])

        return value

    @output
    @render.text
    @reactive.Calc
    def spikesCount():
        initial_data = quality_cluster_data[
            ["product_code [None]", "Cluster", "Spikes Count"]
        ]
        cluster = cluster_number_without_output()

        # Extract only rows that match the cluster
        data = initial_data[initial_data["Cluster"] == cluster]

        # Group the values based on cluster and get value
        data = data.groupby(["Cluster"]).mean().reset_index()
        value = round(data["Spikes Count"].iloc[0])

        return value

    @output
    @render.text
    @reactive.Calc
    def proportionSpikes():
        initial_data = quality_cluster_data[
            ["product_code [None]", "Cluster", "Proportion Spikes"]
        ]
        cluster = cluster_number_without_output()

        # Extract only rows that match the cluster
        data = initial_data[initial_data["Cluster"] == cluster]

        # Group the values based on cluster and get value
        data = data.groupby(["Cluster"]).mean().reset_index()
        # Round to integer
        value = str(data["Proportion Spikes"].iloc[0].round(2)) + " [%]"

        return value

    @output
    @render.text
    @reactive.Calc
    def totalWaste():
        initial_data = quality_cluster_data[
            ["product_code [None]", "Cluster", "Total Waste [m]"]
        ]
        cluster = cluster_number_without_output()

        # Extract only rows that match the cluster
        data = initial_data[initial_data["Cluster"] == cluster]

        # Group the values based on cluster and get value
        data = data.groupby(["Cluster"]).mean().reset_index()
        value = str(data["Total Waste [m]"].iloc[0].round(2)) + " [m]"

        return value

    @output
    @render.text
    @reactive.Calc
    def averageWaste():
        initial_data = quality_cluster_data[
            ["product_code [None]", "Cluster", "Average Waste per Job"]
        ]
        cluster = cluster_number_without_output()

        # Extract only rows that match the cluster
        data = initial_data[initial_data["Cluster"] == cluster]

        # Group the values based on cluster and get value
        data = data.groupby(["Cluster"]).mean().reset_index()
        value = str(data["Average Waste per Job"].iloc[0].round(2)) + " [%]"

        return value

    #################################################################################
    # QUALITY NAV
    #################################################################################

    # Functions to get time and feature
    @reactive.Calc
    def get_daterange2():
        start_date = pd.to_datetime(input.date_range2()[0])
        end_date = pd.to_datetime(input.date_range2()[1])
        dates = [start_date, end_date]
        return dates

    def get_time_and_feature(df, ft):
        sigma = df[["local_time", ft]].copy()
        sigma["local_time"] = pd.to_datetime(sigma["local_time"])

        # Define the specific time span
        start_time = get_daterange2()[0]
        end_time = get_daterange2()[1]

        # Select rows within the time span
        selected_rows = sigma[
            (sigma["local_time"] >= start_time) & (sigma["local_time"] <= end_time)
        ]
        return selected_rows

    @reactive.Calc
    def get_sigma():
        result = get_time_and_feature(numeric_data, "Folienprofil 3-Sigma [%]")
        return result

    # Functions to identify the spikes based on threshhold
    def identify_spikes(df, feature, th):
        df = get_sigma()
        # Calculate z-scores for the feature
        z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
        threshold = th
        # Identify spikes as values with z-score above the threshold
        spikes = df.loc[z_scores > threshold]
        return spikes

    @reactive.Calc
    def get_spikes():
        spikes = identify_spikes(get_sigma(), "Folienprofil 3-Sigma [%]", 3.1)
        return spikes

    # Function to plot spikes, input a df and the name of the feature e.g. ""Folienprofil 3-Sigma [%]"
    def plot_spikes(df, feature):
        spikes = get_spikes()

        df["local_time"] = pd.to_datetime(df["local_time"])

        plt.figure(figsize=(18, 6))

        # Plotting
        plt.plot(df["local_time"], df[feature], color=(0, 0.1098, 0.2902))
        plt.xlabel("Zeit")
        plt.ylabel(feature)
        plt.title(
            feature + " im Zeitverlauf",
            fontweight="bold",
            fontdict={"family": "Trebuchet MS", "size": 14},
        )
        plt.xticks(rotation=45)

        plt.scatter(
            spikes["local_time"], spikes[feature], c="red", marker="o", label="Spikes"
        )

        # Set the facecolor of the axes to transparent
        plt.gca().set_facecolor("none")

        # Set the facecolor of the figure to transparent
        plt.gcf().set_facecolor("none")

        # Set the edgecolor of the figure to transparent
        plt.gcf().set_edgecolor("none")

        # add line at critical value
        plt.axhline(y=8, color="green", linestyle="--", label="Kritisches Level von 8%")
        legend = plt.legend(frameon=True)

        # Format x-axis labels to "dd.mm.yyyy hh:mm"
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y %H:%M"))

        # Set the background color of the legend
        legend.get_frame().set_facecolor("none")

    # Function to plot the spike function based on the auxiliary functions above
    @output
    @render.plot
    # @reactive.Calc
    def spikes_plot():
        # plot spikes --> maybe induce this to the dashboard
        sigma = get_sigma()

        plot_spikes(sigma, "Folienprofil 3-Sigma [%]")

    # Function to show quality values by cluster
    @output
    @render.table
    def quality_values_by_cluster():
        initial_data = quality_cluster_data.drop(columns="product_code [None]")
        initial_data[
            ["mean_bahngeschwindigkeit", "Total Waste [m]", "Average Waste per Job"]
        ] = initial_data[
            ["mean_bahngeschwindigkeit", "Total Waste [m]", "Average Waste per Job"]
        ].round(
            2
        )
        initial_data[["Proportion Spikes"]] = initial_data[["Proportion Spikes"]].round(
            4
        )
        # list of selected clusters
        cluster_list = input.cluster()
        cluster = [int(cluster.split()[1]) for cluster in cluster_list]

        # Extract only rows that match the cluster
        data = initial_data[initial_data["Cluster"].isin(cluster)]

        # Group the values based on cluster and get value
        data = data.groupby(["Cluster"]).mean().reset_index()
        data = data.rename(
            columns={
                "mean_bahngeschwindigkeit": "Durchschnittliche Bahngeschwindigkeit [m/s]",
                "Product Count": "Anzahl Produkte [absolut]",
                "Spikes Count": "Anzahl Spikes [absolut]",
                "Proportion Spikes": "Anzahl Spikes [%]",
                "Total Waste [m]": "Totaler Ausschuss [m]",
                "Average Waste per Job": "Durchschnittlicher Ausschuss pro Job [m]",
            }
        )

        # Round all columns except the last two to two decimal places
        data.iloc[:, 1:] = data.iloc[:, 1:].round(2)

        return data
    # Function to show quality cluster distribution
    @output
    @render.table
    def quality_distribution_tbl():
        data = quality_dist

        # Replace grade names
        grade_translations = {
            "Excellence": "Exzellent",
            "Good": "Gut",
            "Average": "Durchschnittlich",
            "Critical": "Kritisch",
            "Waste": "Abfall",
        }

        data["Grade"].replace(grade_translations, inplace=True)

        # Rename Columns
        data = data.rename(
            columns={"Grade": "Bewertung", "Amount": "Anzahl", "Percentage": "Anteil"}
        )

        # Drop first column
        data = data.drop(data.columns[0], axis=1)

        return data
    # Function to plot quality cluster distribution
    @output
    @render.plot
    def quality_distribution_plt():
        data = quality_dist

        # Calculate the percentage column
        total_amount = data.loc[data["Grade"] == "Total", "Amount"].values[0]
        data["Percentage"] = data["Amount"] / total_amount * 100

        # Replace grade names
        grade_translations = {
            "Excellence": "Exzellent",
            "Good": "Gut",
            "Average": "Durchschnittlich",
            "Critical": "Kritisch",
            "Waste": "Abfall",
        }

        data["Grade"].replace(grade_translations, inplace=True)

        # Extract the relevant columns
        grades = data.loc[data["Grade"] != "Total", "Grade"]
        percentages = data.loc[data["Grade"] != "Total", "Percentage"]

        # Plot the histogram
        fig = plt.figure(facecolor="none")
        ax = fig.add_subplot(111)
        ax.bar(grades, percentages, color=(0, 0.1098, 0.2902))

        # Customize the plot
        ax.set_xlabel("Wertung")
        ax.set_ylabel("Prozentanteil (%)")
        ax.set_title(
            "Verteilung der Qualitätseinstufungen",
            fontweight="bold",
            fontdict={"family": "Trebuchet MS", "size": 14},
        )
        ax.tick_params(rotation=45)

        # Remove upper and right boundaries
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Make inside of graph transparent
        ax.set_facecolor((0, 0, 0, 0))

# Shiny call to combine the two components and depict web-app
app = App(app_ui, server)
