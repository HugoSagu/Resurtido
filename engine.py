from typing import Dict
from datetime import datetime
import numpy as np
import pandas as pd

class ReplenishmentEngine:
    def __init__(
        self,
        lead_time_days: int = 3,
        review_frequency_days: int = 7,
        extra_coverage_days: int = 2,
        min_cv: float = 0.50,
    ):
        self.lt = lead_time_days
        self.r = review_frequency_days
        self.cov_extra = extra_coverage_days
        self.min_cv = min_cv

        self.z_table: Dict[str, float] = {
            "A": 2.05,
            "B": 1.65,
            "C": 1.28,
        }

    def compute_daily_demand(self, df_sales: pd.DataFrame, recent_days_window: int = 60) -> pd.DataFrame:
        sales = df_sales.copy()
        sales["fecha"] = pd.to_datetime(sales["fecha"], errors="coerce")
        sales = sales.dropna(subset=["fecha", "Establecimiento", "Código Mercancía"])

        max_date = sales["fecha"].max()
        cutoff_date = max_date - pd.Timedelta(days=recent_days_window)
        recent = sales[sales["fecha"] >= cutoff_date]

        daily = recent.groupby(["Establecimiento", "Código Mercancía", "fecha"])["Cant. Venta"].sum().reset_index()

        metrics = daily.groupby(["Establecimiento", "Código Mercancía"]).agg(
            Venta_Total=("Cant. Venta", "sum"),
            Sigma_Empirica=("Cant. Venta", "std")
        ).reset_index()

        metrics["Demanda_Diaria_Base"] = metrics["Venta_Total"] / float(recent_days_window)
        metrics["Sigma_Empirica"] = metrics["Sigma_Empirica"].fillna(0.0)

        return metrics[["Establecimiento", "Código Mercancía", "Demanda_Diaria_Base", "Sigma_Empirica"]]

    def compute_pareto(self, df_data: pd.DataFrame) -> pd.DataFrame:
        df = df_data.copy()
        metric_col = "Demanda_Diaria_Base" if "Demanda_Diaria_Base" in df.columns else "Cant. Venta"

        sku_vol = df.groupby(["Establecimiento", "Nivel 2", "Código Mercancía"])[metric_col].sum().reset_index()
        sku_vol = sku_vol.sort_values(by=["Establecimiento", "Nivel 2", metric_col], ascending=[True, True, False])

        sku_vol["total_cat"] = sku_vol.groupby(["Establecimiento", "Nivel 2"])[metric_col].transform("sum")
        sku_vol["pct_share"] = np.where(sku_vol["total_cat"] > 0, sku_vol[metric_col] / sku_vol["total_cat"], 0.0)
        sku_vol["pct_acum"] = sku_vol.groupby(["Establecimiento", "Nivel 2"])["pct_share"].cumsum()

        conditions = [sku_vol["pct_acum"] <= 0.80, sku_vol["pct_acum"] <= 0.95]
        choices = ["A", "B"]
        sku_vol["Clase_ABC"] = np.select(conditions, choices, default="C")

        return sku_vol[["Establecimiento", "Código Mercancía", "Clase_ABC"]]

    def execute_replenishment(self, df_master: pd.DataFrame, mes_objetivo: int = None) -> pd.DataFrame:
        df = df_master.copy()
        df["Factor_Empaque"] = 1
        df["Transito"] = 0

        factor_est = df.get("Factor_Estacional", 1.0)
        df["Demanda_Ajustada"] = (df["Demanda_Diaria_Base"] * factor_est).clip(lower=0.0)

        sigma_piso = df["Demanda_Ajustada"] * self.min_cv
        sigma_base = df.get("Sigma_Empirica", 0.0)
        df["Sigma_D"] = np.maximum(sigma_base, sigma_piso)

        df["Factor_Z"] = df["Clase_ABC"].map(self.z_table).fillna(1.28)
        df["SS"] = np.ceil(df["Factor_Z"] * np.sqrt(self.lt) * df["Sigma_D"])

        df["MIN"] = np.ceil((df["Demanda_Ajustada"] * self.lt) + df["SS"])

        horizonte_max = self.r + self.cov_extra
        df["MAX"] = np.ceil(df["MIN"] + (df["Demanda_Ajustada"] * horizonte_max))

        df["Existencia"] = pd.to_numeric(df["Existencia"], errors="coerce").fillna(0.0)
        df["Gatillo_Reorden"] = (df["Existencia"] + df["Transito"]) <= df["MIN"]

        df["Necesidad_Pzas"] = np.where(
            df["Gatillo_Reorden"],
            np.maximum(0, df["MAX"] - (df["Existencia"] + df["Transito"])),
            0.0
        )
        df["Unidades_Sugeridas"] = np.ceil(df["Necesidad_Pzas"]).astype(int)

        condiciones = [
            df["Existencia"] <= 0,
            df["Existencia"] <= df["SS"],
            df["Existencia"] <= df["MIN"],
            df["Existencia"] > df["MAX"],
        ]
        etiquetas = ["AGOTADO (Stockout)", "CRÍTICO (Bajo SS)", "REORDENAR (Bajo MIN)", "SOBRESTOCK"]
        df["Estado_Inventario"] = np.select(condiciones, etiquetas, default="ÓPTIMO")

        return df
