import sys

def cold_start_proxy(df_features, df_inventario, min_meses=6):
    conteo = df_features.groupby(COL_CN).size().reset_index(name="n_meses")
    cns_nuevos = set(conteo[conteo["n_meses"] < min_meses][COL_CN])
    if not cns_nuevos or COL_MOLECULA not in df_inventario.columns:
        return df_features
    mol_map = df_inventario.set_index(COL_CN).get(COL_MOLECULA, pd.Series(dtype=str)).to_dict()
    df_features["_mol_proxy"] = df_features[COL_CN].map(mol_map)
    df_maduros = df_features[~df_features[COL_CN].isin(cns_nuevos)]
    if df_maduros.empty:
        df_features = df_features.drop(columns=["_mol_proxy"])
        return df_features
    media_mol = df_maduros.groupby("_mol_proxy")[COL_VENTAS].mean().to_dict()
    mask_nuevo = df_features[COL_CN].isin(cns_nuevos)
    df_features.loc[mask_nuevo, COL_VENTAS] = df_features.loc[mask_nuevo, "_mol_proxy"].map(media_mol).fillna(
        df_features.loc[mask_nuevo, COL_VENTAS])
    df_features = df_features.drop(columns=["_mol_proxy"])
    return df_features
