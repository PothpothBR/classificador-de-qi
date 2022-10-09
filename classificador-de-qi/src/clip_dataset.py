import pandas
import json
def clipDataset(data: pandas.DataFrame) -> pandas.DataFrame:
    """padroniza o dataset para utilizar

    Args:
        data (pandas.DataFrame): data frame

    Returns:
        pandas.DataFrame: data frame clipado
    """
    # padroniza nomes
    data["Shrt_Desc"] = data["Shrt_Desc"].map(lambda n: n.split(',')[0].split(' ')[0].split("\'")[0])

    # agrupa por nome da comida
    data_g = {}
    for name in data["Shrt_Desc"]:
        if name not in data_g.keys():
            data_g.update({name: 1})
        else:
            data_g[name] += 1
    
    # ordena do maior para o menor
    data_g = sorted(data_g.items(), key=lambda a: a[1], reverse=True)

    # filta pelos 2 maiores conjuntos
    # filtra comidas pouco atrentes para a classificação, como nomes de empresas, bebidas, nomes genéricos (ex: Molho).
    data_names = []
    i = 0
    for val, w in data_g:
        if val not in (
                "RESTAURANT", "MCDONALD", "GAME", "INF",
                "BABYFOOD", "MORNINGSTAR", "FAST", "WORTHINGTON",
                "KEEBLER", "BEVERAGES", "KASHI", "CAMPBELL",
                "KELLOGG", "ALCOHOLIC", "OIL", "ICE", "USDA",
                "TURKEY", "SNACKS", "SAUCE", "MURRAY", "BF"
            ):
            data_names.append(val)
            i+=1
            if i >=2: break

    # remove linhas que não fazem parte dos filtros
    for i, name in zip(data["index"], data["Shrt_Desc"]):
        if name not in data_names:
            data.drop(i, axis=0, inplace=True)

    # remove colunas pouco atraentes para classificação
    data.drop([
            "Calcium_(mg)","Iron_(mg)","Magnesium_(mg)","Phosphorus_(mg)","Potassium_(mg)",
            "Sodium_(mg)","Zinc_(mg)","Copper_(mg)","Manganese_(mg)","Selenium_(µg)","Vit_C_(mg)",
            "Thiamin_(mg)","Riboflavin_(mg)","Niacin_(mg)","Panto_Acid_mg)","Vit_B6_(mg)",
            "Folate_Tot_(µg)","Folic_Acid_(µg)","Food_Folate_(µg)","Folate_DFE_(µg)","Choline_Tot_ (mg)",
            "Vit_B12_(µg)","Vit_A_IU","Vit_A_RAE","Retinol_(µg)","Alpha_Carot_(µg)","Beta_Carot_(µg)",
            "Beta_Crypt_(µg)","Lycopene_(µg)","Lut+Zea_ (µg)","Vit_E_(mg)","Vit_D_µg","Vit_D_IU",
            "Vit_K_(µg)","FA_Sat_(g)","FA_Mono_(g)","FA_Poly_(g)","Cholestrl_(mg)","GmWt_1",
            "GmWt_Desc1","GmWt_2","GmWt_Desc2","Refuse_Pct","index","NDB_No"],
        axis=1, inplace=True
    )

    # salva o novo dataset
    data.fillna(0).to_csv("./res/clipped.csv")
    
    # agrupa por nome da comida
    data_g = {}
    for name in data["Shrt_Desc"]:
        if name not in data_g.keys():
            data_g.update({name: 1})
        else:
            data_g[name] += 1
            
    with open("./res/info.json", 'w') as file:
        file.write(json.dumps(data_g))
    
    return data