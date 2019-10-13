for i in range(9,21):
    cluster0 = groupedDf.get_group(i)
    cluster0['Tanggal'] = cluster0['Tanggal'].str.replace('-','')
    cluster0 = cluster0[cluster0.Tanggal !=0]
    cluster0['Tanggal'] = pd.to_datetime(cluster0['Tanggal'], format="%Y%m%d",errors="coerce")
    cluster0=cluster0.sort_values(by ='Tanggal')
    exportToCsv0 = cluster0.groupby([cluster0.Tanggal.dt.year, cluster0.Tanggal.dt.month]).agg({'count'})
    exportToCsv0.to_csv("cluster_"+i+"_Arima.csv")
