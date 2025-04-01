import streamlit as st
import pandas as pd
import requests
import os
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Функція завантаження даних
def download_files():
    df_all = pd.DataFrame()
    for ids in range(1, 28):
        url = f"https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?country=UKR&provinceID={ids}&year1=1981&year2=2024&type=Mean"
        response = requests.get(url)
        if response.status_code == 200:
            if not os.path.exists('vhi'):
                os.mkdir('vhi')
                st.write('The folder is created')
            date_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = fr'vhi\vhi_id_{ids}_{date_now}.csv'
            with open(file_name, 'wb') as out:
                out.write(response.content)
            st.write(f"VHI from id {ids} was downloaded at {date_now}")
            headers = ['Year', 'Week', 'SMN', 'SMT', 'VCI', 'TCI', 'VHI', 'empty']
            df = pd.read_csv(file_name, header=1, names=headers, skiprows=1)[:-1]
            df = df.drop(df.loc[df['VHI'] == -1].index)
            df['area'] = int(file_name.split("_")[2])
            df = df.drop(columns=['empty'])
            df_all = pd.concat([df_all, df]).drop_duplicates().reset_index(drop=True)
        else:
            st.error('Помилка завантаження')
            break
    df_all.to_csv(r'vhi\df_all.csv', index=False)

# Функція зчитування даних
def read_csv_data(file_name):
    headers = ['Year', 'Week', 'SMN', 'SMT', 'VCI', 'TCI', 'VHI', 'area']
    df_all = pd.read_csv(file_name, header=1, names=headers, delimiter=',')
    df_all['Year'] = df_all['Year'].astype(int)
    df_all['Week'] = df_all['Week'].astype(int)
    df_all['area'] = df_all['area'].astype(int)
    return df_all

# Якщо дані не існують, завантажуємо їх
data_path = r'vhi\df_all.csv'
if not os.path.exists(data_path):
    download_files()
df = read_csv_data(data_path)

# 2. Словник відповідності областей
region_dict = {
    "Cherkasy": 1,
    "Chernihiv": 2,
    "Chernivtsi": 3,
    "Crimea": 4,
    "Dnipro": 5,
    "Donetsk": 6,
    "Ivano-Frankivsk": 7,
    "Kharkiv": 8,
    "Kherson": 9,
    "Khmelnytskyy": 10,
    "Kiev": 11,
    "Kiev City": 12,
    "Kirovohrad": 13,
    "Luhansk": 14,
    "Lviv": 15,
    "Mykolayiv": 16,
    "Odessa": 17,
    "Poltava": 18,
    "Rivne": 19,
    "Sevastopol": 20,
    "Sumy": 21,
    "Ternopil": 22,
    "Transcarpathia": 23,
    "Vinnytsya": 24,
    "Volyn": 25,
    "Zaporizhzhya": 26,
    "Zhytomyr": 27
}

# 9. Розбиття сторінки на дві колонки
left_col, right_col = st.columns([1, 3])

with left_col:
    st.header("Фільтри")
    # 1. Dropdown для вибору часового ряду
    selected_index = st.selectbox("Оберіть часовий ряд:", options=["VCI", "TCI", "VHI"], index=0)
    
    # 2. Dropdown для вибору області
    region_names = list(region_dict.keys())
    selected_region_name = st.selectbox("Оберіть область:", options=region_names, index=region_names.index("Kiev City"))
    selected_region = region_dict[selected_region_name]
    
    # 3. Slider для інтервалу тижнів
    week_range = st.slider("Інтервал тижнів:", min_value=1, max_value=52, value=(1, 52))
    
    # 4. Slider для інтервалу років
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    year_range = st.slider("Інтервал років:", min_value=min_year, max_value=max_year, value=(2000, 2000), step=1)
    
    # 8. Checkbox для сортування за зростанням/спаданням
    sort_asc = st.checkbox("Сортувати за зростанням")
    sort_desc = st.checkbox("Сортувати за спаданням")
    if sort_asc and sort_desc:
        st.warning("Обидва варіанти сортування увімкнено. Сортування не застосовано.")
    
    # 5. Button для скидання фільтрів до початкових значень
    if st.button("Скинути фільтри"):
        st.experimental_rerun()

# 6. Фільтрація даних для обраного регіону, року та тижня
df_filtered = df[
    (df['area'] == selected_region) &
    (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
    (df['Week'] >= week_range[0]) & (df['Week'] <= week_range[1])
]

# 8. Сортування даних за вибраним індексом 
if sort_asc and not sort_desc:
    df_filtered = df_filtered.sort_values(by=selected_index, ascending=True)
elif sort_desc and not sort_asc:
    df_filtered = df_filtered.sort_values(by=selected_index, ascending=False)

with right_col:
    # 6. Вкладки для графіків і таблиці
    tab1, tab2, tab3 = st.tabs(["Часовий ряд", "Таблиця", "Порівняння"])
    
    with tab1:
        # 7. Перший графік: часовий ряд для обраного регіону за вказаним діапазоном років і тижнів
        st.subheader(f"Часовий ряд {selected_index} для {selected_region_name}")
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_filtered, x='Week', y=selected_index, marker='o')
        plt.title(f"{selected_index} ({year_range[0]}-{year_range[1]})")
        plt.xlabel("Тиждень")
        plt.ylabel(selected_index)
        st.pyplot(plt.gcf())
        plt.clf()
    
    with tab2:
        # 6. Таблиця з відфільтрованими даними
        st.subheader("Відфільтровані дані")
        st.dataframe(df_filtered)
    
    with tab3:
        # 7. Другий графік: порівняння значень обраного індексу між усіма областями
        st.subheader(f"Порівняльний аналіз {selected_index} по областях")
        df_compare = df[
            (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
            (df['Week'] >= week_range[0]) & (df['Week'] <= week_range[1])
        ]
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_compare, x='area', y=selected_index)
        plt.title(f"Порівняння {selected_index} за ({year_range[0]}-{year_range[1]})")
        plt.xlabel("Область (ID)")
        plt.ylabel(selected_index)
        st.pyplot(plt.gcf())
        plt.clf()

st.success("Дані завантажено та оброблено!")
